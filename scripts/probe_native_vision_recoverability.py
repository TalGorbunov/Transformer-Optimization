"""Harvest frozen native h/r count-probe features; all execution requires Slurm.

No interventions, generated answers, or trained checkpoints are used. The CPU
companion fits probes on training rows and selects regularization on dev only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA_BASE = Path('/mnt/data/gabriele/gnn_transformer')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('This diagnostic requires Slurm; no model/data work on login nodes')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


class LastRowCapture:
    """Hooks preserve native attention; capture exactly one row at each site."""

    def __init__(self, attention):
        self.rows = {}
        self.handles = [attention.register_forward_pre_hook(self._h, with_kwargs=True),
                        attention.o_proj.register_forward_pre_hook(self._r)]

    def _save(self, key, value):
        if key in self.rows or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f'Expected one batched full-prefill capture for {key}')
        self.rows[key] = value[0, -1].detach().float().cpu().clone()

    def _h(self, module, args, kwargs):
        self._save('h', kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0])

    def _r(self, module, args):
        self._save('r', args[0])

    def take(self):
        if set(self.rows) != {'h', 'r'}:
            raise ValueError(f'Missing native captures: {set(self.rows)}')
        result, self.rows = self.rows, {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def manifest_records(main_path, count_path):
    """Exact registered order, no sample filtering/truncation; metadata only."""
    from scripts.native_aggregation_vlm_v2 import sample_metadata
    manifests, records, seen = [], [], set()
    expected = [(main_path, [('train_N8', 90), ('train_N16', 90), ('dev_N8', 36),
                            ('dev_N16', 36), ('test_N16', 108), ('test_N32', 108),
                            ('test_N64', 108)]),
                (count_path, [('test_N32', 64), ('test_N64', 64)])]
    common_root = None
    for manifest_index, (path, cells) in enumerate(expected):
        raw = path.read_bytes()
        manifest = json.loads(raw)
        root = Path(manifest['dataset_root']).resolve()
        if manifest.get('schema_version') != 1 or not root.is_relative_to(DATA_BASE):
            raise ValueError('Invalid staged manifest schema or data root')
        if common_root is not None and common_root != root:
            raise ValueError('Main/count manifests have different data roots')
        common_root = root
        if set(manifest['splits']) != {key for key, _ in cells}:
            raise ValueError('Unexpected manifest cells; this diagnostic is fixed to V2 main/count')
        manifests.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        for key, expected_count in cells:
            declared = manifest['splits'][key]['samples']
            if len(declared) != expected_count:
                raise ValueError(f'{key}: expected exactly {expected_count} rows')
            split, n = key.split('_N')
            n = int(n)
            for item in declared:
                directory = Path(item['path']).resolve()
                if directory.parent != root / 'mmred_vfiltered' / f'seq_len_{n}' / split:
                    raise ValueError('Sample outside declared manifest split')
                if str(directory) in seen:
                    raise ValueError('Duplicate sample across manifest cells')
                seen.add(str(directory))
                metadata = sample_metadata(directory, n)
                for field in ('sid', 'gold', 'qa_sha256', 'n_frames'):
                    if metadata[field] != item[field]:
                        raise ValueError(f'Manifest mismatch for {directory}: {field}')
                if len(item['image_files']) != n:
                    raise ValueError('Wrong image-file count')
                for actual, staged in zip(metadata['image_files'], item['image_files']):
                    if Path(actual['path']).resolve() != Path(staged['path']).resolve() or actual['bytes'] != staged['bytes']:
                        raise ValueError('Staged image path/size mismatch')
                    if len(staged.get('sha256', '')) != 64:
                        raise ValueError('Missing staging image hash')
                row = dict(item)
                row.update(question=metadata['question'], row_index=len(records),
                           split=split, cell=('count_N' + str(n)) if manifest_index else key,
                           family='unseen_count' if manifest_index else ('familiar_count' if split == 'test' else split))
                records.append(row)
    if len(records) != 704:
        raise AssertionError('Registered diagnostic requires exactly 704 rows')
    return records, manifests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--count-manifest', type=Path, required=True)
    p.add_argument('--data-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--layer-index', type=int, default=14)
    p.add_argument('--resize', type=int, default=392)
    p.add_argument('--max-seq-tokens', type=int, default=16000)
    a = p.parse_args()
    require_slurm()
    for path in (a.manifest, a.count_manifest, a.data_output):
        if not path.resolve().is_relative_to(DATA_BASE):
            raise SystemExit(f'Manifests/features must reside under {DATA_BASE}')
    if a.layer_index != 14 or a.resize != 392:
        raise SystemExit('This preregistered probe fixes layer_index=14 and resize=392')
    import numpy as np
    import torch
    from transformers import __version__ as transformers_version
    from gnnformer.data import build_count_prompt, build_prompt_inputs, load_mmred_sample
    from gnnformer.runtime import get_layers, load_runtime, move_to_device

    if not torch.cuda.is_available():
        raise SystemExit('Feature harvesting requires a Slurm GPU allocation')
    torch.set_num_threads(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))))
    started = time.time()
    records, manifests = manifest_records(a.manifest, a.count_manifest)
    a.data_output.mkdir(parents=True, exist_ok=False)
    report = a.output / f'harvest_{os.environ["SLURM_JOB_ID"]}'
    report.mkdir(parents=True, exist_ok=False)
    code = report / 'code'
    code.mkdir()
    names = ['scripts/probe_native_vision_recoverability.py', 'scripts/fit_native_vision_recoverability.py',
             'scripts/native_aggregation_vlm_v2.py', 'gnnformer/data.py', 'gnnformer/runtime.py']
    hashes = {}
    for name in names:
        payload = (REPO / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (code / name.replace('/', '_')).write_bytes(payload)
    config = dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'],
                  args={key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()},
                  manifests=manifests, n_samples=len(records), code_sha256=hashes,
                  torch_version=str(torch.__version__), transformers_version=transformers_version,
                  quantization='nf4, double quantization, bf16 compute',
                  image_verification='Staging SHA256 retained; runtime path/byte-size and QA SHA256 validated',
                  h='Final prompt token at input to attention, after input_layernorm',
                  r='Final prompt token concatenated SDPA heads, before o_proj',
                  prompt='Canonical build_count_prompt; images first; native processor chat template',
                  intervention='None; fully frozen backbone, no branch/carrier/LoRA/cache reuse',
                  interpretation='Linear accessibility diagnostic; no native-output method or proof of information absence')
    write_json(report / 'config.json', config)
    print(f'Harvesting {len(records)} samples into {a.data_output}', flush=True)
    runtime = load_runtime(a.model, use_4bit=True, attn_implementation='sdpa', device_map='cuda')
    model = runtime.model
    count_token_ids = [runtime.tokenizer.encode(str(k), add_special_tokens=False) for k in range(10)]
    if any(len(ids) != 1 for ids in count_token_ids):
        raise ValueError('Expected single-token numerals 0..9 for first-token diagnostic')
    count_token_ids = [ids[0] for ids in count_token_ids]
    count_index = torch.tensor(count_token_ids, dtype=torch.long, device=runtime.device)
    config['native_first_token_diagnostic'] = dict(count_token_ids=count_token_ids,
        note='Top-1 first token and raw logits only; not full-answer generation or count exactness, especially for K>=10')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = LastRowCapture(get_layers(model)[a.layer_index].self_attn)
    hs, rs = [], []
    try:
        for index, record in enumerate(records):
            sid, original, question, _states, answer = load_mmred_sample(Path(record['path']))
            if sid != record['sid'] or question != record['question'] or int(answer) != record['gold']:
                raise ValueError('Source QA changed after validation')
            resized = []
            try:
                resized = [frame.resize((a.resize, a.resize)) for frame in original]
                inputs = build_prompt_inputs(runtime.processor, resized, build_count_prompt(question, len(resized)))
            finally:
                for frame in original + resized:
                    frame.close()
            length = inputs['input_ids'].shape[1]
            if length > a.max_seq_tokens or len(original) != record['n_frames']:
                raise ValueError('Unexpected prompt/frame count; truncation forbidden')
            grids = inputs.get('image_grid_thw')
            if grids is None or grids.shape[0] != record['n_frames']:
                raise ValueError('Processor did not preserve image count')
            record['prompt_tokens'] = int(length)
            record['image_tokens'] = int((inputs['input_ids'] == model.config.image_token_id).sum())
            record['image_grid_thw'] = grids.tolist()
            record['input_ids_sha256'] = hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest()
            with torch.inference_mode():
                outputs = model(**move_to_device(inputs, runtime.device), use_cache=False, logits_to_keep=1, return_dict=True)
            features = capture.take()
            first_logits = outputs.logits[0, -1]
            first_id = int(first_logits.argmax())
            record['native_first_token'] = dict(token_id=first_id,
                token_text=runtime.tokenizer.decode([first_id]),
                count_logits=first_logits.index_select(0, count_index).detach().float().cpu().tolist())
            del first_logits, outputs, inputs
            if features['h'].shape != (3584,) or features['r'].shape != (3584,):
                raise ValueError('Unexpected feature width for registered Qwen2.5-VL-7B probe')
            if not all(torch.isfinite(vector).all() for vector in features.values()):
                raise ValueError('Nonfinite feature capture')
            hs.append(features['h'].numpy())
            rs.append(features['r'].numpy())
            if (index + 1) % 20 == 0 or index + 1 == len(records):
                print(f'{index + 1}/{len(records)} rows; elapsed {time.time() - started:.1f}s', flush=True)
    finally:
        capture.close()
    feature_path = a.data_output / 'features.npz'
    np.savez(feature_path, h=np.stack(hs).astype(np.float32), r=np.stack(rs).astype(np.float32))
    write_json(a.data_output / 'records.json', records)
    for label, source in [('main_manifest', a.manifest), ('count_manifest', a.count_manifest)]:
        (a.data_output / f'{label}.json').write_bytes(source.read_bytes())
    config.update(elapsed_seconds=time.time() - started, gpu=torch.cuda.get_device_name(0),
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  feature_file=str(feature_path), feature_sha256=sha256(feature_path),
                  records_sha256=sha256(a.data_output / 'records.json'),
                  feature_shape=[len(records), 3584], feature_dtype='float32', complete=True)
    write_json(report / 'config.json', config)
    write_json(a.data_output / 'harvest.json', config)
    print(json.dumps(dict(feature_file=str(feature_path), report=str(report), elapsed_seconds=config['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
