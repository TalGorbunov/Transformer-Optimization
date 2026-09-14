"""Stage an offline Cosmos tokenizer compatibility mirror on Slurm CPU only.

No downloads, weight loads, installs, or modifications to source snapshots.
Derive slow-tokenizer vocab/merges from existing ordinary byte-level BPE.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/mnt/home/gabriele.serussi/.cache/huggingface/hub/models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423')
DEST_BASE = Path('/mnt/ckpts/gabriele/gnn_transformer')
MANIFEST = Path('/mnt/data/gabriele/gnn_transformer/v2_clean/profile_manifest.json')
METADATA = ('config.json', 'generation_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'preprocessor_config.json',
            'chat_template.json', 'model.safetensors.index.json', 'README.md')


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_cpu_job():
    job = os.environ.get('SLURM_JOB_ID', '')
    ensure(re.fullmatch(r'[0-9]+', job), 'Run through Slurm CPU, not a login node')
    ensure(os.environ.get('SLURM_JOB_PARTITION') == 'cpu', 'Requires cpu partition')
    ensure(not os.environ.get('SLURM_JOB_GPUS', ''), 'No GPU allocation for staging')
    return job


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def validate_bpe(tokenizer):
    model = tokenizer['model']
    ensure(model['type'] == 'BPE', 'Only ordinary byte-level BPE is supported')
    for key, expected in (('dropout', None), ('unk_token', None),
                          ('continuing_subword_prefix', ''), ('end_of_word_suffix', ''),
                          ('fuse_unk', False), ('byte_fallback', False)):
        ensure(model.get(key) == expected, f'Unsupported BPE setting {key}')
    ensure(not model.get('ignore_merges', False), 'ignore_merges is unsupported')
    ensure(tokenizer.get('normalizer') == {'type': 'NFC'}, 'Expected Qwen NFC normalization')
    sequence = tokenizer['pre_tokenizer']
    ensure(sequence['type'] == 'Sequence', 'Expected native pretokenizer sequence')
    split, bytelevel = sequence['pretokenizers']
    ensure(split['type'] == 'Split' and split['behavior'] == 'Isolated'
           and split['invert'] is False, 'Unexpected pretokenizer split')
    expected_bytelevel = dict(type='ByteLevel', add_prefix_space=False,
                              trim_offsets=False, use_regex=False)
    ensure(bytelevel == expected_bytelevel, 'Unexpected byte-level pretokenizer')
    ensure(tokenizer['decoder'] == bytelevel, 'Unexpected byte decoder')
    ensure(tokenizer['post_processor'] == bytelevel, 'Unexpected postprocessor')
    vocab = model['vocab']
    ensure(all(isinstance(token, str) and type(index) is int for token, index in vocab.items()),
           'Invalid vocabulary')
    ensure(sorted(vocab.values()) == list(range(len(vocab))), 'Non-contiguous base vocabulary')
    merges = []
    for entry in model['merges']:
        pair = entry.split(' ') if isinstance(entry, str) else entry
        ensure(isinstance(pair, list) and len(pair) == 2
               and all(isinstance(token, str) and token and not any(c.isspace() for c in token)
                       for token in pair), 'Merge cannot be represented exactly in merges.txt')
        ensure(all(token in vocab for token in pair) and ''.join(pair) in vocab,
               'Merge refers outside base vocabulary')
        merges.append(tuple(pair))
    ensure(len(set(merges)) == len(merges), 'Duplicate merge rank')
    return vocab, merges, split['pattern']['Regex']


def inspect_shards(source, mirror, index):
    mapping = index['weight_map']
    names = sorted(set(mapping.values()))
    expected = [f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)]
    ensure(names == expected, 'Expected four canonical Cosmos shards')
    seen, shards, tensor_bytes = set(), [], 0
    widths = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8,
              'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    for name in names:
        path = source / name
        ensure(path.is_file(), f'Missing source shard: {name}')
        with path.open('rb') as handle:
            prefix = handle.read(8)
            ensure(len(prefix) == 8, 'Truncated safetensors header')
            size = struct.unpack('<Q', prefix)[0]
            ensure(0 < size <= 64 * 1024 * 1024, 'Unexpected safetensors header size')
            raw = handle.read(size)
        ensure(len(raw) == size, 'Truncated safetensors metadata')
        header = json.loads(raw)
        entries = {key: value for key, value in header.items() if key != '__metadata__'}
        offset = 0
        for key, entry in sorted(entries.items(), key=lambda item: item[1]['data_offsets'][0]):
            begin, end = entry['data_offsets']
            ensure(key not in seen and mapping.get(key) == name, 'Weight index/header mismatch')
            ensure(entry['dtype'] in widths, 'Unsupported safetensors dtype')
            expected_bytes = math.prod(entry['shape']) * widths[entry['dtype']]
            ensure(begin == offset and end - begin == expected_bytes, 'Unexpected tensor byte ranges')
            offset = end
            seen.add(key)
        ensure(path.stat().st_size == 8 + size + offset, 'Shard byte size mismatch')
        tensor_bytes += offset
        target = path.resolve(strict=True)
        (mirror / name).symlink_to(target)
        shards.append(dict(name=name, source=str(path), target=str(target),
                           bytes=path.stat().st_size, tensors=len(entries),
                           header_sha256=hashlib.sha256(raw).hexdigest()))
    ensure(seen == set(mapping), 'Some indexed weights are missing')
    ensure(tensor_bytes == index['metadata']['total_size'], 'Weight-index total_size mismatch')
    return shards


def text_cases(manifest, build_count_prompt):
    cases = ['', 'Yes', 'No', ' Yes', ' No', 'YES', 'yes\n', '0', '09', '10', '128',
             'Answer: ', '  Alice\tBob\r\nKitchen\n\n', "Alice's room isn't Bob's.",
             'café cafe\u0301 Å A\u030a', '你好 世界', 'שלום', 'مرحبا', '🙂🏠👩‍🔬',
             '<think>Count matching frames.</think>\n<answer>6</answer>',
             '<|im_start|>assistant\nYes<|im_end|>',
             '<|vision_start|><|image_pad|><|vision_end|>']
    cases.extend(str(number) for number in range(129))
    rng = random.Random(20260910)
    alphabet = "ABCxyz0123456789 .,!?'-\n\t é中🙂"
    cases.extend(''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 161)))
                 for _ in range(40))
    data = json.loads(manifest.read_text())
    records = [row for cell in data['splits'].values() for row in cell['samples']]
    ensure(records, 'Software manifest contains no examples')
    prompts, references = [], []
    for row in records[:4]:
        qa = Path(row['path']) / 'qa.txt'
        lines = qa.read_text().splitlines()
        q_start, q_end = lines.index('question:'), lines.index('answer:')
        question = next(line.strip() for line in lines[q_start + 1:q_end]
                        if line.strip() and not line.strip().startswith('{'))
        prompt = build_count_prompt(question, int(row['n_frames']))
        cases.append(prompt)
        prompts.append((prompt, int(row['n_frames'])))
        references.append(dict(path=str(qa), sha256=digest(qa), question=question))
    return cases, prompts, references


def check_tokenizers(source, mirror, manifest, vocab, merges, pattern):
    import torch
    from PIL import Image
    from copy import copy
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX
    sys.path.insert(0, str(REPO))
    from gnnformer.data import build_count_prompt, build_prompt_inputs

    torch.set_num_threads(1)
    ensure(pattern == PRETOKENIZE_REGEX, 'Fast and slow pretokenizer regex differs')
    fast = AutoTokenizer.from_pretrained(str(source), use_fast=True, local_files_only=True)
    slow = AutoTokenizer.from_pretrained(str(mirror), use_fast=False, local_files_only=True)
    ensure(fast.is_fast and not slow.is_fast, 'Wrong tokenizer implementations')
    ensure(fast.get_vocab() == slow.get_vocab(), 'Vocabulary or added-token IDs differ')
    ensure(slow.encoder == vocab, 'Derived slow base vocabulary differs')
    ensure(slow.bpe_ranks == {pair: rank for rank, pair in enumerate(merges)}, 'Merge ranks differ')
    ensure(fast.get_added_vocab() == slow.get_added_vocab(), 'Added vocabulary differs')
    ensure(set(fast.all_special_ids) == set(slow.all_special_ids), 'Special token IDs differ')
    ensure(fast.special_tokens_map == slow.special_tokens_map, 'Special token mapping differs')
    for attribute in ('bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id',
                      'padding_side', 'truncation_side', 'model_max_length'):
        ensure(getattr(fast, attribute) == getattr(slow, attribute), f'Tokenizer {attribute} differs')
    for index, token in fast.added_tokens_decoder.items():
        other = slow.added_tokens_decoder[index]
        for attribute in ('content', 'single_word', 'lstrip', 'rstrip', 'normalized', 'special'):
            ensure(getattr(token, attribute) == getattr(other, attribute),
                   f'Added-token attribute mismatch: {index}/{attribute}')
    source_config = AutoConfig.from_pretrained(str(source), local_files_only=True).to_dict()
    mirror_config = AutoConfig.from_pretrained(str(mirror), local_files_only=True).to_dict()
    source_config.pop('_name_or_path', None)
    mirror_config.pop('_name_or_path', None)
    ensure(source_config == mirror_config, 'Model config semantics changed')
    processor = AutoProcessor.from_pretrained(str(mirror), use_fast=False, local_files_only=True)
    ensure(not processor.tokenizer.is_fast, 'AutoProcessor did not retain slow tokenizer')
    ensure(not type(processor.image_processor).__name__.endswith('Fast'), 'Unexpected fast image processor')
    ensure(processor.tokenizer.get_vocab() == fast.get_vocab(), 'Processor vocabulary differs')

    cases, prompts, references = text_cases(manifest, build_count_prompt)
    for token in fast.get_added_vocab():
        cases.extend((token, 'x ' + token + ' y'))
    for prompt, frames in prompts:
        messages = [{'role': 'user', 'content':
                     [{'type': 'image'} for _ in range(frames)] + [{'type': 'text', 'text': prompt}]}]
        fast_text = fast.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        slow_text = slow.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ensure(fast_text == slow_text, 'Tokenizer chat templates render differently')
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        cases.extend((fast_text, rendered, rendered + 'Yes', rendered + 'No',
                      rendered + '12' + str(fast.eos_token)))
    checks = []
    for index, text in enumerate(cases):
        for special in (False, True):
            left = fast(text, add_special_tokens=special).input_ids
            right = slow(text, add_special_tokens=special).input_ids
            ensure(left == right, f'Encoding mismatch: case {index}, special={special}')
            ensure(processor.tokenizer(text, add_special_tokens=special).input_ids == left,
                   'Processor text encoding differs')
            for skip in (False, True):
                ensure(fast.decode(left, skip_special_tokens=skip, clean_up_tokenization_spaces=False)
                       == slow.decode(right, skip_special_tokens=skip, clean_up_tokenization_spaces=False),
                       f'Decode mismatch: case {index}')
            checks.append(dict(case=index, add_special_tokens=special, token_ids=left))
    alternate = copy(processor)
    alternate.tokenizer = fast
    pictures = [Image.new('RGB', (56, 56), (30, 50, 70)), Image.new('RGB', (56, 56), (70, 50, 30))]
    prompt = build_count_prompt('How many steps did Alice spend in the kitchen?', 2)
    try:
        slow_inputs = build_prompt_inputs(processor, pictures, prompt)
        fast_inputs = build_prompt_inputs(alternate, pictures, prompt)
        ensure(set(slow_inputs) == set(fast_inputs), 'Processor fields differ')
        for key in slow_inputs:
            left, right = slow_inputs[key], fast_inputs[key]
            ensure(torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right,
                   f'Canonical image-processing parity failed: {key}')
    finally:
        for picture in pictures:
            picture.close()
    return dict(transformers_version=transformers.__version__,
                fast_class=type(fast).__name__, slow_class=type(slow).__name__,
                processor_class=type(processor).__name__, image_processor_class=type(processor.image_processor).__name__,
                vocabulary_size=len(fast.get_vocab()), base_vocabulary_size=len(vocab),
                merge_count=len(merges), deterministic_text_cases=len(cases), encoding_comparisons=len(checks),
                special_ids=fast.all_special_ids, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id,
                checks_sha256=hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest(),
                cases=cases, prompt_sources=references, multimodal_fields=sorted(slow_inputs),
                limitations='Exact vocabulary/merge/metadata parity and deterministic tests, not exhaustive proof for all strings. No model weights loaded.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    job = require_cpu_job()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    started = time.time()
    mirror = DEST_BASE / f'cosmos_reason1_compat_{job}'
    mirror.mkdir(parents=False, exist_ok=False)
    report = dict(schema_version=1, status='running', slurm_job_id=job,
                  source=str(args.source), mirror=str(mirror), script_sha256=digest(Path(__file__)),
                  manifest=str(args.manifest), manifest_sha256=digest(args.manifest), metadata={},
                  source_mutated=False, weights_loaded=False, downloaded=False)
    try:
        for name in METADATA:
            source = args.source / name
            ensure(source.is_file(), f'Missing canonical metadata: {name}')
            target = mirror / name
            shutil.copyfile(source, target)
            ensure(digest(source) == digest(target), 'Metadata copy mismatch')
            report['metadata'][name] = dict(sha256=digest(source), bytes=source.stat().st_size,
                                            source=str(source), source_target=str(source.resolve(strict=True)))
        config = json.loads((mirror / 'config.json').read_text())
        ensure(config['architectures'] == ['Qwen2_5_VLForConditionalGeneration']
               and config['model_type'] == 'qwen2_5_vl'
               and config['hidden_size'] == 3584 and config['num_hidden_layers'] == 28
               and config['num_attention_heads'] == 28 and config['num_key_value_heads'] == 4
               and config['use_sliding_window'] is False, 'Unexpected Cosmos configuration')
        vocab, merges, pattern = validate_bpe(json.loads((mirror / 'tokenizer.json').read_text()))
        write_json(mirror / 'vocab.json', vocab)
        with (mirror / 'merges.txt').open('x', encoding='utf-8', newline='\n') as handle:
            handle.write('#version: 0.2\n')
            handle.writelines(' '.join(pair) + '\n' for pair in merges)
        report['derived'] = {name: dict(sha256=digest(mirror / name), bytes=(mirror / name).stat().st_size)
                             for name in ('vocab.json', 'merges.txt')}
        report['shards'] = inspect_shards(args.source, mirror,
                                         json.loads((mirror / 'model.safetensors.index.json').read_text()))
        report['parity'] = check_tokenizers(args.source, mirror, args.manifest, vocab, merges, pattern)
        for name, entry in report['metadata'].items():
            ensure(digest(args.source / name) == entry['sha256'] == digest(mirror / name),
                   'Source or copied metadata changed during verification')
        report.update(status='passed', elapsed_seconds=time.time() - started)
        write_json(mirror / 'COMPATIBILITY_PASSED.json', report)
        print(json.dumps(dict(status='passed', model_path=str(mirror),
                              report=str(mirror / 'COMPATIBILITY_PASSED.json')), sort_keys=True), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.time() - started)
        write_json(mirror / 'COMPATIBILITY_FAILED.json', report)
        print(json.dumps(dict(status='failed', mirror=str(mirror), error=report['error'])), flush=True)
        raise


if __name__ == '__main__':
    main()
