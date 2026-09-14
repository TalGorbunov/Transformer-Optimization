"""V12 native generation with actual and training-reference image streams.

Keeps the immutable V7 ordinary generation, broadcast, mask and mRoPE contract.
All modes execute the same augmented batch and differ only at native readout. No training
or efficacy release is implied by this source file. Native integration must be
profiled before use; no inference feature cache substitutes for image inputs.
"""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts import native_vision_v7_runtime as native
from gnnformer.parallel_local_reference import ParallelLocalReference
need=native.need


REFERENCE_MANIFEST=Path('/mnt/data/gabriele/gnn_transformer/v10_balanced/main_manifest.json')


def reference_bank(question,manifest_file=REFERENCE_MANIFEST):
    """The exact ordered N8/K0 then N16/K0 training occurrence bank for q."""
    native.require_slurm()
    from scripts.stage_native_vision_v7_features import read,sha
    manifest_file=Path(manifest_file).resolve()
    need(manifest_file==REFERENCE_MANIFEST,'Reference bank must use the unchanged V10 training corpus')
    manifest=read(manifest_file);scenes=[];occurrences=[]
    for n in (8,16):
        candidates=[r for r in manifest['splits'][f'train_N{n}']['samples'] if r['question']==question and r['gold']==0]
        need(len(candidates)==1,'Require exactly one registered K0 training scene per N/question')
        row=candidates[0]
        need(row['n_frames']==n and len(row['image_files'])==n and sha(Path(row['path'])/'qa.txt')==row['qa_sha256'],
            'K0 reference source QA/frame count changed')
        scenes.append(dict(sid=row['sid'],question=question,n_frames=n,gold=0,qa_sha256=row['qa_sha256'],
            path=row['path'],content_sha256=row['content_sha256']))
        for i,image in enumerate(row['image_files']):
            need(sha(image['path'])==image['sha256'],'Reference image bytes changed')
            occurrences.append(dict(image,reference_index=len(occurrences),source_sid=row['sid'],source_n_frames=n,
                source_frame_index=i,source_qa_sha256=row['qa_sha256'],role='reference'))
    result=dict(schema_version=1,question=question,question_sha256=native.object_sha(question),reference_count=24,
        manifest_file=str(manifest_file),manifest_sha256=sha(manifest_file),scenes=scenes,occurrences=occurrences,
        ordering='all eight N8K0 frames then all sixteen N16K0 frames in original image order; no deduplication',
        selection='training labels only; no query-answer or prefix-based selection')
    result['bank_sha256']=native.object_sha(result)
    return result


def prepare_scene(processor,sample,bank,*,prefix_ids=(),verify_processor_parity=False):
    """Prepare N actual +24 reference N1 streams and one text-only global row.

    The native processor sees only the original image bytes and question. Known
    K0 source labels select the bank externally; no QA/label/role token is added.
    Returned benchmark n_frames remains actual N; processed_image_rows is N+24.
    """
    n=sample['n_frames'];question=sample['question']
    need(type(n) is int and n>0 and len(sample['image_files'])==n and bank['question']==question
        and bank['reference_count']==len(bank['occurrences'])==24,'Actual scene/reference bank mismatch')
    payload={k:v for k,v in bank.items() if k!='bank_sha256'}
    need(native.object_sha(payload)==bank['bank_sha256'],'Reference bank metadata changed')
    references=[dict(path=r['path'],sha256=r['sha256']) for r in bank['occurrences']]
    synthetic=dict(sid=sample['sid'],question=question,n_frames=n+24,image_files=list(sample['image_files'])+references)
    bundle=native.prepare_scene(processor,synthetic,'parallel',prefix_ids=prefix_ids,
        verify_processor_parity=verify_processor_parity)
    meta=bundle['metadata'];meta.update(n_frames=n,actual_n_frames=n,processed_image_rows=n+24,reference_rows=24,
        local_elements=n,reference_elements=24,reference_bank=bank,reference_bank_sha256=bank['bank_sha256'],
        row_kinds=['actual']*n+['reference']*24+['global'],actual_image_sha256=[r['sha256'] for r in sample['image_files']],
        reference_image_sha256=[r['sha256'] for r in references],reference_source_sids=[r['source_sid'] for r in bank['occurrences']],
        reference_source_frame_indices=[r['source_frame_index'] for r in bank['occurrences']],
        no_role_or_label_tokens=True,benchmark_n_is_actual=True)
    validate_bundle(bundle);return bundle


def validate_bundle(bundle):
    meta=bundle['metadata'];n=meta['actual_n_frames'];bank=meta['reference_bank']
    need(meta['arm']=='parallel' and meta['n_frames']==n and meta['processed_image_rows']==n+24
        and meta['reference_rows']==24 and meta['row_count']==n+25 and meta['global_row']==n+24
        and meta['row_kinds']==['actual']*n+['reference']*24+['global']
        and meta['question']==bank['question'] and meta['reference_bank_sha256']==bank['bank_sha256'],
        'Actual/reference/global native row partition changed')
    need(meta['image_sha256']==meta['actual_image_sha256']+meta['reference_image_sha256']
        and meta['reference_image_sha256']==[r['sha256'] for r in bank['occurrences']]
        and meta['reference_source_sids']==[r['source_sid'] for r in bank['occurrences']]
        and meta['reference_source_frame_indices']==[r['source_frame_index'] for r in bank['occurrences']]
        and len(bundle['row_inputs'])==n+25 and meta['no_role_or_label_tokens'] and meta['benchmark_n_is_actual'],
        'Reference occurrence order, bank hashes or benchmark N changed')
    need(native.object_sha({k:v for k,v in bank.items() if k!='bank_sha256'})==bank['bank_sha256'],
        'Reference bank binding differs')
    return dict(passed=True,actual_n_frames=n,reference_rows=24,processed_image_rows=n+24,global_row=n+24,
        reference_bank_sha256=bank['bank_sha256'])


def generate_native(model, processor, branch, bundle, *, mode, core_key, max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures contain actual/reference/global states before final norm and the readout write.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    native.require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native.native_contract(model, branch)
    need(branch is not None, 'Native V12 generation requires the explicit shared branch')
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = native.audit_layout(get_rope_index_fn(model), bundle)
    config, policy = native.generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    need(meta['arm'] == 'parallel' and width == meta['original_prompt_width'],
         'Natural generation starts from the unchanged parallel prompt without supplied answer tokens')
    validate_bundle(bundle)
    need(mode in ('base','background','sham') and isinstance(core_key,str) and core_key,'Unregistered reference intervention or missing core key')
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['processed_image_rows'], prompt_length=width)
        processors.append(broadcast)
    counts = dict(model=0, visual=0, language=0)
    step_inputs, captures, position_rows = [], [], []
    def model_hook(module, args, kwargs):
        index = counts['model']; counts['model'] += 1
        fusion.configure_queries([width-1 if index==0 else 0],[width+index-1])
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        need(ids.shape == (batch, width if index == 0 else 1), 'Unexpected prefill/cache token shape')
        expected_mask = native.append_prefix(bundle['inputs'], [0] * index)['attention_mask']
        need(torch.equal(mask.detach().cpu(), expected_mask), 'Native cached padding/key mask differs')
        if index == 0:
            need(torch.equal(ids.detach().cpu(), bundle['inputs']['input_ids'])
                 and kwargs.get('pixel_values') is not None, 'Native prefill changed inputs or omitted images')
        else:
            need(kwargs.get('pixel_values') is None and kwargs.get('past_key_values') is not None
                 and kwargs['past_key_values'].get_seq_length() == width + index - 1,
                 'Decode must use the native cache and no repeated visual forward')
            need(torch.equal(ids, ids[-1:].expand_as(ids)), 'Parallel generated prefixes diverged')
        step_inputs.append(ids.detach().cpu().clone())
    def visual_hook(*_): counts['visual'] += 1
    def language_hook(module, args, kwargs):
        index = counts['language']; counts['language'] += 1
        positions = kwargs['position_ids'].detach().cpu()
        need(positions.shape == (4, batch, width if index == 0 else 1),
             'Installed native generation must supply text plus three mRoPE axes')
        mask = kwargs['attention_mask'].detach().cpu()
        text = mask.long().cumsum(-1) - 1
        if index == 0:
            expected = layout['position_ids']
            valid = mask.bool()
            need(torch.equal(positions[0][valid], text[valid]), 'Native text positions differ')
        else:
            expected = (layout['rope_deltas'].view(1, batch, 1) + width + index - 1).expand(3, -1, -1)
            need(torch.equal(positions[0], text[:, -1:]), 'Native cached text positions differ')
        need(torch.equal(positions[1:], expected), 'Native generation logical mRoPE differs')
        position_rows.append(native.tensor_info(positions))
    begin = time.perf_counter()
    with ExitStack() as stack:
        fusion = stack.enter_context(ParallelLocalReference(norm,branch,n_actual_rows=meta['actual_n_frames'],
            n_reference_rows=24,mode=mode,anchor_n=16,sham_key=[core_key,meta['question']],
            query_indices=[width-1],stream_positions=[width-1],capture=capture))
        def output_hook(module, args, output):
            need(fusion.calls==fusion.actual_reads==fusion.reference_reads==counts['model'],'Reference query read/write counts differ')
            if capture: captures.append(fusion.export_last_capture(cpu=cpu))
        for handle in (model.register_forward_pre_hook(model_hook, with_kwargs=True),
                       model.model.visual.register_forward_pre_hook(visual_hook),
                       model.model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True),
                       model.register_forward_hook(output_hook)):
            stack.callback(handle.remove)
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=config,
                                    logits_processor=processors, logits_to_keep=1)
        torch.cuda.synchronize()
        suffix = output.sequences[:, width:]
        steps = suffix.shape[1]
        need(1 <= steps <= 4 and len(output.logits) == steps and counts == dict(model=steps, visual=1, language=steps)
             and fusion.calls == steps and (broadcast is None or broadcast.calls == steps),
             'Require exactly one native model/fusion invocation per generated token and one visual prefill')
        need(torch.equal(suffix, suffix[-1:].expand_as(suffix)), 'Final broadcast sequences differ')
        generated = suffix[-1].detach().cpu().tolist()
        for index in range(1, steps):
            need(bool((step_inputs[index] == generated[index-1]).all()), 'Executed cached token differs from output history')
        raw = torch.stack([value[-1] for value in output.logits])
        need(raw.ndim == 2 and raw.shape[0] == steps and bool(torch.isfinite(raw).all()), 'Raw global logits are malformed')
        # Greedy choices must be the GLOBAL raw argmax, not a hidden mask/penalty.
        need(raw.argmax(-1).detach().cpu().tolist() == generated, 'Generation altered raw global greedy choices')
        eos = policy['native_eos_token_ids']
        completed = generated[-1] in eos
        need(not any(token in eos for token in generated[:-1]) and (completed or steps == 4),
             'Unexpected early stop or an interior EOS')
        values = native.normal_copies(dict(raw_logits=raw), cpu=cpu)
    return dict(values, generated_ids=generated, raw_text=processor.tokenizer.decode(generated, skip_special_tokens=False),
        text=processor.tokenizer.decode(generated, skip_special_tokens=True), completed=completed,
        truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, layout=layout['metadata'], generation=policy, native_dtype='torch.float16',
                      reference_mode=mode,core_key=core_key,read_boundary='before_native_final_norm',
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls),
        captures=captures, model_seconds=time.perf_counter()-begin)

