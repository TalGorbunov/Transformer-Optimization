"""Shared native V7 preparation/readout for matched parallel and joint arms.

No model loading, training loop, answer lookup, vocabulary mask, external tally,
custom attention, or KV-cache implementation lives here. Parallel rows contain
complete canonical N1 prompts followed by one text-only, N-agnostic global row.
Joint is one ordinary all-image row with the same global instruction. Both arms
use the same FP32 ParallelLocalAggregation before the frozen native final norm;
the native carry/norm/head remain FP16. Generation broadcasts only the global
choice in the parallel arm. This is software, not an aggregation efficacy claim.

All tensor/image/model work requires Slurm. Importing this module is lightweight.
Preparation and self_test may run in CPU Slurm; actual native forwards require
GPU Slurm. Callers own source/data/checkpoint freezing and artifact publication.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def require_slurm(*, gpu=False):
    need(os.environ.get('SLURM_JOB_ID'), 'All tensor/image/model work requires Slurm')
    partition = os.environ.get('SLURM_JOB_PARTITION')
    need(partition in ('cpu', 'gpu'), 'Expected a CPU or GPU Slurm partition')
    if gpu:
        need(partition == 'gpu' and os.environ.get('SLURM_JOB_GPUS'),
             'Native model execution requires GPU Slurm')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_info(value):
    require_slurm()
    import torch
    tensor = value.detach().cpu().contiguous()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest())


def normal_copies(values, *, cpu=False):
    """Detach constants and remove inference-tensor status for later readout fits."""
    require_slurm()
    import torch
    with torch.inference_mode(False), torch.no_grad():
        return {key: (value.detach().cpu().clone() if cpu else value.detach().clone())
                for key, value in values.items()}


def encode_target(tokenizer, answer):
    """Canonical count-token sequence plus exactly the tokenizer EOS target.

    This supports multi-digit targets without treating their first digit as a
    complete answer. Callers must independently freeze actual template-boundary
    tokenization before training; this function does not inspect a scene label.
    """
    require_slurm()
    need(isinstance(answer, int) and not isinstance(answer, bool) and answer >= 0,
         'Expected a nonnegative integer target')
    ids = list(tokenizer(str(answer), add_special_tokens=False)['input_ids'])
    eos = tokenizer.eos_token_id
    need(ids and isinstance(eos, int) and not any(x in tokenizer.all_special_ids for x in ids),
         'Count target must be ordinary tokens followed by native tokenizer EOS')
    need(tokenizer.decode(ids, skip_special_tokens=False) == str(answer), 'Count tokens do not round trip')
    return ids + [eos]


def pack_rows(rows, pad_token_id):
    """Ordinary heterogeneous left padding; concatenate visual rows in row order."""
    require_slurm()
    import torch
    need(rows and isinstance(pad_token_id, int), 'Need complete rows and an integer pad ID')
    width = max(int(row['input_ids'].shape[1]) for row in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    visual = []
    for index, row in enumerate(rows):
        need(set(row) in ({'input_ids', 'attention_mask'},
                         {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'}),
             'Unsupported native processor fields')
        rid, rmask = row['input_ids'], row['attention_mask']
        need(rid.ndim == 2 and rid.shape[0] == 1 and rid.shape[1] > 0
             and rid.dtype == torch.long and rmask.shape == rid.shape
             and rid.device.type == 'cpu' and bool((rmask == 1).all()),
             'Each row must be complete, unpadded CPU token input')
        length = rid.shape[1]
        ids[index, -length:] = rid[0]
        mask[index, -length:] = 1
        if 'pixel_values' in row:
            need(row['image_grid_thw'].ndim == 2 and row['image_grid_thw'].shape[1] == 3,
                 'Malformed native image grid')
            visual.append(row)
    packed = dict(input_ids=ids, attention_mask=mask)
    if visual:
        packed.update(pixel_values=torch.cat([r['pixel_values'] for r in visual]),
                      image_grid_thw=torch.cat([r['image_grid_thw'] for r in visual]))
    return packed


def append_prefix(inputs, prefix_ids):
    require_slurm()
    import torch
    need(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in prefix_ids),
         'Prefix must be a list of nonnegative token IDs')
    result = dict(inputs)
    if prefix_ids:
        ids = inputs['input_ids']
        tail = torch.tensor(prefix_ids, dtype=ids.dtype, device=ids.device).unsqueeze(0).expand(ids.shape[0], -1)
        result['input_ids'] = torch.cat((ids, tail), dim=1)
        result['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones_like(tail)), dim=1)
    return result


def prepare_scene(processor, sample, arm, *, resize=392, prefix_ids=(), verify_processor_parity=False):
    """Return {inputs,row_inputs,metadata}; read only question/images/N/SID.

    Optional prefix IDs are broadcast to complete rows after the original chat
    suffix. They are never inferred from sample.gold. Images retain original
    pixels/Step labels before the registered square resize. Full processor
    parity is an optional CPU-stage check, not a new inference operation.
    """
    require_slurm()
    import torch
    from PIL import Image
    from gnnformer.data import build_count_prompt
    from gnnformer.parallel_local_prompts import build_set_count_prompt
    need(arm in ('parallel', 'joint') and resize == 392, 'Expected a V7 arm and fixed392 resize')
    images, question = sample['image_files'], sample['question']
    need(isinstance(question, str) and question.strip() and len(images) == sample['n_frames'] > 0,
         'Scene question/frame coverage differs')
    prefix_ids = list(prefix_ids)
    need(all(x not in processor.tokenizer.all_special_ids for x in prefix_ids),
         'Do not append special or EOS tokens as an observed reasoning prefix')
    pad = processor.tokenizer.pad_token_id
    need(isinstance(pad, int), 'Native tokenizer must define padding; do not mutate it')
    frames, rows, conversations, original_side = [], [], [], processor.tokenizer.padding_side
    try:
        for record in images:
            path = Path(record['path'])
            if 'sha256' in record:
                need(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], 'Original image bytes changed')
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                try:
                    frame = rgb.resize((resize, resize))
                finally:
                    rgb.close()
            frames.append(frame)
        global_prompt = build_set_count_prompt(question)
        if arm == 'parallel':
            for frame in frames:
                conversations.append([dict(role='user', content=[dict(type='image', image=frame),
                    dict(type='text', text=build_count_prompt(question, 1))])])
            conversations.append([dict(role='user', content=[dict(type='text', text=global_prompt)])])
        else:
            conversations.append([dict(role='user', content=[dict(type='image', image=frame) for frame in frames]
                                                    + [dict(type='text', text=global_prompt)])])
        for conversation in conversations:
            rows.append(dict(processor.apply_chat_template(conversation, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt')))
        need(arm != 'parallel' or ('pixel_values' not in rows[-1]
             and all(row['image_grid_thw'].shape[0] == 1 for row in rows[:-1])),
             'Parallel global must be text only and each local row exactly one image')
        packed = pack_rows(rows, pad)
        need(packed['image_grid_thw'].shape[0] == len(images), 'Native visual grid count differs from the scene')
        if verify_processor_parity:
            processor.tokenizer.padding_side = 'left'
            native = dict(processor.apply_chat_template(conversations, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors='pt', padding=True))
            need(set(native) == set(packed) and all(torch.equal(native[key], packed[key]) for key in native),
                 'Manual packing differs from the ordinary batched native processor')
        lengths = [int(row['input_ids'].shape[1]) for row in rows]
        metadata = dict(arm=arm, sid=sample.get('sid'), n_frames=len(images), question=question,
            question_sha256=object_sha(question), global_prompt=global_prompt,
            local_prompt=None if arm == 'joint' else build_count_prompt(question, 1),
            row_kinds=['local'] * len(images) + ['global'] if arm == 'parallel' else ['joint'],
            row_prompt_tokens=lengths, original_prompt_width=int(packed['input_ids'].shape[1]),
            prefix_ids=prefix_ids, row_count=len(rows), global_row=len(rows)-1,
            local_elements=len(images) if arm == 'parallel' else 1,
            image_sha256=[record.get('sha256') for record in images],
            image_paths=[str(record['path']) for record in images], resize=resize,
            processor_parity_checked=bool(verify_processor_parity))
        rows = [append_prefix(row, prefix_ids) for row in rows]
        packed = append_prefix(packed, prefix_ids)
        metadata['prompt_width'] = int(packed['input_ids'].shape[1])
        metadata['input_identity'] = {key: tensor_info(value) for key, value in packed.items()}
        return dict(inputs=packed, row_inputs=rows, metadata=metadata)
    finally:
        processor.tokenizer.padding_side = original_side
        for frame in frames:
            frame.close()


def audit_layout(rope_index, bundle):
    """Verify packed logical mRoPE against each complete isolated row."""
    require_slurm()
    import torch
    inputs, rows, meta = bundle['inputs'], bundle['row_inputs'], bundle['metadata']
    need(inputs['input_ids'].shape == inputs['attention_mask'].shape
         and inputs['input_ids'].shape == (meta['row_count'], meta['prompt_width']), 'Bundle shape differs')
    need({key: tensor_info(value) for key, value in inputs.items()} == meta['input_identity'],
         'Prepared native tensors changed')
    positions, deltas = rope_index(input_ids=inputs['input_ids'], image_grid_thw=inputs.get('image_grid_thw'),
                                  attention_mask=inputs['attention_mask'])
    need(positions.shape == (3, meta['row_count'], meta['prompt_width']), 'Expected three native mRoPE axes')
    for index, row in enumerate(rows):
        length = row['input_ids'].shape[1]
        left = meta['prompt_width'] - length
        one, delta = rope_index(input_ids=row['input_ids'], image_grid_thw=row.get('image_grid_thw'),
                               attention_mask=row['attention_mask'])
        need(torch.equal(inputs['input_ids'][index, -length:], row['input_ids'][0])
             and bool((inputs['attention_mask'][index, :left] == 0).all())
             and bool((inputs['attention_mask'][index, left:] == 1).all())
             and torch.equal(positions[:, index, -length:], one[:, 0])
             and torch.equal(deltas[index:index+1] + left, delta),
             'Packed row changed its native tokens, mask, logical positions or delta')
    return dict(position_ids=positions, rope_deltas=deltas,
                metadata=dict(position_ids=tensor_info(positions), rope_deltas=deltas.tolist(),
                              every_unpadded_row_exact=True))


class JointLocalNative:
    """One-row control: the joint last-query h is both global and one local item."""

    def __init__(self, norm, branch, *, capture=False):
        require_slurm()
        import torch
        from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
        need(isinstance(norm, torch.nn.Module) and isinstance(branch, ParallelLocalAggregation),
             'Expected native norm and the shared V7 core')
        need(not any(p.requires_grad for p in norm.parameters()), 'Native norm must remain frozen')
        self.norm, self.branch, self.capture = norm, branch, bool(capture)
        self.calls, self.last_query_position, self.last_capture, self._handle = 0, None, None, None

    @property
    def active(self):
        return self._handle is not None

    def __enter__(self):
        need(not self.active, 'Joint controller is already active')
        need(not any(p.requires_grad for p in self.norm.parameters()), 'Native norm must remain frozen')
        self.calls, self.last_query_position, self.last_capture = 0, None, None
        self._handle = self.norm.register_forward_pre_hook(self._before_norm, with_kwargs=True)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __exit__(self, *_):
        self.close()
        return False

    def _before_norm(self, module, args, kwargs):
        import torch
        need(module is self.norm and self.active, 'Unexpected joint norm call')
        need(not any(p.requires_grad for p in module.parameters()), 'Native norm must remain frozen')
        positional = bool(args)
        need(not (positional and 'hidden_states' in kwargs), 'Ambiguous native hidden states')
        hidden = args[0] if positional else kwargs['hidden_states']
        need(isinstance(hidden, torch.Tensor) and hidden.is_floating_point() and hidden.ndim == 3
             and hidden.shape[0] == 1 and hidden.shape[1] > 0 and hidden.shape[2] == self.branch.hidden_size,
             'Joint arm must use exactly one native row [1,L,H]')
        global_states = hidden[0, -1:, :]
        local = global_states.unsqueeze(0)
        delta = self.branch(local, global_states, output_dtype=hidden.dtype)
        need(delta.shape == global_states.shape and delta.dtype == hidden.dtype, 'Native residual shape/dtype differs')
        fused = hidden.clone()
        fused[0, -1:, :] = global_states + delta
        self.calls += 1
        self.last_query_position = hidden.shape[1] - 1
        self.last_capture = None
        if self.capture:
            self.last_capture = dict(local_states=local.detach().clone(), global_states=global_states.detach().clone(),
                                     delta=delta.detach().clone(), fused_global=fused[0, -1:, :].detach().clone())
        if positional:
            return (fused,) + args[1:], kwargs
        updated = dict(kwargs)
        updated['hidden_states'] = fused
        return args, updated

    def export_last_capture(self, *, cpu=False):
        return None if self.last_capture is None else normal_copies(self.last_capture, cpu=cpu)


def native_contract(model, branch=None):
    require_slurm(gpu=True)
    import torch
    need(not model.training and not any(p.requires_grad for p in model.parameters()),
         'Freeze the complete native model and put it in eval mode before use')
    norm = model.model.language_model.norm
    need(norm.weight.dtype == model.lm_head.weight.dtype == torch.float16,
         'V7 requires the actual native FP16 norm/head, not a recast backbone')
    need(model.config.text_config._attn_implementation == 'sdpa', 'V7 uses native SDPA')
    if branch is not None:
        from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
        need(isinstance(branch, ParallelLocalAggregation) and branch.hidden_size == 3584 and branch.rank == 96
             and branch.merge == 'sum' and branch.post_activation == 'silu', 'V7 core/operator differs')
        need(sum(p.numel() for p in branch.parameters()) == 1041600
             and all(p.dtype == torch.float32 and p.device == norm.weight.device for p in branch.parameters()),
             'V7 core must have1041600 FP32 parameters on the native device')
    return norm


def _controller(norm, branch, meta, *, capture):
    if meta['arm'] == 'parallel':
        from gnnformer.parallel_local_native import ParallelLocalNative
        return ParallelLocalNative(norm, branch, n_local_rows=meta['n_frames'], capture=capture)
    return JointLocalNative(norm, branch, capture=capture)


def _raw_split(hidden, meta):
    need(hidden.ndim == 3 and hidden.shape[0] == meta['row_count'] and hidden.shape[-1] == 3584,
         'Actual native hidden row/width differs')
    global_states = hidden[-1, -1:, :]
    local = hidden[:-1, -1:, :] if meta['arm'] == 'parallel' else global_states.unsqueeze(0)
    return dict(local_states=local, global_states=global_states)


def forward_native(model, bundle, branch=None, *, capture=True, cpu=True):
    """One uncached native forward, optionally fusing the shared V7 branch.

    This handles one last query only. For the EOS training position, prepare a
    separate bundle with the observed preceding count token. Never append a gold
    answer when harvesting the first prediction position.
    """
    require_slurm(gpu=True)
    import torch
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, branch)
    meta = bundle['metadata']
    layout = audit_layout(get_rope_index_fn(model), bundle)
    inputs = move_to_device(bundle['inputs'], model.device)
    counts, observed = dict(model=0, visual=0, norm=0), {}
    def model_hook(*_): counts['model'] += 1
    def visual_hook(*_): counts['visual'] += 1
    def norm_hook(module, args):
        counts['norm'] += 1
        need(args[0].dtype == torch.float16, 'Native carry must stay FP16')
        if capture: observed.update(normal_copies(_raw_split(args[0], meta), cpu=cpu))
    def language_hook(module, args, kwargs):
        observed['positions'] = kwargs['position_ids'].detach().cpu().clone()
        observed['mask'] = kwargs['attention_mask'].detach().cpu().clone()
    begin = time.perf_counter()
    with ExitStack() as stack:
        for handle in (model.register_forward_pre_hook(model_hook),
                       model.model.visual.register_forward_pre_hook(visual_hook),
                       norm.register_forward_pre_hook(norm_hook),
                       model.model.language_model.register_forward_pre_hook(language_hook, with_kwargs=True)):
            stack.callback(handle.remove)
        fusion = stack.enter_context(_controller(norm, branch, meta, capture=capture)) if branch is not None else None
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        need(counts == dict(model=1, visual=1, norm=1) and (fusion is None or fusion.calls == 1),
             'One native full forward/visual/fusion call required')
        need(torch.equal(observed['positions'], layout['position_ids'])
             and torch.equal(observed['mask'], bundle['inputs']['attention_mask'])
             and torch.equal(model.model.rope_deltas.detach().cpu(), layout['rope_deltas']),
             'Native forward changed staged logical mRoPE/mask')
        need(output.past_key_values is None and output.logits.shape[:2] == (meta['row_count'], 1)
             and bool(torch.isfinite(output.logits).all()), 'Uncached native logits contract differs')
        values = normal_copies(dict(global_logits=output.logits[-1, -1]), cpu=cpu)
        if capture:
            values.update({key: observed[key] for key in ('local_states', 'global_states')})
            if fusion is not None:
                values['fusion'] = fusion.export_last_capture(cpu=cpu)
    return dict(values, metadata=dict(meta, layout=layout['metadata'], native_dtype='torch.float16',
                                     branch_dtype=None if branch is None else 'torch.float32'),
                counters=counts, model_seconds=time.perf_counter()-begin)


def extract_raw_states(model, bundle, *, cpu=True):
    """Frozen native h only: no branch or generated-answer conditioning added."""
    return forward_native(model, bundle, branch=None, capture=True, cpu=cpu)


def generation_policy(model, tokenizer, *, max_new_tokens=4):
    """Fresh greedy config: preserve native EOS IDs, disable inherited penalties."""
    require_slurm()
    from transformers import GenerationConfig
    need(max_new_tokens == 4, 'V7 native evaluation has a fixed four-token budget')
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    need(eos and len(set(eos)) == len(eos) and tokenizer.eos_token_id in eos
         and all(isinstance(x, int) and x in tokenizer.all_special_ids for x in eos),
         'Native generation EOS/tokenizer target boundary differs')
    need(isinstance(tokenizer.pad_token_id, int), 'Native pad token is required')
    config = GenerationConfig(max_new_tokens=4, do_sample=False, num_beams=1,
        num_return_sequences=1, repetition_penalty=1.0, use_cache=True,
        bos_token_id=model.generation_config.bos_token_id, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, output_logits=True)
    return config, dict(max_new_tokens=4, do_sample=False, num_beams=1, repetition_penalty=1.0,
        native_eos_token_ids=eos, target_eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, vocabulary_mask=False, other_logits_processors=False)


def generate_native(model, processor, branch, bundle, *, max_new_tokens=4, capture=False, cpu=True):
    """Native cached generation; parallel choices use the global row only.

    Return global generated_ids and [generated_tokens,vocabulary] raw_logits;
    raw means native output before the broadcast processor, without normalization
    or numeric masking. EOS completion/truncation are separate from any parser.
    Optional captures contain each token's pre-final-RMS inputs and fused carry.
    No comparison with uncached decoding or reasoning efficacy is implied.
    """
    require_slurm(gpu=True)
    import torch
    from transformers import LogitsProcessorList
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    from gnnformer.runtime import get_rope_index_fn, move_to_device
    norm = native_contract(model, branch)
    need(branch is not None, 'Native V7 generation requires the explicit shared branch')
    meta, inputs = bundle['metadata'], move_to_device(bundle['inputs'], model.device)
    layout = audit_layout(get_rope_index_fn(model), bundle)
    config, policy = generation_policy(model, processor.tokenizer, max_new_tokens=max_new_tokens)
    width, batch = meta['prompt_width'], meta['row_count']
    processors = LogitsProcessorList()
    broadcast = None
    if meta['arm'] == 'parallel':
        broadcast = GlobalBroadcastLogitsProcessor(n_local_rows=meta['n_frames'], prompt_length=width)
        processors.append(broadcast)
    counts = dict(model=0, visual=0, language=0)
    step_inputs, captures, position_rows = [], [], []
    def model_hook(module, args, kwargs):
        index = counts['model']; counts['model'] += 1
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        need(ids.shape == (batch, width if index == 0 else 1), 'Unexpected prefill/cache token shape')
        expected_mask = append_prefix(bundle['inputs'], [0] * index)['attention_mask']
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
        position_rows.append(tensor_info(positions))
    begin = time.perf_counter()
    with ExitStack() as stack:
        fusion = stack.enter_context(_controller(norm, branch, meta, capture=capture))
        def output_hook(module, args, output):
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
        values = normal_copies(dict(raw_logits=raw), cpu=cpu)
    return dict(values, generated_ids=generated, raw_text=processor.tokenizer.decode(generated, skip_special_tokens=False),
        text=processor.tokenizer.decode(generated, skip_special_tokens=True), completed=completed,
        truncated=not completed, finish_reason='eos' if completed else 'length',
        metadata=dict(meta, layout=layout['metadata'], generation=policy, native_dtype='torch.float16',
                      branch_dtype='torch.float32', generation_position_ids=position_rows),
        counters=dict(counts, fusion=steps, broadcast=0 if broadcast is None else broadcast.calls),
        captures=captures, model_seconds=time.perf_counter()-begin)


def self_test():
    """Small CPU checks; native HF integration remains a separate GPU gate."""
    require_slurm()
    need(os.environ.get('SLURM_JOB_PARTITION') == 'cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Runtime self-tests require CPU Slurm')
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    from gnnformer.parallel_local_native import GlobalBroadcastLogitsProcessor
    torch.set_num_threads(4)
    results = []
    a = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
             pixel_values=torch.tensor([[1., 2.]]), image_grid_thw=torch.tensor([[1, 1, 1]]))
    b = dict(input_ids=torch.tensor([[4]]), attention_mask=torch.ones(1, 1, dtype=torch.long))
    packed = pack_rows([a, b], 99)
    need(packed['input_ids'].tolist() == [[1, 2, 3], [99, 99, 4]]
         and packed['attention_mask'].tolist() == [[1, 1, 1], [0, 0, 1]]
         and torch.equal(packed['pixel_values'], a['pixel_values']), 'Heterogeneous packing failed')
    extended = append_prefix(packed, [5, 6])
    need(extended['input_ids'].tolist() == [[1, 2, 3, 5, 6], [99, 99, 4, 5, 6]]
         and packed['input_ids'].shape[1] == 3, 'Prefix append changed original inputs')
    results.append('heterogeneous_visual_text_padding_and_immutable_prefix')
    with torch.random.fork_rng():
        torch.manual_seed(20260927)
        branch = ParallelLocalAggregation(8, rank=3)
        norm = torch.nn.Identity()
        hidden = torch.randn(1, 3, 8, dtype=torch.float16)
        before = hidden.clone()
        with JointLocalNative(norm, branch, capture=True) as joint:
            zero = norm(hidden)
            need(torch.equal(zero, hidden) and joint.calls == 1, 'Zero branch changed native carry')
            exported = joint.export_last_capture(cpu=True)
            need(torch.equal(exported['local_states'][0], exported['global_states']), 'Joint state alias contract differs')
        need(not norm._forward_pre_hooks and torch.equal(hidden, before), 'Joint hook/input not restored')
        with torch.no_grad(): branch.up.weight.normal_(0, .2)
        expected = hidden.clone()
        expected[0, -1:] = hidden[0, -1:] + branch(hidden[0, -1:].unsqueeze(0), hidden[0, -1:], output_dtype=hidden.dtype)
        with JointLocalNative(norm, branch) as joint:
            actual = norm(hidden)
            need(torch.equal(actual, expected) and torch.equal(actual[:, :-1], hidden[:, :-1]),
                 'Joint hook differs from explicit same-core operation or changes earlier positions')
            actual.float().square().sum().backward()
        need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).any())
                 for p in branch.parameters()), 'A joint branch parameter is disconnected')
        with JointLocalNative(norm, branch, capture=True) as joint:
            with torch.inference_mode(): norm(hidden)
            need(all(not value.is_inference() for value in joint.export_last_capture().values()),
                 'Exported captures must be normal constants')
        try:
            with JointLocalNative(norm, branch): norm(hidden.expand(2, -1, -1))
        except ValueError:
            pass
        else:
            raise AssertionError('Joint controller accepted a fabricated two-row batch')
        need(not norm._forward_pre_hooks, 'Exceptional exit leaked the joint hook')
        results.extend(['zero_joint_identity', 'joint_same_core_exact_operation', 'joint_all_parameters_active',
                        'normal_detached_capture', 'joint_rejects_duplicate_batch', 'exception_cleanup'])
    processor = GlobalBroadcastLogitsProcessor(n_local_rows=1, prompt_length=3)
    raw = torch.tensor([[9., 1., 2.], [1., 8., 2.]])
    broadcast = processor(packed['input_ids'], raw)
    need(broadcast.argmax(-1).tolist() == [1, 1] and raw[0].argmax().item() == 0,
         'Global choice must broadcast without mutating raw scores')
    try:
        processor(torch.tensor([[1, 2, 3, 5], [99, 99, 4, 6]]), raw)
    except ValueError:
        pass
    else:
        raise AssertionError('Diverged generated histories were accepted')
    results.append('global_broadcast_raw_preservation_and_history_rejection')
    return dict(passed=True, tests=results, native_hf_integration_tested=False,
                no_model_or_aggregation_accuracy_claim=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2))
