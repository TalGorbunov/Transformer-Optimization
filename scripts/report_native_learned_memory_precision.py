"""Independent CPU report with a bounded descriptive RMS reduction comparison.

The native execution source is immutable. Only centered_rms in the cached/full
descriptive table permits two FP64 ULPs; TV, top1, pass flags, ownership and all
native head checks remain exact. The original failed report and diagnostic are
mandatory ancestors. No model, head, fit or native generation is executed.
"""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import profile_native_learned_memory as original

need, read, save, sha = original.need, original.read, original.save, original.sha
native, runtime, v11 = original.native, original.runtime, original.v11
backend, cosmos = original.backend, original.cosmos
PLACEMENTS = original.PLACEMENTS
audit_capture, kv_compare = original.audit_capture, original.kv_compare
load_bound, tensor_equal = original.load_bound, original.tensor_equal
ORIGINAL_SHA = '74f92f1f916e4573e00e3b17b5b241984b3ee41266992e541df98436ac0f3898'
DIAG = original.OUT / 'report_diagnosis_443153'
DIAG_SHA = '597dd9af14ceb5149379caf7cb9acf48cc334b8a284ac0333518de4df6539275'
DETAIL_SHAS = {
    'qwen': '00a8b80c45ed282649b52528a66fc281fd4a9ebec898a41c359bbc5e82ffe2a5',
    'cosmos': '47daa709d4deeb1d8c7a1088aa69dfe2d757d71d02b62e7d79a199c4f044a6f8',
}
FAILURE = original.OUT / 'report_443139/failure.json'
FAILURE_SHA = '39b8527d29baf62753a09995ab6fb69269147535726060030993aa1d96f68253'
OWN = ('scripts/report_native_learned_memory_precision.py',
       'slurm/native_learned_memory_precision_report.sbatch')
RULE = dict(field='centered_rms', maximum_fp64_ulps=2,
            formula='abs(a-b) <= 2 * max(math.ulp(a), math.ulp(b))',
            finite_nonnegative_required=True, all_other_fields_exact=True,
            native_head_checks_unchanged=True, tv_max_unchanged=.02,
            top1_and_numerical_flags_exact=True)


def dispatcher_sources():
    return {name: sha(REPO / name) for name in OWN}


def sources():
    return {**original.sources(), **dispatcher_sources()}


def compare_descriptive_rows(recomputed, saved):
    """Match exact row ownership; reveal every permitted scalar discrepancy."""
    expected = {'n_frames', 'placement', 'step', 'row', 'cached_label', 'full_label',
                'binding', 'full_vocabulary_tv', 'top1_equal', 'raw_maximum_absolute',
                'centered_maximum_absolute', 'centered_rms', 'numerical_rule_passed'}
    key = lambda row: (row['n_frames'], row['placement'], row['step'], row['row'])
    need(len(recomputed) == len(saved) and bool(saved), 'Descriptive row count differs')
    need(all(set(row) == expected for row in [*recomputed, *saved]), 'Descriptive fields differ')
    need(len({key(row) for row in recomputed}) == len(recomputed)
         and len({key(row) for row in saved}) == len(saved), 'Duplicate descriptive ownership')
    differences = []
    for actual, prior in zip(sorted(recomputed, key=key), sorted(saved, key=key)):
        need(all(type(actual[k]) is type(prior[k]) and actual[k] == prior[k]
                 for k in expected - {'centered_rms'}), 'Exact descriptive metadata/metric differs')
        a, b = actual['centered_rms'], prior['centered_rms']
        need(type(a) is float and type(b) is float and math.isfinite(a)
             and math.isfinite(b) and a >= 0 and b >= 0, 'Invalid descriptive RMS')
        unit = max(math.ulp(a), math.ulp(b))
        need(abs(a - b) <= 2 * unit, 'Descriptive RMS exceeds two FP64 ULPs')
        if a != b:
            differences.append(dict(n_frames=actual['n_frames'], placement=actual['placement'],
                step=actual['step'], row=actual['row'], cached_label=actual['cached_label'],
                full_label=actual['full_label'], field='centered_rms', saved=b, recomputed=a,
                absolute=abs(a-b), relative=abs(a-b)/max(abs(a), abs(b)),
                maximum_ulp=unit, ulps=abs(a-b)/unit))
    return dict(passed=True, comparisons=len(saved), rule=RULE, differences=differences,
                all_non_rms_fields_exact=True,
                saved_descriptive_failures=sum(not row['numerical_rule_passed'] for row in saved),
                recomputed_descriptive_failures=sum(not row['numerical_rule_passed'] for row in recomputed))


def verify_diagnosis(plan, runs):
    need(sha(REPO / 'scripts/profile_native_learned_memory.py') == ORIGINAL_SHA,
         'Immutable native execution/report source changed')
    need(sha(DIAG / 'summary.json') == DIAG_SHA and sha(FAILURE) == FAILURE_SHA,
         'Exact diagnostic/original failure changed')
    diag = read(DIAG / 'summary.json'); failure = read(FAILURE)
    need(diag['passed'] and diag['completed'] and diag['diagnostic_only']
         and diag['no_threshold_or_source_change'] and diag['slurm_job_id'] == '443153'
         and diag['original_failure_file'] == str(FAILURE)
         and diag['original_failure_sha256'] == FAILURE_SHA
         and failure['message'] == 'Fixed prefix/write coverage differs'
         and failure['source_sha256'] == plan['source_sha256'] == original.sources(),
         'Different diagnostic or failure ancestry')
    for name, digest in diag['source_sha256'].items():
        need(sha(REPO / name) == digest
             and sha(DIAG / 'source' / name.replace('/', '_')) == digest,
             'Diagnostic source/current snapshot differs')
    for name, digest in failure['source_sha256'].items():
        need(sha(FAILURE.parent / 'source' / name.replace('/', '_')) == digest,
             'Failed report source snapshot differs')
    need({s['model_key'] for _, s in runs} == set(DETAIL_SHAS) and len(runs) == 2,
         'Exactly the two diagnosed native runs required')
    details = {}
    for directory, summary in runs:
        key = summary['model_key']; path = DIAG / (key + '_diagnosis.json')
        need(sha(path) == DETAIL_SHAS[key], 'Detailed diagnostic changed')
        detail = read(path)
        need(detail['run'] == str(directory) and detail['summary_sha256'] == sha(directory / 'summary.json')
             and detail['saved_sha256'] == sha(directory / 'comparisons.json')
             and detail['comparisons'] == 328 and detail['all_flags_equal']
             and detail['original_descriptive_failures'] == detail['recomputed_descriptive_failures']
             and set(detail['field_maxima']) == {'centered_rms'}
             and all(row['field'] == 'centered_rms' for row in detail['differences']),
             'Diagnostic does not justify the limited RMS dispatcher')
        aggregate = next(row for row in diag['runs'] if row['run'] == str(directory))
        need(aggregate == {k:v for k,v in detail.items() if k not in ('raw_bindings', 'differences')},
             'Diagnostic summary/detail join differs')
        raw = {row['path']: row['sha256'] for row in read(directory / 'forwards.json')}
        need(all(raw.get(path) == digest for path, digest in detail['raw_bindings'].items()),
             'Diagnostic raw ownership differs')
        # The complete audit below rehashes these and every other raw tensor.
        details[key] = dict(file=str(path), sha256=DETAIL_SHAS[key], field_maxima=detail['field_maxima'])
    return dict(protocol='native_learned_memory_rms_precision_dispatcher', rule=RULE,
        source_sha256=dispatcher_sources(), original_source_sha256=ORIGINAL_SHA,
        diagnostic_file=str(DIAG / 'summary.json'), diagnostic_sha256=DIAG_SHA,
        original_failure_file=str(FAILURE), original_failure_sha256=FAILURE_SHA,
        details=details, no_native_execution_or_threshold_change=True)


# AUDIT_RUN_COPY: generated from the immutable source; verify_copy_contract()
# below requires exact AST equality after the two declared replacements.


def audit_run(torch,directory,summary,plan,packet):
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(plan['models'][summary['model_key']]['path'],trust_remote_code=True,use_fast=False,local_files_only=True)
    owner,fn,api=backend.native_api(processor) if summary['model_key']=='qwen' else cosmos.native_api(processor)
    need(api==plan['models'][summary['model_key']]['native_identity']['native_api'],'Independent report native API differs')
    records=read(directory/'forwards.json');need(len(records)==43 and len({r['label'] for r in records})==43,'Fixed native call coverage differs')
    by_label={r['label']:r for r in records};entry=plan['models'][summary['model_key']];cases={c['n_frames']:c for c in entry['cases']}
    head_metrics=[];normalization_exact=[];cache_comparisons=[];writes=0;raw_headers={}
    def raw_for(label):
        record=by_label[label];need(sha(record['path'])==record['sha256'],'Forward raw changed');return torch.load(record['path'],map_location='cpu',weights_only=True)
    expected=set()
    for n in (16,64):
        expected.update(f'N{n}_bare_cached{t}' for t in range(3))
        expected.update(f'N{n}_{c}_{p}_cached{t}' for c in ('zero','active') for p in PLACEMENTS for t in range(3))
        expected.update(f'N{n}_active_{p}_full{t}' for p in PLACEMENTS for t in (1,2))
    expected.update(('N16_pre_last_perturb_prefill','N16_pre_last_altered_next','N16_pre_last_restored_next',
                     'N16_post_last_perturb_prefill','N16_post_last_altered_next'))
    need(set(by_label)==expected,'Prescribed fixed-call labels differ')
    for record in records:
        need(record['passed'] and sha(record['partial_file'])==record['partial_sha256'],'Missing pre-gate raw artifact')
        raw=raw_for(record['label']);context=record['context'];n,t=context['n_frames'],context['step'];case=cases[n]
        bundle=packet['bundles'][case['case_id']+f'_t{t}'];width=case['prompt_width'];mask=bundle['inputs']['attention_mask'];layout=case['variants'][t]['layout']
        actual_layout=native.audit_layout(lambda **kw:fn(owner,**kw),bundle);need(actual_layout['metadata']==layout,'Independent CPU native mRoPE layout differs')
        positions=raw['position_ids'];cached=record['use_cache'];query=[width-1] if t==0 else [0]
        ids=bundle['inputs']['input_ids'] if not cached or t==0 else bundle['inputs']['input_ids'][:,-1:]
        need(raw['label']==record['label'] and raw['context']==context and torch.equal(raw['input_ids'],ids)
             and torch.equal(raw['attention_mask'],mask) and torch.equal(raw['expected_mask'],mask),'Raw input/current-prefix ownership differs')
        expected_positions=actual_layout['position_ids'] if not cached or t==0 else torch.cat(((mask.cumsum(-1)-1)[:,-1:].unsqueeze(0),actual_layout['position_ids'][:,:,-1:]),0)
        need(torch.equal(positions,expected_positions) and torch.equal(positions,raw['expected_positions']) and positions.shape==(3 if not cached or t==0 else 4,n+1,ids.shape[1]),'Actual position-axis layout differs')
        # CPU recomputation uses the exact model-specific installed rope helper, no model.
        need(native.tensor_info(raw['common_penultimate_hidden'])['shape']==[n+1,ids.shape[1],3584]
             and raw['native_logits'].shape==(n+1,1,152064) and raw['native_logits'].dtype==torch.float16,'Native shape/dtype differs')
        masks=v11.old.mask_check(torch,raw['last_block_mask'],raw['attention_mask'],raw['cache_position'])
        need(masks==record['mask_audit'],'Native causal/padding mask audit differs')
        expected_visual={k:native.tensor_info(bundle['inputs'][k]) for k in ('pixel_values','image_grid_thw')} if not cached or t==0 else {}
        # Native cached preparation may retain image_grid_thw, but never pixels.
        need(all(raw['visual_input_identity'].get(k)==v for k,v in expected_visual.items())
             and (not cached or t==0 or 'pixel_values' not in raw['visual_input_identity']), 'Actual image ownership differs')
        metrics=[v11.old.metric(torch,raw['replay_logits'][i,0],raw['native_logits'][i,0]) for i in range(n+1)]
        need(metrics==record['head_replay'] and all(m['numerical_rule_passed'] for m in metrics),'Independent same-state native head replay failed')
        head_metrics+=metrics
        norm_exact=torch.equal(raw['native_normalized'],raw['replay_normalized'])
        need(raw['native_normalized'].shape==raw['replay_normalized'].shape==(n+1,1,3584)
             and norm_exact==record['normalized_exact'],'Normalized replay identity evidence differs')
        normalization_exact.append(norm_exact)
        if raw['fusion'] is not None:
            history=query if cached else list(range(width-1,width+t));stream=[width-1+t] if cached else history
            audit_capture(torch,raw,context['placement'],history,stream);writes+=1
            need(record['formula']['passed'] and record['formula']['ordinary_formula_exact']
                 and record['formula']['applied_override']==(raw['intervention'] is not None),'GPU functional core binding differs')
        if cached:
            need(raw['kv']==record['kv'] and len(raw['kv'])==28,'Actual KV evidence incomplete')
            for layer in raw['kv']:
                need(all(layer[k]['shape']==[n+1,4,mask.shape[1],128] and layer[k]['dtype']=='torch.float16' for k in ('key','value')),'Actual all-layer native KV shape differs')
            if record['old_cache_length']:
                need(record['previous_prefix_exact_by_layer']==[True]*28,'Past KV was modified')
                if context['condition']!='intervention':
                    previous=by_label[record['label'][:-1]+str(t-1)]['kv']
                    need(all(a['retained_'+k]==b[k] for a,b in zip(raw['kv'],previous) for k in ('key','value')),'Retained-prefix tensor digests differ')
            if context['condition'] in ('zero','active'):
                baseline=raw_for(f'N{n}_bare_cached{t}')
                need(torch.equal(raw['common_penultimate_hidden'],baseline['common_penultimate_hidden']),'Common lower read differs by placement')
                check=kv_compare(raw['kv'],baseline['kv'],context['placement'],context['condition'])
                saved=next(x for x in read(directory/'kv_checks.json') if x['label']==record['label'])
                need(all(saved[k]==v for k,v in check.items()),'KV contrast metadata differs')
                if context['condition']=='zero':need(torch.equal(raw['native_logits'],baseline['native_logits'])
                    and torch.equal(raw['actual_norm_input'],baseline['actual_norm_input']) and not bool(raw['fusion']['delta'].any()),'Zero identity failed independently')
                del baseline
        if context['kind']=='full':
            cached_raw=raw_for(f'N{n}_active_{context["placement"]}_cached{t}')
            cache_comparisons.extend(dict(n_frames=n,placement=context['placement'],step=t,row=i,
                cached_label=cached_raw['label'],full_label=raw['label'],binding=False,
                **v11.old.metric(torch,cached_raw['native_logits'][i,0],raw['native_logits'][i,0])) for i in range(n+1))
            other=raw_for(f'N{n}_active_{PLACEMENTS[1] if context["placement"]==PLACEMENTS[0] else PLACEMENTS[0]}_full{t}')
            need(torch.equal(raw['common_penultimate_hidden'],other['common_penultimate_hidden']),'Full-prefix pre/post common read differs')
            del cached_raw,other
        raw_headers[record['label']]=dict(query_tokens=ids.shape[1],key_tokens=mask.shape[1]);del raw
    saved_comparisons=read(directory/'comparisons.json')
    key=lambda x:(x['n_frames'],x['placement'],x['step'],x['row'])
    precision=compare_descriptive_rows(cache_comparisons,saved_comparisons)
    need(len(cache_comparisons)==328 and writes==37,'Fixed prefix/write coverage differs')
    persistence_data=read(directory/'persistence.json');need(sha(persistence_data['file'])==persistence_data['sha256'],'Persistence index differs')
    banks={k:load_bound(torch,v) for k,v in persistence_data['cache_artifacts'].items()};first=persistence_data['first_write']
    need(set(banks)=={'baseline','altered','restored'} and all(len(x)==28 for x in banks.values())
         and first==cases[16]['prompt_width']-1,'Persistence cache fork coverage/position differs')
    for bank,label in (('baseline','N16_active_pre_last_cached0'),('altered','N16_pre_last_perturb_prefill')):
        expected=by_label[label]['kv']
        need(all(native.tensor_info(value)==expected[i][field] for i,pair in enumerate(banks[bank])
                 for field,value in zip(('key','value'),pair)), 'Archived cache fork is not the actual bound native prefill')
    need([x['placement'] for x in persistence_data['results']]==list(PLACEMENTS),'Missing persistence placement')
    for item in persistence_data['results']:
        place=item['placement'];tags=['altered','restored'] if place=='pre_last' else ['altered']
        need(item['prefill_label']==f'N16_{place}_perturb_prefill'
             and item['baseline_prefill_label']==f'N16_active_{place}_cached0'
             and item['baseline_next_label']==f'N16_active_{place}_cached1'
             and [x['kind'] for x in item['outcomes']]==tags
             and [x['label'] for x in item['outcomes']]==[f'N16_{place}_{tag}_next' for tag in tags],
             'Persistence causal comparison/cardinality differs')
    for i,(base,altered,restored) in enumerate(zip(banks['baseline'],banks['altered'],banks['restored'])):
        for a,b,c in zip(base,altered,restored):
            need(torch.equal(a,c),'Restoration did not recover every baseline cache byte')
            if i<27:need(torch.equal(a,b),'Perturbation changed a lower-layer cache')
            else:
                need(torch.equal(a[:-1],b[:-1]) and torch.equal(a[-1,:,:first],b[-1,:,:first])
                     and torch.equal(a[-1,:,first+1:],b[-1,:,first+1:]) and not torch.equal(a[-1,:,first],b[-1,:,first]),
                     'Perturbation escaped the one first-query global K/V slot')
    del banks
    for result in persistence_data['results']:
        changed=raw_for(result['prefill_label']);baseline=raw_for(result['baseline_prefill_label'])
        common=changed['fusion']['global_states'].float();direction=torch.sin(torch.arange(3584,dtype=torch.float32));direction*=torch.rsqrt(direction.square().mean())
        expected=changed['intervention']['original_delta']+.1*common.square().mean(-1,keepdim=True).sqrt()*direction
        need(torch.allclose(expected,changed['fusion']['delta'],rtol=1e-5,atol=1e-6)
             and torch.equal(changed['intervention']['original_delta'],baseline['fusion']['delta']), 'Fixed intervention magnitude/ordinary delta differs')
        base_next=raw_for(result['baseline_next_label'])
        for item in result['outcomes']:
            actual=raw_for(item['label']);same=torch.equal(actual['native_logits'],base_next['native_logits'])
            need(same==item['all_row_logits_equal_baseline'] and same==(item['kind']=='restored' or result['placement']=='post_last')
                 and torch.equal(actual['native_logits'][:-1],base_next['native_logits'][:-1])
                 and torch.equal(actual['fusion']['delta'],base_next['fusion']['delta'])
                 and torch.equal(actual['common_penultimate_hidden'],base_next['common_penultimate_hidden'])
                 and v11.old.metric(torch,actual['native_logits'][-1,0],base_next['native_logits'][-1,0])==item['metric'],'Fixed-token persistence result differs')
            del actual
        del changed,baseline,base_next
    gradients=read(directory/'gradients.json');need([x['placement'] for x in gradients]==list(PLACEMENTS),'Missing placement gradient replay')
    for item in gradients:
        value=load_bound(torch,item);raw=raw_for(item['source_label']);gradient=value['gradient'];placement=item['placement'];q=raw['fusion']['query_indices']
        need(gradient.shape==value['deltas'].shape==(4,3584) and gradient.dtype==torch.float32 and bool(torch.isfinite(gradient).all())
             and bool((gradient[0]==0).all()) and bool(gradient[1].ne(0).any())==(placement=='pre_last')
             and bool(gradient[2].ne(0).any()) and bool((gradient[3]==0).all()),'Quantized temporal/future/crossscene derivative differs')
        need(value['write_indices']==[[q[1]]]+[[]]*15+[q] and value['loss_indices']==[[]]*16+[[q[1]]]
             and value['loss_to_write_indices']==[2] and bool((value['deltas'][0]==0).all())
             and torch.equal(value['deltas'][1:],raw['fusion']['delta']), 'Selected-loss/all-history replay layout differs')
        need(torch.equal(value['block_input'],raw['actual_last_block_input'])
             and tensor_equal(torch,value['attention_mask'],raw['last_block_mask'])
             and torch.equal(value['cache_position'],raw['cache_position'])
             and all(torch.equal(a,b) for a,b in zip(value['position_embeddings'],raw['position_embeddings'])),'Actual native replay inputs differ')
        metrics=[v11.old.metric(torch,value['native_shape_logits'][i,0],raw['native_logits'][i,0]) for i in range(17)]
        need(metrics==item['head_replay'] and all(m['numerical_rule_passed'] for m in metrics),'Independent full-batch final-block/head replay failed')
        head_metrics+=metrics
        loss=torch.nn.functional.cross_entropy(value['loss_logits'].float(),torch.tensor([entry['forced_ids'][1]]))
        need(value['loss_logits'].shape==(1,152064) and math.isclose(float(loss),value['loss'],rel_tol=1e-6,abs_tol=1e-6), 'Selected-only loss projection differs')
        del raw,value
    naturals=read(directory/'naturals.json');need([x['placement'] for x in naturals]==list(PLACEMENTS),'Two N16 natural trajectories required')
    natural_tokens=0
    for item in naturals:
        value=load_bound(torch,item);heads=load_bound(torch,dict(file=item['replay_file'],sha256=item['replay_sha256']))
        tokens=value['generated_ids'];steps=len(tokens);natural_tokens+=steps;n16=cases[16];width=n16['prompt_width']
        need(1<=steps<=8 and len(value['captures'])==len(heads)==len(item['partials'])==len(item['formulas'])==steps
             and all(x['passed'] and x['ordinary_formula_exact'] and not x['applied_override'] for x in item['formulas']) and item['rope_restored'], 'Natural runtime coverage/cleanup differs')
        need(value['metadata']['native_identity_sha256']==entry['native_identity_sha256']
             and value['metadata']['scene_input_identity']==runtime.input_identity(packet['bundles'][n16['case_id']+'_t0'],entry['native_identity_sha256'])
             and value['counters']==dict(model=steps,visual=1,language=steps,norm=steps,head=steps,broadcast=steps,selection=steps,probe_head=0), 'Natural runtime/input/native identity differs')
        need(torch.equal(value['raw_logits'],value['raw_logits'].half().float()) and value['raw_logits'].shape==(steps,152064)
             and value['raw_logits'].argmax(-1).tolist()==tokens and [x['top1_token_id'] for x in value['logit_records']]==tokens,
             'Natural unfiltered global argmax differs')
        eos=value['metadata']['generation']['native_eos_token_ids']
        need(not any(t in eos for t in tokens[:-1]) and value['completed']==(tokens[-1] in eos)
             and value['truncated']==(tokens[-1] not in eos) and (value['completed'] or steps==8),'Natural EOS/length status differs')
        for t,(cap,replay,ref) in enumerate(zip(value['captures'],heads,item['partials'])):
            actual=load_bound(torch,ref)
            need(cap['query_indices']==([width-1] if t==0 else [0]) and cap['stream_positions']==[width-1+t]
                 and cap['write_location']==item['placement'] and cap['selection_mode']=='clip'
                 and torch.equal(cap['fused_query_hidden'],actual['capture']['fused_query_hidden'])
                 and torch.equal(value['raw_logits'][t],actual['native_logits'][-1,0].float()),'Natural current-write/capture/recorder differs')
            need(torch.equal(cap['messages'],cap['gates'].unsqueeze(-1)*cap['payload']) and torch.equal(cap['gates'],cap['scores'].clamp(0,1))
                 and torch.equal(cap['native_delta'],cap['delta'].half()) and torch.equal(cap['write_output'],cap['write_input']+cap['native_delta']), 'Natural learned selector/cast differs')
            metrics=[v11.old.metric(torch,replay['logits'][i,0],actual['native_logits'][i,0]) for i in range(17)]
            need(metrics==replay['all_rows']==item['all_row_head_replay'][t] and all(m['numerical_rule_passed'] for m in metrics), 'Natural full-batch native head evidence differs')
            head_metrics+=metrics
        del value,heads
    need(summary['counts']==dict(model=43+natural_tokens,visual=22,last_block=45+natural_tokens,head=45+natural_tokens,core=37+natural_tokens)
         and summary['extra_head_calls']==47+natural_tokens<=63 and natural_tokens<=16,'Independent complete call count differs')
    return dict(passed=True,model_key=summary['model_key'],slurm_job_id=summary['slurm_job_id'],native_identity=entry['native_identity'],
        native_identity_sha256=entry['native_identity_sha256'],counts=summary['counts'],extra_head_calls=47+natural_tokens,
        all_row_head_comparisons=len(head_metrics),maximum_head_tv=max(m['full_vocabulary_tv'] for m in head_metrics),
        normalized_exact_count=sum(normalization_exact),fixed_normalization_comparisons=len(normalization_exact),
        zero_checks=12,common_read_checks=24,all_layer_kv_contrasts=24,persistence_restoration_passed=True,
        actual_quantized_temporal_gradients_passed=True,cached_full_comparisons=328,cache_metric_precision=precision,
        cached_full_failures=[x for x in cache_comparisons if not x['numerical_rule_passed']],
        natural_tokens=natural_tokens,natural_global_prefixes_retained=True,
        kv_scope='All28-layer equality/past-prefix digest checks bind fixed calls; natural runtime binds native masks/cache lengths/current token history',
        same_state_replay_passed=True,no_accuracy=True,no_fit=True)


def verify_copy_contract():
    text = (REPO / 'scripts/profile_native_learned_memory.py').read_text()
    need(sha(REPO / 'scripts/profile_native_learned_memory.py') == ORIGINAL_SHA, 'Original audit source changed')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'audit_run')
    reference = ast.get_source_segment(text, node)
    before = "need(sorted(cache_comparisons,key=key)==sorted(saved_comparisons,key=key) and len(cache_comparisons)==328 and writes==37,'Fixed prefix/write coverage differs')"
    after = "precision=compare_descriptive_rows(cache_comparisons,saved_comparisons)\n    need(len(cache_comparisons)==328 and writes==37,'Fixed prefix/write coverage differs')"
    need(reference.count(before) == 1 and reference.count('cached_full_comparisons=328,') == 1,
         'Frozen audit replacement anchor differs')
    expected = reference.replace(before, after).replace('cached_full_comparisons=328,',
        'cached_full_comparisons=328,cache_metric_precision=precision,')
    own = Path(__file__).read_text()
    actual = next(n for n in ast.parse(own).body if isinstance(n, ast.FunctionDef) and n.name == 'audit_run')
    need(ast.dump(ast.parse(expected).body[0], include_attributes=False)
         == ast.dump(actual, include_attributes=False), 'Copied audit differs outside the two declared edits')


def self_test():
    import copy
    base = dict(n_frames=16, placement='pre_last', step=1, row=0,
                cached_label='c', full_label='f', binding=False, full_vocabulary_tv=.01,
                top1_equal=True, raw_maximum_absolute=.1, centered_maximum_absolute=.1,
                centered_rms=.125, numerical_rule_passed=True)
    def rejects(row):
        try:
            compare_descriptive_rows([row], [base])
        except ValueError:
            return
        raise ValueError('Malformed descriptive fixture was accepted')
    need(compare_descriptive_rows([base], [base])['differences'] == [], 'Exact fixture failed')
    row = copy.deepcopy(base); row['centered_rms'] = math.nextafter(base['centered_rms'], math.inf)
    need(compare_descriptive_rows([row], [base])['differences'][0]['ulps'] == 1., 'One ULP fixture failed')
    row['centered_rms'] = base['centered_rms'] + 2 * math.ulp(base['centered_rms'])
    need(compare_descriptive_rows([row], [base])['differences'][0]['ulps'] == 2., 'Two ULP boundary failed')
    row['centered_rms'] = base['centered_rms'] + 3 * math.ulp(base['centered_rms']); rejects(row)
    for field, value in (('full_vocabulary_tv', math.nextafter(.01, math.inf)), ('top1_equal', False),
                         ('numerical_rule_passed', False), ('top1_equal', 1), ('row', 1),
                         ('cached_label', 'different'), ('centered_rms', float('nan')),
                         ('centered_rms', float('inf')), ('centered_rms', -.1)):
        row = copy.deepcopy(base); row[field] = value; rejects(row)
    verify_copy_contract()
    return dict(passed=True, groups=6, cases=13, scope='Exact fields, two-ULP RMS boundary, flags/ownership/finiteness and copied-audit AST')


def report(args, out, frozen, reporter_frozen):
    import torch
    torch.set_num_threads(4)
    plan = original.verify_plan(args.plan)
    runs = [original.verify_run(path, plan) for path in args.runs]
    provenance = verify_diagnosis(plan, runs)
    tests = self_test(); save(out / 'selftest.json', tests)
    packet = torch.load(plan['prepared_file'], map_location='cpu', weights_only=True)
    results = []
    for directory, summary in runs:
        audit = audit_run(torch, directory, summary, plan, packet)
        save(out / (summary['model_key'] + '_audit.json'), audit)
        results.append(audit)
    resources = original.accounting(out); successful = {row['job_id']:row for row in resources['jobs']}
    for _, summary in runs:
        row = successful[summary['slurm_job_id']]
        need(row['state']=='COMPLETED' and row['exit_code']=='0:0' and row['gpus']==1,
             'Successful source-bound software allocation missing')
    analysis = dict(passed=True, completed=True, protocol=original.PROTOCOL, source_sha256=frozen,
        reporter_source_sha256=reporter_frozen, precision_dispatcher=provenance,
        plan_file=str(args.plan), plan_sha256=sha(args.plan),
        runs=[dict(directory=str(d), summary_sha256=sha(d/'summary.json'), audit=a)
              for (d,_),a in zip(runs,results)], resources=resources, selftest=tests,
        no_model_or_head_execution=True, no_fit=True, no_accuracy=True,
        conclusion_scope='Native software, write persistence and fixed-token continuous credit only; no beneficial memory or reasoning claim')
    save(out/'analysis.json', analysis)
    return dict(passed=True, completed=True, phase='report', protocol=original.PROTOCOL,
        source_sha256=frozen, reporter_source_sha256=reporter_frozen, precision_dispatcher=provenance,
        plan_file=str(args.plan), plan_sha256=sha(args.plan), analysis_file=str(out/'analysis.json'),
        analysis_sha256=sha(out/'analysis.json'), models=list(original.MODELS), resources=resources,
        computational_integrity_passed=True, native_replay_passed=True,
        temporal_gradient_passed=True, persistence_restoration_passed=True)


def verify_profile(path):
    """Require both the unchanged native audit contract and this exact dispatcher."""
    path=Path(path); path=path/'summary.json' if path.is_dir() else path
    summary=original.verify_profile(path); analysis=read(summary['analysis_file'])
    need(summary['reporter_source_sha256']==analysis['reporter_source_sha256']==dispatcher_sources(),
         'Independent precision reporter source changed')
    for name,digest in dispatcher_sources().items():
        need(sha(path.parent/'source'/name.replace('/','_'))==digest, 'Dispatcher source copy differs')
    plan=original.verify_plan(summary['plan_file'])
    runs=[original.verify_run(item['directory'],plan) for item in analysis['runs']]
    need(summary['precision_dispatcher']==analysis['precision_dispatcher']==verify_diagnosis(plan,runs),
         'Precision correction provenance changed')
    verify_copy_contract()
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--runs',nargs=2,type=Path,required=True)
    args=parser.parse_args(); native.require_slurm(gpu=False)
    need(original.os.environ.get('SLURM_JOB_PARTITION')=='cpu', 'CPU-only independent reporter')
    job=original.os.environ['SLURM_JOB_ID'];out=original.OUT/('report_precision_'+job)
    out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    frozen=original.snapshot(out); reporter_frozen=dispatcher_sources()
    for name,digest in reporter_frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==digest,'Dispatcher source changed while freezing')
    save(out/'reporter_source_hashes.json',reporter_frozen)
    save(out/'request.json',dict(argv=sys.argv,slurm_job_id=job,phase='report',source_sha256=frozen,
        reporter_source_sha256=reporter_frozen,rule=RULE,original_failure_file=str(FAILURE),
        original_failure_sha256=FAILURE_SHA,diagnostic_file=str(DIAG/'summary.json'),diagnostic_sha256=DIAG_SHA))
    try:
        result=report(args,out,frozen,reporter_frozen)
        need(original.sources()==frozen and dispatcher_sources()==reporter_frozen,'Sources changed during report')
        result.update(slurm_job_id=job,seconds=time.perf_counter()-started)
        save(out/'summary.json',result);print(original.json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),slurm_job_id=job,
            source_sha256=frozen,reporter_source_sha256=reporter_frozen,
            original_failure_file=str(FAILURE),diagnostic_file=str(DIAG/'summary.json'),
            seconds=time.perf_counter()-started,partial_outputs_retained=True))
        raise


if __name__=='__main__':
    main()
