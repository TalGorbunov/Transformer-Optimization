"""CPU-only post-hoc V7 response surface; no training or VLM forward.

The 216 synthetic repeated-state multisets are not MMReD test examples. All
predictions concern the first native token, not EOS-completed accuracy. Native
FP16 norm/head weights are exact; CPU kernels need not match GPU arithmetic.
"""
from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
DATA=Path('/mnt/data/gabriele/gnn_transformer')
BASE=REPO/'outputs/native_aggregation_vlm'
REPORT=BASE/'v7/report_441910/analysis.json'
REFERENCE=BASE/'readout_capacity/run_441720/summary.json'
OWN=('scripts/probe_native_vision_v7_response_surface.py','slurm/native_vision_v7_response_surface.sbatch',
     'gnnformer/parallel_local_aggregation.py')
CATEGORIES=('positive','char_only','room_only','neither')

def need(x,message):
    if not x: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def objsha(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bind(path,digest): need(sha(path)==digest,'Changed input: '+str(path))
def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def thash(tensor):
    import torch
    x=tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()
def state_hash(state):
    return objsha({k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=thash(v)) for k,v in state.items()})
def finite(*values):
    import torch
    need(all(bool(torch.isfinite(x).all()) for x in values),'Nonfinite diagnostic tensor')
def category(character,room,target_character,target_room):
    c,r=character==target_character,room==target_room
    return 'positive' if c and r else ('char_only' if c else ('room_only' if r else 'neither'))

def load_inputs(report_path):
    import torch
    report=read(report_path);need(report['audit_passed'] is True,'Require completed independent V7 report')
    audits=[report['runs'][f'parallel_s{s}'] for s in (8,9)]
    need(audits[0]['cache']==audits[1]['cache'],'Parallel seeds used different caches')
    binding=audits[0]['cache'];bind(binding['file'],binding['sha256']);cache=read(binding['file'])
    need(cache['complete'] and cache['training_only'],'Require complete training-only states')
    bind(cache['plan_file'],cache['plan_sha256']);plan=read(cache['plan_file'])
    need(cache['plan_sha256']==binding['plan_sha256'] and cache['scenes']==plan['scenes'],'Cache plan/scene mismatch')
    source=plan['source_files']['training_manifest'];bind(source['path'],source['sha256']);manifest=read(source['path'])
    samples={r['sid']:r for cell in ('train_N8','train_N16') for r in manifest['splits'][cell]['samples']}
    need(len(samples)==1890 and set(samples)==set(cache['scenes']),'Training scene coverage differs')
    questions=sorted({r['question'] for r in samples.values()})[:2];need(len(questions)==2,'Need two fixed questions')
    labels={};selected_samples={q:[] for q in questions}
    for sid,row in sorted(samples.items()):
        if row['question'] not in questions: continue
        selected_samples[row['question']].append(row);scene=cache['scenes'][sid]
        need(all(scene[k]==row[k] for k in ('gold','question','qa_sha256','content_sha256','n_frames')),'Training scene identity changed')
        path=Path(row['path'])/'qa.txt';bind(path,row['qa_sha256']);lines=path.read_text().splitlines()
        text=[x.strip() for x in lines[lines.index('question:')+1:lines.index('answer:')] if x.strip()]
        states=[ast.literal_eval(x) for x in text if x.startswith('{')]
        need([x for x in text if not x.startswith('{')]==[row['question']]
             and objsha(dict(states=states,question=row['question']))==row['content_sha256'], 'QA semantic identity differs')
        need(len(states)==row['n_frames'] and int(lines[lines.index('answer:')+1])==row['gold'],'QA count differs')
        positives=0
        for i,(state,ids) in enumerate(zip(states,scene['local_feature_ids'])):
            room,people=next(iter(state['rooms'].items()));need(len(people)==1 and state['step_id']==i+1,'Unexpected frame semantics')
            label=category(people[0],room,row['target_character'],row['target_room']);positives+=label=='positive'
            f=plan['features'][ids[0]];pid=f['pair_id'];pair=plan['pairs'][pid]
            need(f['prefix_ids']==[] and f['kind']=='local' and f['question']==row['question']
                 and pair['image_sha256']==row['image_files'][i]['sha256'],'Empty local feature/evidence differs')
            value=dict(category=label,question=row['question'],pair_id=pid,feature_id=ids[0],
                image_sha256=pair['image_sha256'],image_path=pair['image_path'],step_id=i+1,
                character=people[0],room=room)
            if pid in labels: need(labels[pid]==value,'Conflicting repeated pair metadata')
            labels[pid]=value
        need(positives==row['gold'],'Independent local recount differs')
    prototypes={}
    wanted=set()
    for q in questions:
        prototypes[q]={}
        for cat in CATEGORIES:
            choices=sorted((r for r in labels.values() if r['question']==q and r['category']==cat),key=lambda r:r['pair_id'])
            need(choices,'Missing prototype category');prototypes[q][cat]=choices[0]
            bind(choices[0]['image_path'],choices[0]['image_sha256'])
        for row in selected_samples[q]:
            scene=cache['scenes'][row['sid']]
            wanted.update(ids[0] for ids in scene['local_feature_ids']);wanted.add(scene['global_feature_ids'][0])
    files={cache['features'][fid]['file']:cache['features'][fid]['file_sha256'] for fid in wanted};states={}
    for path,digest in files.items():
        bind(path,digest);blob=torch.load(path,map_location='cpu',weights_only=True)
        need(blob['schema_version']==1 and blob['states'].dtype==torch.float16,'Wrong cached native dtype')
        for fid in sorted(wanted):
            loc=cache['features'][fid]
            if loc['file']!=path: continue
            x=blob['states'][loc['row']]
            need(blob['feature_ids'][loc['row']]==fid and x.shape==(3584,) and thash(x)==loc['state_sha256'],'Cached state identity differs')
            finite(x);states[fid]=x.clone()
        del blob
    need(set(states)==wanted,'Incomplete required state coverage')
    return report,audits,cache,plan,questions,selected_samples,prototypes,states


def load_head(reference_path,expected_model):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    ref=read(reference_path)['states_identity'];need(ref['model']==expected_model,'Native head/backbone identity differs')
    model=Path(ref['model']['path'])
    for name,digest in ref['model']['metadata_sha256'].items(): bind(model/name,digest)
    cfg=read(model/'config.json');cfg=cfg.get('text_config',cfg);index=read(model/'model.safetensors.index.json')['weight_map']
    weights={};tensor_sources={}
    for name,key in (('head','lm_head.weight'),('norm','model.norm.weight')):
        path=model/index[key]
        with safe_open(str(path),framework='pt',device='cpu') as archive: weights[name]=archive.get_tensor(key).to(torch.float16).contiguous()
        tensor_sources[name]=dict(file=str(path),key=key,shape=list(weights[name].shape),dtype=str(weights[name].dtype),sha256=thash(weights[name]))
        need(tensor_sources[name]['sha256']==ref['native_'+('head' if name=='head' else 'norm')+'_weight_sha256'],'Native FP16 weight fingerprint differs')
    need(weights['head'].shape==(cfg['vocab_size'],3584) and weights['norm'].shape==(3584,),'Native readout dimensions differ')
    norm=Qwen2RMSNorm(3584,eps=cfg['rms_norm_eps']).to(dtype=torch.float16)
    norm.load_state_dict({'weight':weights['norm']});norm.eval().requires_grad_(False)
    bind(ref['states_file'],ref['states_sha256']);blob=torch.load(ref['states_file'],map_location='cpu',weights_only=True)
    h=blob['hidden'][:4];native=blob['native_logits'][:4].float()
    logits=torch.nn.functional.linear(norm(h),weights['head']).float()
    finite(logits);p=logits.softmax(-1);q=native.softmax(-1)
    replay=dict(rows=4,full_vocabulary_tv_max=float((p-q).abs().sum(-1).mul(.5).max()),
        logits_max_abs_difference=float((logits-native).abs().max()),top1_equal=bool((logits.argmax(-1)==native.argmax(-1)).all()),
        numerical_gate=False,reason='Exact native FP16 weights; CPU kernels can differ from archived GPU kernels')
    module=Path(inspect.getsourcefile(Qwen2RMSNorm))
    provenance=dict(reference_file=str(reference_path),reference_sha256=sha(reference_path),tensor_sources=tensor_sources,
        native_norm_source=str(module),native_norm_source_sha256=sha(module),native_rms_epsilon=cfg['rms_norm_eps'],cpu_replay=replay)
    return norm,weights['head'],provenance
def zero_padding_check(branch,local,g):
    import torch
    zeros=torch.zeros((48,1,branch.hidden_size),dtype=local.dtype)
    null=branch.encode(zeros,g)
    before=branch(local,g,output_dtype=torch.float32)
    after=branch(torch.cat((local,zeros)),g,output_dtype=torch.float32)
    scale=max(1.,float(before.abs().max()))
    error=float((before-after).abs().max())
    result=dict(zero_message_exact=bool((null==0).all()),delta_max_abs_difference=error,
        delta_relative_max_difference=error/scale,fp16_delta_equal=bool(torch.equal(before.half(),after.half())),
        tolerance='FP32 max_abs <= 1e-6 + 1e-5 * max(1, max_abs(original_delta))')
    result['passed']=result['zero_message_exact'] and error<=1e-6+1e-5*scale
    return result


def surface(branch,g,prototype_states,training_states,training_records,norm,head,count_ids):
    import torch
    messages=branch.encode(torch.stack([prototype_states[c] for c in CATEGORIES]).unsqueeze(1),g)[:,0]
    message={cat:messages[i] for i,cat in enumerate(CATEGORIES)}
    train_z=torch.stack([branch.aggregate(branch.encode(x,g))[0] for x in training_states])
    train_norm=train_z.norm(dim=-1);train_k=torch.tensor([r['gold'] for r in training_records])
    rows=[];zs=[]
    for cat in CATEGORIES[1:]:
        for n in (16,64):
            for k in range(9):
                rows.append(dict(negative_category=cat,n_frames=n,k=k))
                zs.append(k*message['positive']+(n-k)*message[cat])
    z=torch.stack(zs);global_batch=g.expand(len(rows),-1)
    delta32=branch.decode(z,global_batch,output_dtype=torch.float32)
    carry32=global_batch.float()+delta32
    delta=delta32.to(torch.float16);carry=global_batch+delta
    normalized=norm(carry);logits=torch.nn.functional.linear(normalized,head)
    finite(messages,train_z,z,delta32,carry32,carry,normalized,logits)
    need(logits.dtype==carry.dtype==normalized.dtype==torch.float16,'Native diagnostic arithmetic dtype changed')
    lf=logits.float();numeric=lf[:,count_ids];logz=torch.logsumexp(lf,-1)
    best_values,best_ids=lf.topk(2,dim=-1)
    distances=torch.cdist(z,train_z)
    scale=max(float(train_norm.median()),1e-12)
    for i,row in enumerate(rows):
        k=row['k'];target=count_ids[k];mask=torch.arange(9)!=k
        nearest=int(distances[i].argmin());same=torch.where(train_k==k)[0]
        same_index=int(same[distances[i,same].argmin()])
        other=float(best_values[i,1] if int(best_ids[i,0])==target else best_values[i,0])
        row.update(first_token_id=int(best_ids[i,0]),target_token_id=target,
            first_token_correct=int(best_ids[i,0])==target,numeral_argmax=int(numeric[i].argmax()),
            numeral_argmax_correct=int(numeric[i].argmax())==k,
            raw_first_token_nll=float(logz[i]-lf[i,target]),
            numeral_probability_mass=float(torch.exp(torch.logsumexp(numeric[i],-1)-logz[i])),
            gold_minus_best_other_numeral=float(numeric[i,k]-numeric[i,mask].max()),
            gold_minus_best_other_vocabulary=float(lf[i,target])-other,
            z_norm=float(z[i].norm()),delta_norm=float(delta32[i].norm()),
            pre_rms_carry_norm=float(carry[i].float().norm()),post_rms_carry_norm=float(normalized[i].float().norm()),
            nearest_training_sid=training_records[nearest]['sid'],nearest_training_l2=float(distances[i,nearest]),
            nearest_training_l2_over_median_train_z_norm=float(distances[i,nearest])/scale,
            nearest_same_k_sid=training_records[same_index]['sid'],nearest_same_k_l2=float(distances[i,same_index]),
            coordinates_inside_train_minmax_fraction=float(((z[i]>=train_z.min(0).values)&(z[i]<=train_z.max(0).values)).float().mean()))
    lookup={(r['negative_category'],r['n_frames'],r['k']):i for i,r in enumerate(rows)}
    geometry=[];corrections=[];adjacent=[]
    for cat in CATEGORIES[1:]:
        d=message['positive']-message[cat];background=message[cat]
        denom=float(d.norm()*background.norm())
        sv=torch.linalg.svdvals(torch.stack((d,background)))
        geometry.append(dict(negative_category=cat,positive_norm=float(message['positive'].norm()),
            negative_norm=float(background.norm()),positive_minus_negative_norm=float(d.norm()),
            signal_background_cosine=None if denom==0 else float((d@background)/denom),
            signal_background_singular_values=sv.tolist(),
            singular_value_ratio=None if float(sv[0])==0 else float(sv[1]/sv[0]),
            positive_negative_messages_exactly_equal=bool(torch.equal(message['positive'],background))))
        for k in range(9):
            lo,hi=lookup[cat,16,k],lookup[cat,64,k]
            corrected=z[hi:hi+1]-48*background
            correction_carry=g+branch.decode(corrected,g,output_dtype=torch.float16)
            correction_norm=norm(correction_carry)
            corrections.append(dict(negative_category=cat,k=k,
                added_negative_z_norm=float((48*background).norm()),
                z_difference_minus_48_negative_maxabs=float((z[hi]-z[lo]-48*background).abs().max()),
                corrected_z_minus_N16_maxabs=float((corrected[0]-z[lo]).abs().max()),
                corrected_carry_equals_N16=bool(torch.equal(correction_carry[0],carry[lo])),
                corrected_normalized_carry_equals_N16=bool(torch.equal(correction_norm[0],normalized[lo])),
                first_token_N16_correct=rows[lo]['first_token_correct'],first_token_N64_correct=rows[hi]['first_token_correct'],
                scope='Algebraic removal of added prototype negatives; not an inference method or an additional scored point'))
        for n in (16,64):
            for k in range(8):
                a,b=lookup[cat,n,k],lookup[cat,n,k+1]
                adjacent.append(dict(negative_category=cat,n_frames=n,k_from=k,k_to=k+1,
                    z_l2=float((z[b]-z[a]).norm()),fp32_carry_l2=float((carry32[b]-carry32[a]).norm()),
                    fp16_carry_l2=float((carry[b].float()-carry[a].float()).norm()),
                    normalized_carry_l2=float((normalized[b].float()-normalized[a].float()).norm()),
                    fp16_carry_exactly_equal=bool(torch.equal(carry[a],carry[b])),
                    normalized_carry_exactly_equal=bool(torch.equal(normalized[a],normalized[b])),
                    full_vocabulary_logits_exactly_equal=bool(torch.equal(logits[a],logits[b]))))
    cat='neither';i=lookup[cat,16,2]
    local=torch.cat((prototype_states['positive'].expand(2,-1),prototype_states[cat].expand(14,-1))).unsqueeze(1)
    actual=branch(local,g,output_dtype=torch.float32)
    algebra_error=float((actual[0]-delta32[i]).abs().max())
    algebra_scale=max(1.,float(delta32[i].abs().max()))
    algebra=dict(delta_max_abs_difference=algebra_error,
        passed=algebra_error<=1e-6+1e-5*algebra_scale,
        tolerance='FP32 max_abs <= 1e-6 + 1e-5 * max(1, max_abs(algebraic_delta))')
    padding=zero_padding_check(branch,training_states[0],g)
    diagnostics=dict(rows=rows,geometry=geometry,negative_removal=corrections,adjacent_counts=adjacent,
        actual_zero_padding=padding,actual_multiset_vs_algebra=algebra,
        training_reference=dict(n=len(training_records),sids=[r['sid'] for r in training_records],
            counts=[r['gold'] for r in training_records],n_frames=[r['n_frames'] for r in training_records],
            z_norm_min=float(train_norm.min()),z_norm_median=float(train_norm.median()),z_norm_max=float(train_norm.max()),
            scope='Genuine model-training scenes for the same question; descriptive support geometry, not held-out validation'))
    tensors=dict(messages=messages,training_z=train_z,z=z,delta_fp32=delta32,carry_fp32=carry32,
        carry_fp16=carry,normalized_carry_fp16=normalized,raw_logits=logits)
    return diagnostics,tensors


def self_test():
    import torch
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260929)
        branch=ParallelLocalAggregation(8,rank=3).eval().requires_grad_(False)
        branch.up.weight.normal_(0,.1)
        g=torch.randn(1,8,dtype=torch.float16);x=torch.randn(3,1,8,dtype=torch.float16)
        result=zero_padding_check(branch,x,g);need(result['passed'],'Actual zero padding failed')
        need(bool((branch.encode(torch.zeros_like(x),g)==0).all()),'Zero local message is not exactly null')
        messages=branch.encode(x,g);z=branch.aggregate(messages)
        need(torch.allclose(branch(x,g,output_dtype=torch.float32),branch.decode(z,g,output_dtype=torch.float32)), 'Encode/decode graph differs')
        positive,negative=messages[0,0],messages[1,0]
        z16=2*positive+14*negative;z64=2*positive+62*negative
        need(torch.allclose(z64-z16,48*negative,rtol=1e-5,atol=1e-6),'Repeated-negative identity differs')
        need(thash(torch.tensor([[1.,2.]]))==hashlib.sha256(torch.tensor([[1.,2.]]).view(torch.uint8).numpy().tobytes()).hexdigest(),'Tensor hash differs')
    return dict(passed=True,tests=['exact_zero_messages','actual_zero_padding','encode_decode_equivalence','negative_extension_identity','tensor_hash'],zero_padding=result)
def execute(args,out,frozen,tests):
    import torch
    from transformers import AutoTokenizer
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    report,audits,cache,plan,questions,samples,prototypes,states=load_inputs(args.report)
    norm,head,head_provenance=load_head(args.head_reference,cache['model'])
    tokenizer=AutoTokenizer.from_pretrained(cache['model']['path'],use_fast=False,local_files_only=True)
    count_ids=[plan['count_token_ids'][str(k)] for k in range(9)]
    need(len(set(count_ids))==9 and all(tokenizer(str(k),add_special_tokens=False)['input_ids']==[count_ids[k]] for k in range(9)), 'Native single-token count identities changed')
    groups=[];raw_groups=[];bindings=[]
    for audit in audits:
        seed=audit['seed'];directory=Path(audit['run_directory'])
        bind(directory/'config.json',audit['config_sha256']);bind(directory/'summary.json',audit['summary_sha256'])
        config=read(directory/'config.json');summary=read(directory/'summary.json');selected=summary['selected']
        need(config['arm']=='parallel' and config['seed']==seed and seed in (8,9),'Wrong selected run')
        need(selected['step']==audit['selected_step'] and selected['checkpoint_sha256']==audit['selected_checkpoint_sha256']
             and selected['parameter_sha256']==audit['selected_parameter_sha256'],'Selected checkpoint changed')
        bind(REPO/'gnnformer/parallel_local_aggregation.py',config['source_sha256']['gnnformer/parallel_local_aggregation.py'])
        need(config['cache_binding']['sha256']==audit['cache']['sha256'],'Run/cache binding differs')
        bind(selected['checkpoint'],selected['checkpoint_sha256'])
        blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        need(blob['config']==config and blob['step']==selected['step'] and state_hash(blob['branch'])==selected['parameter_sha256'],'Checkpoint tensors/metadata changed')
        branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(blob['branch'],strict=True)
        need(sum(p.numel() for p in branch.parameters())==1041600 and all(p.dtype==torch.float32 for p in branch.parameters()),'Branch architecture/dtype differs')
        bindings.append(dict(seed=seed,run_directory=str(directory),config_sha256=audit['config_sha256'],
            summary_sha256=audit['summary_sha256'],checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
            parameter_sha256=selected['parameter_sha256'],step=selected['step']))
        for q in questions:
            records=samples[q]
            gids={cache['scenes'][r['sid']]['global_feature_ids'][0] for r in records}
            need(len(gids)==1,'Question-global empty state unexpectedly depends on N/K')
            gid=next(iter(gids));need(plan['features'][gid]['prefix_ids']==[],'Global state contains an answer')
            g=states[gid].unsqueeze(0)
            local=[torch.stack([states[ids[0]] for ids in cache['scenes'][r['sid']]['local_feature_ids']]).unsqueeze(1) for r in records]
            chosen={cat:states[prototypes[q][cat]['feature_id']] for cat in CATEGORIES}
            result,tensors=surface(branch,g,chosen,local,records,norm,head,count_ids)
            for row in result['rows']: row['first_token_text']=tokenizer.decode([row['first_token_id']],skip_special_tokens=False)
            result.update(seed=seed,question=q,global_feature_id=gid,prototypes=prototypes[q])
            groups.append(result);raw_groups.append(dict(seed=seed,question=q,rows=result['rows'],**tensors))
            print(json.dumps(dict(seed=seed,question=q,synthetic_points=len(result['rows']),zero_padding=result['actual_zero_padding']['passed'])),flush=True)
        del branch,blob
    need(len(groups)==4 and sum(len(g['rows']) for g in groups)==216,'Fixed response-surface coverage differs')
    rawdir=DATA/'native_aggregation_vlm_v7_response_surface'/f'run_{os.environ["SLURM_JOB_ID"]}'
    rawdir.mkdir(parents=True,exist_ok=False);rawfile=rawdir/'tensors.pt'
    torch.save(dict(schema_version=1,groups=raw_groups,count_token_ids=count_ids),rawfile)
    passed=all(g['actual_zero_padding']['passed'] and g['actual_multiset_vs_algebra']['passed'] for g in groups)
    analysis=dict(schema_version=1,passed=passed,diagnostic_only=True,no_fit=True,vlm_forward_calls=0,
        native_head_synthetic_rows=216,native_head_reference_replay_rows=4,
        source_sha256=frozen,self_tests=tests,report_file=str(args.report),report_sha256=sha(args.report),
        training_cache_file=audits[0]['cache']['file'],training_cache_sha256=audits[0]['cache']['sha256'],
        training_plan_file=cache['plan_file'],training_plan_sha256=cache['plan_sha256'],
        selected_checkpoints=bindings,head=head_provenance,count_token_ids=count_ids,
        selection_rule='First two lexicographic training questions; lowest pair_id independently within positive, char_only, room_only, neither; no outcome selection',
        questions=questions,groups=groups,raw_file=str(rawfile),raw_sha256=sha(rawfile),
        limitations=[
            'Synthetic repeated images retain the prototype Step label; they are outside the original distinct-Step generator law.',
            'All source states and reference sums are from model training; no held-out native local states are claimed.',
            'N64 replication changes set size while keeping local-state distribution fixed; it does not test unseen Step-label features.',
            'These are raw first-token predictions, not generation, EOS completion, or real MMReD accuracy.',
            'Exact FP16 weight fingerprints and native RMS implementation are verified; CPU kernels can differ from GPU execution.',
            'Negative removal is an algebraic diagnostic using known prototype categories, not a deployable cancellation rule.',
            'Nonzero count direction proves distinguishability on a fixed-N synthetic ray in exact arithmetic, not trained readout calibration.',
            'Coordinate bounds and nearest-training distances are descriptive geometry, not a calibrated distribution test.'])
    save(out/'analysis.json',analysis)
    lines=['# V7 synthetic response surface','',
        '216 fixed synthetic first-token points; no fitting and no VLM forward. This is a post-hoc diagnostic, not a new efficacy evaluation.','',
        '| Seed | Question | Synthetic N16 first-token correct | Synthetic N64 first-token correct | Raw-zero padding |',
        '|---:|---|---:|---:|---|']
    for group in groups:
        totals={n:sum(r['first_token_correct'] for r in group['rows'] if r['n_frames']==n) for n in (16,64)}
        lines.append(f"| {group['seed']} | {group['question']} | {totals[16]}/27 | {totals[64]}/27 | {group['actual_zero_padding']['passed']} |")
    lines+=['','All 216 native FP16 raw-logit rows, messages, genuine training sums and pre/post-RMS carries are archived. Full-vocabulary predictions are separate from restricted numeral argmax diagnostics.','',
        'The native FP16 norm/head tensor hashes match the earlier actual-model capture. The four-row CPU-versus-GPU replay is descriptive and retained in analysis.json.','',
        'A failure on a fixed-K negative extension can be localized to the added negative message direction. If different counts survive in z but collapse in the native carry or acquire incorrect numeral ordering, the failure is downstream of that synthetic statistic. Neither observation proves an intrinsic model capacity limit.','',
        'Successful synthetic N64 decoding does not establish real N64 generalization: genuine later Step labels and new local features are absent here.','',
        '[Complete diagnostic and input provenance](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=passed,diagnostic_only=True,no_fit=True,vlm_forward_calls=0,synthetic_points=216,
        analysis_file=str(out/'analysis.json'),analysis_sha256=sha(out/'analysis.json'),raw_file=str(rawfile),raw_sha256=sha(rawfile))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true')
    p.add_argument('--report',type=Path,default=REPORT);p.add_argument('--head-reference',type=Path,default=REFERENCE)
    p.add_argument('--output',type=Path);args=p.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'All tensor/model-metadata work requires CPU Slurm')
    import torch
    torch.set_num_threads(min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))));torch.set_num_interop_threads(1)
    begin=time.perf_counter();mode='selftest' if args.self_test else 'run'
    out=args.output or BASE/'v7/response_surface'/f'{mode}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    frozen={name:sha(REPO/name) for name in OWN}
    for name in OWN: (out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',frozen)
    with torch.inference_mode():
        tests=self_test()
        result=dict(passed=True,tests=tests,diagnostic_only=True,vlm_forward_calls=0) if args.self_test else execute(args,out,frozen,tests)
    for name,digest in frozen.items(): bind(REPO/name,digest)
    result.update(source_sha256=frozen,elapsed_seconds=time.perf_counter()-begin,slurm_job_id=os.environ['SLURM_JOB_ID'])
    save(out/'summary.json',result)
    (out/'INDEX.md').write_text('# V7 response-surface diagnostic\n\n[Summary](summary.json) · [Frozen source](source_hashes.json)\n')
    print(json.dumps(dict(passed=result['passed'],summary_file=str(out/'summary.json'),seconds=result['elapsed_seconds'])),flush=True)
    if not result['passed']: raise SystemExit('Diagnostic software equivalence check failed; all observations retained')

if __name__=='__main__': main()
