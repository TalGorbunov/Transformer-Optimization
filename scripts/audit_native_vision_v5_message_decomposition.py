"""Optional descriptive decomposition of V5 saved first-prefill frame messages.

For short messages A_i and mapped long messages B_j, let J map the original
16 semantic frames into the longer scene, d_L be 1 (SUM) or L (MEAN), and
S=sum_i A_i. Then
  M_L-M_16 = sum_i(B_J[i]-A_i)/d_L + sum_{j not in J}B_j/d_L
             + S*(1/d_L-1/d_16).
The three terms are mapped semantic-frame drift, added-negative contribution,
and deterministic normalization change. The last term is zero for SUM. The
first plus third is the change in the normalized original-frame contribution.

Mapped frames preserve room/character semantics but visible Step labels are
renumbered after interleaving. Thus drift combines changed Step pixels, query
and context changes, and numerical effects. This is not a causal isolation of
query dependence. All 108 anchors and both extensions enter, regardless of
model correctness; no new training, classifier, primary criterion or efficacy
subset is introduced. Exact image-byte matches are counted, never selected.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

REPO=Path(__file__).resolve().parents[1]
V5=REPO/'outputs/native_aggregation_vlm/v5'
DATA=Path('/mnt/data/gabriele/gnn_transformer/v4_diversity')
DIAGNOSTICS=DATA.parent/'v5_diagnostics'
CHECKPOINTS=Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v5')
SOURCES=('scripts/audit_native_vision_v5_message_decomposition.py',
         'slurm/native_vision_v5_message_decomposition.sbatch')


def need(value,message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def cosine(a,b,np):
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.clip(np.dot(a,b)/denominator,-1,1)) if denominator else None


def decomposition(short,long,positions,condition,np):
    need(condition in ('sum','mean'),'Unregistered pooling')
    short,long=np.asarray(short,dtype=np.float64),np.asarray(long,dtype=np.float64)
    small,large=len(short),len(long)
    need(short.ndim==long.ndim==2 and short.shape[1]==long.shape[1] and large>small,
         'Invalid message dimensions')
    need(len(positions)==small and len(set(positions))==small and
         all(type(i) is int and 0<=i<large for i in positions) and positions==sorted(positions),
         'Anchor map must be ordered, unique and complete')
    need(bool(np.isfinite(short).all() and np.isfinite(long).all()),'Nonfinite messages')
    mask=np.ones(large,dtype=bool);mask[positions]=False
    denominator_short=1 if condition=='sum' else small
    denominator_long=1 if condition=='sum' else large
    old=short.sum(0)
    drift=(long[positions]-short).sum(0)/denominator_long
    extra=long[mask].sum(0)/denominator_long
    normalization=old*(1/denominator_long-1/denominator_short)
    delta=long.sum(0)/denominator_long-old/denominator_short
    reconstructed=drift+extra+normalization
    need(np.allclose(delta,reconstructed,rtol=1e-12,atol=1e-12),'Additive decomposition failed')
    vectors=dict(mapped_semantic_drift=drift,added_negative_contribution=extra,
                 normalization_change=normalization,normalized_original_change=drift+normalization,
                 total_merged_change=delta)
    similarities={f'{name}_vs_total':cosine(value,delta,np) for name,value in vectors.items()
                  if name!='total_merged_change'}
    similarities.update(drift_vs_added=cosine(drift,extra,np),
                        original_change_vs_added=cosine(drift+normalization,extra,np))
    frame_cos=[cosine(a,b,np) for a,b in zip(short,long[positions])]
    return dict(norms={name:float(np.linalg.norm(value)) for name,value in vectors.items()},
        cosines=similarities,algebra_reconstruction_l2=float(np.linalg.norm(delta-reconstructed)),
        algebra_reconstruction_max_abs=float(np.max(np.abs(delta-reconstructed))),
        mapped_per_frame_drift_l2=[float(x) for x in np.linalg.norm(long[positions]-short,axis=1)],
        mapped_per_frame_cosine=frame_cos),delta


def scalar_stats(values,np):
    finite=[float(v) for v in values if v is not None]
    need(all(math.isfinite(v) for v in finite),'Nonfinite summary statistic')
    return dict(n=len(values),defined=len(finite),mean=float(np.mean(finite)) if finite else None,
        median=float(np.median(finite)) if finite else None,minimum=min(finite) if finite else None,
        maximum=max(finite) if finite else None)


def audit_prediction_identity(row,source):
    """Bind predictions to the audited manifest, which owns the frame maps.

    Frozen V5 rows serialize identity and family fields, but not frame maps.
    A future row that does supply a map must agree exactly with the manifest.
    Map validity and semantic/image correspondence are checked by audit_pairs.
    """
    required=('path','sid','n_frames','gold','pair_id','test_family')
    need(all(key in row and key in source and row[key]==source[key] for key in required) and
         row.get('mode')=='all' and row.get('tag')=='test' and
         row.get('cell')==f"length_N{source['n_frames']}",
         'Paired prediction differs from manifest/evaluation mode')
    for key in ('anchor_positions','parent_positions'):
        need(key in source,'Audited manifest lacks a frame map')
        if key in row:
            need(row[key]==source[key],f'Optional prediction {key} conflicts with audited manifest')


def self_test(np):
    short=np.array([[1.,2.],[3.,4.]])
    long=np.array([[2.,3.],[7.,8.],[5.,6.],[9.,10.]])
    for condition in ('sum','mean'):
        result,delta=decomposition(short,long,[0,2],condition,np)
        expected=long.sum(0)-short.sum(0) if condition=='sum' else long.mean(0)-short.mean(0)
        np.testing.assert_allclose(delta,expected)
        need(result['algebra_reconstruction_max_abs']==0,'Simple exact identity failed')
    # Unchanged originals and zero inserted messages still dilute a MEAN.
    padded=np.array([[1.,2.],[0.,0.],[3.,4.],[0.,0.]])
    summed,_=decomposition(short,padded,[0,2],'sum',np)
    averaged,delta=decomposition(short,padded,[0,2],'mean',np)
    need(summed['norms']['total_merged_change']==0 and averaged['norms']['mapped_semantic_drift']==0 and
         averaged['norms']['added_negative_contribution']==0,'No-drift/zero-addition check failed')
    np.testing.assert_allclose(delta,-short.mean(0)/2)
    need(averaged['norms']['normalization_change']==averaged['norms']['total_merged_change'],
         'MEAN normalization was omitted')
    zero,_=decomposition(np.zeros((2,3)),np.zeros((4,3)),[0,2],'sum',np)
    need(all(value is None for value in zero['cosines'].values()),'Zero-vector cosines must be undefined')
    for mapping,condition in (([0,0],'sum'),([2,0],'sum'),([0,4],'sum'),([0,2],'max')):
        try:
            decomposition(short,long,mapping,condition,np)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid map/pooling accepted')
    rng=np.random.default_rng(20260916)
    for large in (32,64):
        for condition in ('sum','mean'):
            positions=sorted(int(i) for i in rng.choice(large,16,replace=False))
            result,_=decomposition(rng.normal(size=(16,96)),rng.normal(size=(large,96)),positions,condition,np)
            need(result['algebra_reconstruction_l2']<1e-12,'Random additive identity failed')
    source=dict(path='/audited/sample',sid='sample',n_frames=32,gold=3,pair_id='pair',
                test_family='length',anchor_positions=list(range(16)),parent_positions=list(range(16)))
    row={key:source[key] for key in ('path','sid','n_frames','gold','pair_id','test_family')}
    row.update(mode='all',tag='test',cell='length_N32')
    audit_prediction_identity(row,source)  # The exact frozen V5 schema omits maps.
    audit_prediction_identity(dict(row,anchor_positions=source['anchor_positions'],
                                   parent_positions=source['parent_positions']),source)
    invalid=[dict(row,anchor_positions=None),dict(row,anchor_positions=list(range(1,17))),
             dict(row,parent_positions=[]),dict(row,path='/other/sample'),dict(row,pair_id='other'),
             dict(row,mode='off'),dict(row,tag='dev'),dict(row,cell='length_N64'),
             {key:value for key,value in row.items() if key!='sid'}]
    for bad in invalid:
        try:
            audit_prediction_identity(bad,source)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid prediction identity/optional frame map accepted')
    print('PASS: SUM/MEAN identities, normalization dilution, zero cosines, map/pooling rejection '
          'and frozen prediction schema/identity checks',flush=True)


def audit_source_sample(row,hash_cache):
    path=Path(row['path'])
    need(path.resolve()==DATA/'mmred_vfiltered'/f"seq_len_{row['n_frames']}"/'test'/row['sid'],
         'Manifest sample escaped canonical test data')
    qa=path/'qa.txt';need(digest(qa)==row['qa_sha256'],'QA hash changed')
    text=qa.read_text().splitlines()
    begin,end=text.index('question:'),text.index('answer:')
    block=[line.strip() for line in text[begin+1:end] if line.strip()]
    states=[ast.literal_eval(line) for line in block if line.startswith('{')]
    questions=[line for line in block if not line.startswith('{')]
    need(questions==[row['question']] and int(text[end+1])==row['gold'] and
         len(states)==row['n_frames'],'QA question/count/length differs')
    normalized=json.dumps(dict(states=states,question=row['question']),sort_keys=True,separators=(',',':'))
    need(hashlib.sha256(normalized.encode()).hexdigest()==row['content_sha256'],'Semantic content hash differs')
    frames=[]
    for index,state in enumerate(states):
        need(state['step_id']==index+1 and len(state['rooms'])==1,'Unexpected Step/room state')
        room,characters=next(iter(state['rooms'].items()))
        need(len(characters)==1,'Expected one character per frame')
        frames.append((characters[0],room))
    target=(row['target_character'],row['target_room'])
    need(sum(frame==target for frame in frames)==row['gold'],'Recounted gold differs')
    need(len(row['image_files'])==len(frames),'Image coverage differs')
    for index,image in enumerate(row['image_files']):
        image_path=Path(image['path']);need(image_path==path/f'{index:03d}.png','Image order/path differs')
        stat=image_path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        need(stat.st_size==image['bytes'],'Image byte size differs')
        if key not in hash_cache:hash_cache[key]=digest(image_path)
        need(hash_cache[key]==image['sha256'],'Actual image bytes differ from manifest')
    return frames


def audit_pairs(manifest,np):
    need(manifest['schema_version']==1 and Path(manifest['dataset_root'])==DATA,'Wrong main data root/schema')
    groups={};hash_cache={};semantics={}
    for n in (16,32,64):
        records=manifest['splits'][f'test_N{n}']['samples']
        need(len(records)==108,'Expected all108 anchors per length')
        for row in records:
            need(row['n_frames']==n and row['split']=='test' and row['pair_id']==row['anchor_id'],
                 'Invalid family identity')
            group=groups.setdefault(row['pair_id'],{})
            need(n not in group,'Duplicate pair/length');group[n]=row
            semantics[row['sid']]=audit_source_sample(row,hash_cache)
    need(len(groups)==108 and all(set(group)=={16,32,64} for group in groups.values()),'Incomplete anchor families')
    pairs=[]
    for pair_id,group in groups.items():
        short=group[16];base=semantics[short['sid']]
        need(short['anchor_positions']==list(range(16)),'N16 is not its own ordered anchor')
        for n in (32,64):
            long=group[n];positions=long['anchor_positions'];parent=group[n//2]
            need(len(positions)==16 and len(set(positions))==16 and positions==sorted(positions) and
                 all(type(i) is int and 0<=i<n for i in positions),'Invalid anchor positions')
            need(long['parent_n_frames']==n//2 and len(long['parent_positions'])==n//2 and
                 long['parent_positions']==sorted(set(long['parent_positions'])) and
                 all(type(i) is int and 0<=i<n for i in long['parent_positions']), 'Invalid immediate-parent mapping')
            need([long['parent_positions'][i] for i in parent['anchor_positions']]==positions,
                 'Direct and composed anchor mappings disagree')
            need(all(long[k]==short[k] for k in ('pair_id','anchor_id','gold','question','target_character','target_room')),
                 'Paired target/question/gold changed')
            frames=semantics[long['sid']]
            need([frames[i] for i in positions]==base and
                 [frames[i] for i in long['parent_positions']]==semantics[parent['sid']],
                 'Mapped character/room semantics changed')
            added=[i for i in range(n) if i not in set(positions)]
            target=(short['target_character'],short['target_room'])
            need(all(frames[i]!=target for i in added),'An inserted frame is positive')
            byte_matches=[short['image_files'][i]['sha256']==long['image_files'][j]['sha256']
                          for i,j in enumerate(positions)]
            same_steps=[i==j for i,j in enumerate(positions)]
            need(byte_matches==same_steps,'Unexpected image identity beyond visible Step renumbering')
            pairs.append(dict(pair_id=pair_id,n_frames=n,short_sid=short['sid'],long_sid=long['sid'],
                gold=short['gold'],anchor_positions=positions,added_positions=added,
                mapped_frames=16,byte_identical_mapped_frames=sum(byte_matches),
                changed_step_mapped_frames=16-sum(same_steps),
                short_image_sha256=[image['sha256'] for image in short['image_files']],
                mapped_long_image_sha256=[long['image_files'][j]['sha256'] for j in positions],
                semantic_frame_identity_verified=True,all_added_frames_negative=True))
    return pairs,dict(samples=324,anchors=108,comparisons=216,
        per_length_comparisons={str(n):108 for n in (32,64)},
        actual_unique_image_inodes_hashed=len(hash_cache),
        all_qa_content_gold_and_image_hashes_verified=True,all_parent_and_anchor_compositions_verified=True,
        mapped_frame_comparisons=216*16,byte_identical_mapped_frames=sum(p['byte_identical_mapped_frames'] for p in pairs),
        changed_step_mapped_frames=sum(p['changed_step_mapped_frames'] for p in pairs))


def load_diagnostic(row,canonical,run,condition,torch,np):
    path=Path(row['branch_diagnostics_path'])
    need(path.resolve()==DIAGNOSTICS/run.name/f"{row['sid']}.pt" and
         canonical['path']==str(path) and canonical['sha256']==row['branch_diagnostics_sha256']==digest(path),
         'Saved diagnostic path/hash differs from canonical verified report')
    value=torch.load(path,map_location='cpu',weights_only=True);n=row['n_frames']
    shapes=dict(query_position=(),image_ends=(n,),visible=(n,),frame_messages=(n,96),
        frame_message_norms=(n,),merged_message=(96,),residual=(3584,),native_output_norm=())
    need(set(value)==set(shapes) and all(isinstance(value[k],torch.Tensor) and tuple(value[k].shape)==shape and
         bool(torch.isfinite(value[k]).all()) for k,shape in shapes.items()),'Diagnostic shape/finite audit failed')
    need(value['query_position'].dtype==torch.int64 and int(value['query_position'])==row['prompt_tokens']-1 and
         value['image_ends'].dtype==torch.int64 and value['visible'].dtype==torch.bool and
         bool(value['visible'].all()) and bool((value['image_ends']>=0).all()) and
         bool((value['image_ends'][1:]>value['image_ends'][:-1]).all()) and
         torch.equal(value['visible'],value['query_position']>value['image_ends']), 'First-prefill visibility differs')
    messages=value['frame_messages'].double().numpy();merged=value['merged_message'].double().numpy()
    # Use the frozen report's float32 reduction for its recorded merge tolerance.
    frames32=value['frame_messages'].float()
    merged32=frames32.sum(0) if condition=='sum' else frames32.mean(0)
    need(torch.allclose(value['merged_message'].float(),merged32,rtol=1e-4,atol=1e-5) and
         torch.allclose(value['frame_message_norms'].float(),frames32.norm(dim=-1),rtol=1e-5,atol=1e-6),
         'Saved merge/frame norm differs')
    residual=float(value['residual'].float().norm());native=float(value['native_output_norm'])
    need(native>=0,'Negative native attention norm')
    ratio=residual/native if native else None
    for key,expected in (('branch_residual_norm',residual),('native_attention_output_norm',native),
                         ('branch_to_native_attention_norm_ratio',ratio)):
        need((row[key] is None and expected is None) or
             (expected is not None and row[key] is not None and math.isclose(row[key],expected,rel_tol=1e-5,abs_tol=1e-6)),
             'Saved residual/native ratio differs')
    return dict(messages=messages,merged=merged,ratio=ratio,residual_norm=residual,native_norm=native)


def summarize(pairs,np):
    result=dict(comparisons=len(pairs),anchors=len({r['pair_id'] for r in pairs}),
        byte_identical_mapped_frames=sum(r['byte_identical_mapped_frames'] for r in pairs),
        changed_step_mapped_frames=sum(r['changed_step_mapped_frames'] for r in pairs))
    for category in ('norms','cosines'):
        result[category]={key:scalar_stats([row[category][key] for row in pairs],np) for key in pairs[0][category]}
    for key in ('algebra_reconstruction_l2','algebra_reconstruction_max_abs','saved_merge_reconstruction_l2',
                'saved_merge_reconstruction_max_abs'):
        result[key]=scalar_stats([row[key] for row in pairs],np)
    result['mapped_per_frame_drift_l2']=scalar_stats([v for row in pairs for v in row['mapped_per_frame_drift_l2']],np)
    result['mapped_per_frame_cosine']=scalar_stats([v for row in pairs for v in row['mapped_per_frame_cosine']],np)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=V5/'analysis.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    need(bool(os.environ.get('SLURM_JOB_ID')),'Run this CPU diagnostic through Slurm')
    import numpy as np
    self_test(np)
    if args.self_test:return
    import torch
    torch.set_num_threads(4)
    started=time.monotonic()
    need(args.analysis.resolve()==V5/'analysis.json','Require the canonical completed V5 analysis')
    analysis_sha=digest(args.analysis);analysis=read(args.analysis)
    need(set(analysis['runs'])=={'sum','mean'},'Incomplete canonical V5 conditions')
    for name,sha in analysis['analysis_source_sha256'].items():
        need(digest(V5/'analysis_code'/name.replace('/','_'))==sha,'Canonical report source snapshot changed')
    sources={name:digest(REPO/name) for name in SOURCES}
    manifest_path=DATA/'main_manifest.json';manifest_sha=digest(manifest_path);manifest=read(manifest_path)
    pairs,data_audit=audit_pairs(manifest,np)
    source_rows={row['sid']:row for n in (16,32,64) for row in manifest['splits'][f'test_N{n}']['samples']}
    all_records=[];summaries={};provenance={};ratio_summaries={}
    for condition in ('sum','mean'):
        summaries[condition]={};provenance[condition]={};ratio_summaries[condition]={}
        need(set(analysis['runs'][condition])=={'4','5'},'Require both fixed seeds')
        for seed in (4,5):
            info=analysis['conditions'][condition][str(seed)]
            run=Path(analysis['runs'][condition][str(seed)]).resolve()
            need(run.parent==V5/'main'/condition/f'seed{seed}','Noncanonical run directory')
            config=read(run/'config.json');summary=read(run/'summary.json')
            need(config['run_id']==summary['run_id']==run.name and config['condition']==summary['condition']==condition and
                 config['seed']==seed and config['profile'] is False and summary['profile'] is False and
                 config['architecture']['protocol']=='vision_v5_extensive' and config['architecture']['rank']==96 and
                 config['architecture']['operator']['merge']==condition,'Run/model/seed/pooling identity differs')
            need(config['manifest_sha256']==info['manifest_sha256']==manifest_sha==digest(run/'staged_manifest.json'),
                 'Run/report/canonical manifest differs')
            expected=analysis['training_source_sha256']
            need(config['code_sha256']==read(run/'source_hashes.json')==expected,'Training source identity differs')
            for name,sha in expected.items():
                need(digest(run/'code'/name.replace('/','_'))==sha,'Training source snapshot changed')
            checkpoint=Path(info['selected_checkpoint'])
            need(checkpoint.resolve()==CHECKPOINTS/run.name/'best.pt' and str(checkpoint)==summary['selected_checkpoint'] and
                 digest(checkpoint)==info['selected_checkpoint_sha256'],'Selected checkpoint changed')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            need(saved['architecture']==config['architecture'] and saved['epoch']==info['selected_epoch'] and
                 all(saved['config'][key]==config[key] for key in
                     ('run_id','condition','seed','architecture','code_sha256','manifest_sha256')),
                 'Checkpoint/model/config identity differs')
            del saved
            rows=read(run/'predictions.json')
            need(len(rows)==452 and len({r['sid'] for r in rows})==452 and
                 Counter(row['cell'] for row in rows)==Counter(length_N16=108,length_N32=108,length_N64=108,
                     unseen_count_N32=64,unseen_count_N64=64),'All452 predictions must be present')
            canonical={row['sid']:row for row in info['prefill_diagnostics']['records']}
            need(set(canonical)=={row['sid'] for row in rows},'Canonical diagnostic ledger coverage differs')
            diagnostics={};selected={}
            for row in rows:
                if not row['cell'].startswith('length_'):continue
                source=source_rows[row['sid']]
                audit_prediction_identity(row,source)
                diagnostics[row['sid']]=load_diagnostic(row,canonical[row['sid']],run,condition,torch,np)
                selected[row['sid']]=row
            need(set(diagnostics)==set(source_rows),'Incomplete all-anchor diagnostics')
            run_records=[]
            for pair in pairs:
                short,long=diagnostics[pair['short_sid']],diagnostics[pair['long_sid']]
                result,delta=decomposition(short['messages'],long['messages'],pair['anchor_positions'],condition,np)
                saved_delta=long['merged']-short['merged'];error=saved_delta-delta
                record=dict(pair,condition=condition,seed=seed,run_id=run.name,**result,
                    saved_merge_reconstruction_l2=float(np.linalg.norm(error)),
                    saved_merge_reconstruction_max_abs=float(np.max(np.abs(error))),
                    short_diagnostic_sha256=selected[pair['short_sid']]['branch_diagnostics_sha256'],
                    long_diagnostic_sha256=selected[pair['long_sid']]['branch_diagnostics_sha256'])
                run_records.append(record)
            all_records.extend(run_records)
            summaries[condition][str(seed)]={'all':summarize(run_records,np),
                **{f'N16_to_N{n}':summarize([r for r in run_records if r['n_frames']==n],np) for n in (32,64)}}
            ratio_summaries[condition][str(seed)]={f'N{n}':dict(samples=108,
                branch_to_native_attention_norm_ratio=scalar_stats([diagnostics[sid]['ratio'] for sid,row in selected.items() if row['n_frames']==n],np),
                branch_residual_norm=scalar_stats([diagnostics[sid]['residual_norm'] for sid,row in selected.items() if row['n_frames']==n],np),
                native_attention_output_norm=scalar_stats([diagnostics[sid]['native_norm'] for sid,row in selected.items() if row['n_frames']==n],np))
                for n in (16,32,64)}
            provenance[condition][str(seed)]=dict(run=str(run),config_sha256=digest(run/'config.json'),
                summary_sha256=digest(run/'summary.json'),predictions_sha256=digest(run/'predictions.json'),
                selected_checkpoint=str(checkpoint),selected_checkpoint_sha256=info['selected_checkpoint_sha256'],
                selected_epoch=info['selected_epoch'],architecture=config['architecture'],training_source_sha256=expected)
    need(digest(args.analysis)==analysis_sha and digest(manifest_path)==manifest_sha and
         sources=={name:digest(REPO/name) for name in SOURCES},'Inputs/source changed while running')
    out=V5/'message_decomposition'/f"decomposition_{os.environ['SLURM_JOB_ID']}"
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    for name in SOURCES:shutil.copyfile(REPO/name,out/'source'/name.replace('/','_'))
    write(out/'pairs.json',all_records)
    write(out/'data_audit.json',dict(data_audit,pairs=pairs))
    notes=[__doc__,
        'All216 comparisons per model use108 fixed anchors twice, once per extension. Descriptive summaries do not treat those comparisons as independent evidence.',
        'Exact-image byte identity and changed Step counts are audit metadata only; no correctness-conditioned or exact-image subset is selected.',
        'Vector calculations use float64 conversions of saved tensors. Algebraic reconstruction and disagreement with separately saved float32 merges are reported separately.',
        'Cosines with a zero vector are undefined and excluded only from that cosine mean; total/defined denominators remain explicit.',
        'Native-attention-relative residual ratios are separate scalar diagnostics over324 unique scenes per model. Native norm is the selected ordinary attention output before branch addition, not the full residual stream.',
        'There is no counterfactual re-forward, classifier, semantic attribution of individual message coordinates or test-dependent checkpoint selection.']
    result=dict(protocol='v5_optional_semantic_frame_message_decomposition',passed=True,
        canonical_analysis=str(args.analysis),canonical_analysis_sha256=analysis_sha,
        manifest=str(manifest_path),manifest_sha256=manifest_sha,source_sha256=sources,
        provenance=provenance,data_audit=data_audit,summaries=summaries,
        native_attention_relative_residual_ratios=ratio_summaries,pairs_sha256=digest(out/'pairs.json'),
        data_audit_sha256=digest(out/'data_audit.json'),seconds=time.monotonic()-started,notes=notes)
    write(out/'analysis.json',result)
    lines=['# V5 mapped semantic-frame message decomposition','',
        'Optional descriptive diagnostic on all108 familiar-count anchor families per model. No correctness filtering.','',
        f"Across both extensions, {data_audit['byte_identical_mapped_frames']}/{data_audit['mapped_frame_comparisons']} mapped image pairs are byte-identical; {data_audit['changed_step_mapped_frames']} have renumbered Step labels.",'',
        '| Arm | Seed | Extension | Drift norm | Added-negative norm | Normalization norm | Total change norm | Drift / added cosine | Saved-merge reconstruction L2 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'undefined' if value is None else f'{value:.6g}'
    for condition in ('sum','mean'):
        for seed in ('4','5'):
            for n in (32,64):
                group=summaries[condition][seed][f'N16_to_N{n}'];norms=group['norms']
                values=[norms[k]['mean'] for k in ('mapped_semantic_drift','added_negative_contribution',
                                                  'normalization_change','total_merged_change')]
                values += [group['cosines']['drift_vs_added']['mean'],group['saved_merge_reconstruction_l2']['mean']]
                lines.append(f"| {condition} | {seed} | N16→N{n} | "+' | '.join(fmt(v) for v in values)+' |')
    lines+=['','Each entry is the mean of108 pair-level statistics. Norms of additive vectors do not add. Full per-pair values, cosine denominators, reconstruction errors and separately summarized residual/native ratios are retained in JSON.','',
        'Mapped-frame drift combines changed Step pixels with query/context and numerical changes. It does not isolate context-induced corruption or demonstrate a causal mechanism.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'INDEX.md').write_text('# Optional V5 decomposition artifacts\n\n- [Report](REPORT.md)\n- [Verified summaries and provenance](analysis.json)\n- [Every paired comparison](pairs.json)\n- [Semantic/image audit](data_audit.json)\n- [Source snapshots](source/)\n')
    print(json.dumps(dict(output=str(out),comparisons=len(all_records),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
