"""Fixed CPU synthetic response comparison for all four selected V8 cores.

Reuses the immutable V7 training-state selection and surface algebra. These
432 synthetic first-token points are not MMReD accuracy or new model fitting.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
import time
import json
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts import probe_native_vision_v7_response_surface as old
OWN=('scripts/probe_native_vision_v8_response_surface.py','slurm/native_vision_v8_response_surface.sbatch',*old.OWN)
OUT=old.BASE/'v8/response_surface'


def execute(args,out,frozen,tests):
    import torch
    from transformers import AutoTokenizer
    from gnnformer.parallel_local_aggregation import ParallelLocalAggregation
    report_path=args.report.resolve();report=old.read(report_path)
    completion_path=report_path.parent/'summary.json';completion=old.read(completion_path)
    old.need(completion['passed'] is True and Path(completion['analysis_file']).resolve()==report_path
             and completion['analysis_sha256']==old.sha(report_path)
             and completion['source_sha256']==report['source_sha256'],'Independent report completion binding differs')
    for name,digest in report['source_sha256'].items():
        old.bind(report_path.parent/'code'/name.replace('/','_'),digest)
    old.need(report['audit_passed'] is True and report['passed'] is True,'Completed independent V8 report required')
    expected={f'{c}_s{s}' for c in ('ce','consistency') for s in (10,11)}
    old.need(set(report['runs'])==expected,'All four registered V8 models required')
    for name,digest in report['source_sha256'].items(): old.bind(REPO/name,digest)
    _,old_audits,cache,plan,questions,samples,prototypes,states=old.load_inputs(old.REPORT)
    cache_binding=old_audits[0]['cache']
    norm,head,head_provenance=old.load_head(args.head_reference,cache['model'])
    tokenizer=AutoTokenizer.from_pretrained(cache['model']['path'],use_fast=False,local_files_only=True)
    count_ids=[plan['count_token_ids'][str(k)] for k in range(9)]
    old.need(len(set(count_ids))==9 and all(tokenizer(str(k),add_special_tokens=False)['input_ids']==[count_ids[k]] for k in range(9)), 'Native count token identity changed')
    groups=[];raw_groups=[];bindings=[]
    for key in sorted(expected):
        audit=report['runs'][key];condition=audit['condition'];seed=audit['seed'];directory=Path(audit['run_directory'])
        old.bind(directory/'config.json',audit['config_sha256']);old.bind(directory/'summary.json',audit['summary_sha256'])
        config=old.read(directory/'config.json');summary=old.read(directory/'summary.json');selected=summary['selected']
        old.need(key==f'{condition}_s{seed}' and config['arm']=='parallel' and config['condition']==condition
                 and config['seed']==seed and summary['passed'] and summary['completed'],'Selected run identity differs')
        old.need(config['cache_binding']=={k:cache_binding[k] for k in ('file','sha256')},'V8 uses a different frozen training cache')
        old.need(selected['step']==audit['selected_step'] and selected['checkpoint']==audit['selected_checkpoint']
                 and selected['checkpoint_sha256']==audit['selected_checkpoint_sha256']
                 and selected['parameter_sha256']==audit['selected_parameter_sha256'],'Selected model differs from independent report')
        for name,digest in config['source_sha256'].items(): old.bind(REPO/name,digest)
        old.bind(selected['checkpoint'],selected['checkpoint_sha256'])
        blob=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)
        old.need(blob['config']==config and blob['step']==selected['step'] and old.state_hash(blob['branch'])==selected['parameter_sha256'],'Checkpoint contents changed')
        branch=ParallelLocalAggregation().eval().requires_grad_(False);branch.load_state_dict(blob['branch'],strict=True)
        old.need(sum(p.numel() for p in branch.parameters())==1041600 and all(p.dtype==torch.float32 for p in branch.parameters()),'Native branch differs')
        bindings.append(dict(condition=condition,seed=seed,run_directory=str(directory),config_sha256=audit['config_sha256'],
            summary_sha256=audit['summary_sha256'],checkpoint=selected['checkpoint'],checkpoint_sha256=selected['checkpoint_sha256'],
            parameter_sha256=selected['parameter_sha256'],step=selected['step']))
        for question in questions:
            records=samples[question];gids={cache['scenes'][r['sid']]['global_feature_ids'][0] for r in records}
            old.need(len(gids)==1,'Global state unexpectedly depends on N/K');gid=next(iter(gids))
            old.need(plan['features'][gid]['prefix_ids']==[],'Global state contains an answer prefix')
            g=states[gid].unsqueeze(0)
            local=[torch.stack([states[ids[0]] for ids in cache['scenes'][r['sid']]['local_feature_ids']]).unsqueeze(1) for r in records]
            chosen={category:states[prototypes[question][category]['feature_id']] for category in old.CATEGORIES}
            result,tensors=old.surface(branch,g,chosen,local,records,norm,head,count_ids)
            for row in result['rows']:row['first_token_text']=tokenizer.decode([row['first_token_id']],skip_special_tokens=False)
            result.update(condition=condition,seed=seed,question=question,global_feature_id=gid,prototypes=prototypes[question])
            groups.append(result);raw_groups.append(dict(condition=condition,seed=seed,question=question,rows=result['rows'],**tensors))
            print(json.dumps(dict(condition=condition,seed=seed,question=question,synthetic_points=len(result['rows']))),flush=True)
        del branch,blob
    old.need(len(groups)==8 and sum(len(g['rows']) for g in groups)==432,'Fixed diagnostic coverage differs')
    rawdir=old.DATA/'native_aggregation_vlm_v8_response_surface'/f'run_{os.environ["SLURM_JOB_ID"]}'
    rawdir.mkdir(parents=True,exist_ok=False);rawfile=rawdir/'tensors.pt'
    torch.save(dict(schema_version=1,groups=raw_groups,count_token_ids=count_ids),rawfile)
    passed=all(g['actual_zero_padding']['passed'] and g['actual_multiset_vs_algebra']['passed'] for g in groups)
    analysis=dict(schema_version=1,passed=passed,diagnostic_only=True,no_fit=True,vlm_forward_calls=0,
        native_head_synthetic_rows=432,native_head_reference_replay_rows=4,source_sha256=frozen,self_tests=tests,
        report_file=str(report_path),report_sha256=old.sha(report_path),
        report_completion_file=str(completion_path),report_completion_sha256=old.sha(completion_path),
        prototype_ancestor_file=str(old.REPORT),prototype_ancestor_sha256=old.sha(old.REPORT),
        training_cache=cache_binding,training_plan_file=cache['plan_file'],training_plan_sha256=cache['plan_sha256'],
        selected_checkpoints=bindings,head=head_provenance,count_token_ids=count_ids,questions=questions,groups=groups,
        raw_file=str(rawfile),raw_sha256=old.sha(rawfile),
        selection_rule='Exactly the V7 first two training questions and lowest pair_id prototypes, all four V8 selected cores',
        limitations=['Synthetic repeated states preserve prototype Step labels and violate the distinct-Step generator law.',
            'First-token predictions are not native integer-plus-EOS MMReD accuracy.',
            'All source states and comparison sums are training-only; no new VLM forwards.',
            'Known-category negative removal is an algebraic diagnostic, not a deployed inference rule.',
            'Exact FP16 weight identities do not imply identical CPU/GPU arithmetic; replay differences remain descriptive.',
            'No model selection or original V8 acceptance decision depends on this diagnostic.'])
    old.save(out/'analysis.json',analysis)
    lines=['# V8 fixed synthetic response comparison','',
        '432 fixed synthetic first-token points; zero fitting and zero VLM calls. This is a mechanism diagnostic, not held-out accuracy.','',
        '| Condition | Seed | Question | N16 first-token correct | N64 first-token correct |',
        '|---|---:|---|---:|---:|']
    for group in groups:
        totals={n:sum(r['first_token_correct'] for r in group['rows'] if r['n_frames']==n) for n in (16,64)}
        lines.append(f"| {group['condition']} | {group['seed']} | {group['question']} | {totals[16]}/27 | {totals[64]}/27 |")
    lines+=['','The same training-only local prototypes, native head and response computation are used for every model. Successful synthetic N64 decoding does not establish actual N64 generalization or remove the later-Step distribution shift.','',
        '[All raw predictions, geometry, floating-point checks and input identities](analysis.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(passed=passed,diagnostic_only=True,no_fit=True,vlm_forward_calls=0,synthetic_points=432,
        analysis_file=str(out/'analysis.json'),analysis_sha256=old.sha(out/'analysis.json'),raw_file=str(rawfile),raw_sha256=old.sha(rawfile))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true')
    p.add_argument('--report',type=Path);p.add_argument('--source-check',type=Path)
    p.add_argument('--head-reference',type=Path,default=old.REFERENCE);args=p.parse_args()
    old.need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
             and not os.environ.get('SLURM_JOB_GPUS'),'CPU Slurm required')
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.perf_counter()
    mode='selftest' if args.self_test else 'run';out=OUT/f'{mode}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir();frozen={name:old.sha(REPO/name) for name in OWN}
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    old.save(out/'source_hashes.json',frozen)
    with torch.inference_mode():
        tests=old.self_test()
        if args.self_test:result=dict(passed=True,tests=tests,diagnostic_only=True,vlm_forward_calls=0)
        else:
            old.need(args.report is not None and args.source_check is not None,'Require report and matching unit source freeze')
            unit=old.read(args.source_check/'summary.json')
            old.need(unit['passed'] and unit['source_sha256']==frozen,'Unit/source freeze differs')
            result=execute(args,out,frozen,tests)
    for name,digest in frozen.items():old.bind(REPO/name,digest)
    result.update(source_sha256=frozen,elapsed_seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'])
    old.save(out/'summary.json',result)
    (out/'INDEX.md').write_text('# V8 fixed response diagnostic\n\n[Summary](summary.json) · [Frozen sources](source_hashes.json)\n')
    print(json.dumps(dict(passed=result['passed'],summary_file=str(out/'summary.json'))),flush=True)
    if not result['passed']:raise SystemExit('Diagnostic software check failed; all observations retained')
if __name__=='__main__':main()
