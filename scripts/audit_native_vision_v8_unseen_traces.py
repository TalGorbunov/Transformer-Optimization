"""Descriptive V8 unseen-count traces from already verified natural generations.

No model calls, new predictions, fitting, or interventions. First-token correctness
means the first generated token equals the first token of the canonical complete
numeral: never scan past whitespace or reasoning text. K9 and K10--16 are separate.
Immediate EOS after a correct first token is expected for K9, but prematurely ends
K10--16. Metrics overlap; malformed-or-truncated is their union. This does not test
whether latent representations contain a count or establish reasoning composition.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'outputs/native_aggregation_vlm/v8'
REPORT=BASE/'report_441974/analysis.json'
OUT=BASE/'unseen_trace_diagnostic'
OWN=('scripts/audit_native_vision_v8_unseen_traces.py','slurm/native_vision_v8_unseen_traces.sbatch')
CONDITIONS=('ce','consistency')
SEEDS=(10,11)
EOS=(151645,151643)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def sources():return {name:sha(REPO/name) for name in OWN}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V8 unseen-count trace diagnostic\n\n[Summary](summary.json) · [Report](REPORT.md) · [Input bindings](plan.json) · [Trace records](traces.json) · [Sources](source_hashes.json)\n')
    return frozen


def classify(ids,text,gold,gold_ids,eos=EOS):
    need(isinstance(ids,list) and 1<=len(ids)<=4 and all(type(t) is int for t in ids),'Invalid generated token list')
    need(gold_ids and all(type(t) is int and t not in eos for t in gold_ids),'Invalid numeral tokenization')
    completed=ids[-1] in eos
    need(not any(t in eos for t in ids[:-1]) and (completed or len(ids)==4),'Invalid natural EOS/truncation history')
    stripped=text.strip();parsed=int(stripped) if re.fullmatch(r'[0-9]+',stripped) else None
    first=ids[0]==gold_ids[0]
    early=first and len(ids)==2 and ids[1] in eos
    return dict(first_numeral_token_correct=first,whole_answer_correct_with_eos=completed and parsed==gold,
        correct_first_token_then_immediate_eos=early,wrong_first_token=not first,
        malformed=parsed is None,truncated=not completed,malformed_or_truncated=parsed is None or not completed,
        completed=completed,parseable=parsed is not None,parsed_numeral=parsed,
        parsed_count_correct=parsed==gold,canonical_numeral_tokens_complete=(ids[:-1] if completed else ids)==gold_ids)


def grouped(rows):
    fields=('first_numeral_token_correct','whole_answer_correct_with_eos','correct_first_token_then_immediate_eos',
            'wrong_first_token','malformed','truncated','malformed_or_truncated','completed','parseable')
    result=[]
    for condition in CONDITIONS:
        for seed in SEEDS:
            for n in (32,64):
                for family,ks in (('K9',{9}),('K10_16',set(range(10,17)))):
                    cell=[r for r in rows if r['condition']==condition and r['seed']==seed
                          and r['n_frames']==n and r['gold'] in ks]
                    need(len(cell)==(8 if family=='K9' else 56),'Unseen trace group denominator differs')
                    counts={f:sum(r[f] for r in cell) for f in fields}
                    result.append(dict(condition=condition,seed=seed,n_frames=n,count_family=family,n=len(cell),
                        counts=counts,rates={k:v/len(cell) for k,v in counts.items()},
                        immediate_eos_given_correct_first=dict(n=counts['first_numeral_token_correct'],
                            count=counts['correct_first_token_then_immediate_eos'],
                            rate=counts['correct_first_token_then_immediate_eos']/counts['first_numeral_token_correct']
                                if counts['first_numeral_token_correct'] else None),
                        generated_numeral_histogram=dict(sorted(Counter(str(r['parsed_numeral']) for r in cell if r['parseable']).items())),
                        completed_numeral_histogram=dict(sorted(Counter(str(r['parsed_numeral']) for r in cell if r['parseable'] and r['completed']).items())),
                        generated_text_histogram=dict(sorted(Counter(r['text'] for r in cell).items()))))
    return result


def self_test():
    a=classify([1,99],'1',10,[1,0],(99,))
    need(a['first_numeral_token_correct'] and a['correct_first_token_then_immediate_eos']
         and not a['whole_answer_correct_with_eos'],'Correct first digit is not a complete multi-digit answer')
    b=classify([1,0,99],'10',10,[1,0],(99,))
    need(b['whole_answer_correct_with_eos'] and not b['correct_first_token_then_immediate_eos'],'Complete two-digit answer failed')
    c=classify([9,99],'9',9,[9],(99,))
    need(c['whole_answer_correct_with_eos'] and c['correct_first_token_then_immediate_eos'],'K9 expected EOS differs')
    d=classify([1,0,3,2],'1032',10,[1,0],(99,))
    need(d['truncated'] and d['first_numeral_token_correct'] and not d['whole_answer_correct_with_eos'],'Truncation was rescued')
    e=classify([42,99],'Answer: 10',10,[1,0],(99,))
    need(e['malformed'] and e['wrong_first_token'],'Parser scanned a malformed answer for a last number')
    for ids in ([1,99,0],[1],[]):
        try:classify(ids,'1',10,[1,0],(99,))
        except ValueError:pass
        else:raise AssertionError('Invalid stop sequence accepted')
    return dict(passed=True,tests=['first_digit_vs_whole_numeral','premature_vs_expected_EOS',
        'truncation_always_incorrect','strict_complete_text_parser','reject_invalid_stop_histories'])


def analyze(args,out,frozen):
    import torch,transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(4);start=time.perf_counter();bindings={}
    def bind(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        need(expected is None or actual==expected,'Artifact changed: '+str(path))
        need(str(path) not in bindings or bindings[str(path)]==actual,'Conflicting binding')
        bindings[str(path)]=actual
        return path
    report_path=args.report.resolve();need(report_path==REPORT,'Only the canonical completed V8 report is allowed')
    summary_path=report_path.parent/'summary.json';completion=read(bind(summary_path))
    bind(report_path,completion['analysis_sha256']);report=read(report_path)
    need(completion['passed'] is True and Path(completion['analysis_file']).resolve()==report_path
         and report['passed'] is True and report['audit_passed'] is True
         and completion['source_sha256']==report['source_sha256'],'Canonical report completion failed')
    for name,digest in report['source_sha256'].items():
        bind(REPO/name,digest);bind(report_path.parent/'code'/name.replace('/','_'),digest)
    need(set(report['runs'])=={f'{c}_s{s}' for c in CONDITIONS for s in SEEDS},'Need all four audited V8 runs')
    inputs=[];tokenizer=None;model_metadata=None;tokenizations=None
    for key,r in sorted(report['runs'].items()):
        directory=Path(r['run_directory']).resolve()
        need(directory.parent==BASE,'Noncanonical run directory')
        config=read(bind(directory/'config.json',r['config_sha256']))
        summary=read(bind(directory/'summary.json',r['summary_sha256']))
        need(key==f"{config['condition']}_s{config['seed']}" and config['arm']=='parallel'
             and summary['passed'] and summary['completed'] and summary['native_test_count']==452,'Run identity/completion differs')
        selected=summary['selected']
        need(selected['checkpoint']==r['selected_checkpoint'] and selected['checkpoint_sha256']==r['selected_checkpoint_sha256']
             and selected['parameter_sha256']==r['selected_parameter_sha256'] and selected['step']==r['selected_step'],
             'Reported selected checkpoint differs')
        bind(selected['checkpoint'],selected['checkpoint_sha256'])
        plan=read(bind(config['plan_file'],config['plan_sha256']))
        need(plan['source_sha256']==config['source_sha256'] and plan['policy']==config['policy'],'Run/CPU source policy differs')
        for name,digest in config['source_sha256'].items():bind(REPO/name,digest);bind(directory/'code'/name.replace('/','_'),digest)
        mpath=Path(plan['fresh_manifests']['count']).resolve()
        need(mpath==Path('/mnt/data/gabriele/gnn_transformer/v8_fresh/count_manifest.json'),'Unexpected unseen-count manifest')
        manifest=read(bind(mpath,plan['artifact_bindings'][str(mpath)]))
        for label in ('audit','inventory','stage_plan'):
            if label+'_file' in manifest:bind(manifest[label+'_file'],manifest[label+'_sha256'])
        rows={r['sid']:r for cell in manifest['splits'].values() for r in cell['samples']}
        need(len(rows)==128 and Counter((r['n_frames'],r['gold']) for r in rows.values())==
             Counter({(n,k):8 for n in (32,64) for k in range(9,17)}),'Count manifest balance differs')
        if tokenizer is None:
            model_metadata=config['model'];model=Path(model_metadata['path'])
            for name,digest in model_metadata['metadata_sha256'].items():bind(model/name,digest)
            need(config['runtime']['transformers_version']==str(transformers.__version__),'Tokenizer software version differs')
            tokenizer=AutoTokenizer.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
            tokenizations={str(k):tokenizer(str(k),add_special_tokens=False)['input_ids'] for k in range(9,17)}
            need(all(tokenizer.decode(ids,skip_special_tokens=False)==k for k,ids in tokenizations.items()),'Numeral tokenizer roundtrip differs')
            need(len(tokenizations['9'])==1 and all(len(tokenizations[str(k)])==2 for k in range(10,17)),
                 'Registered one-digit/two-digit numeral distinction no longer holds')
        need(config['model']==model_metadata and config['policy']['native_eos']==list(EOS)
             and config['policy']['max_new_tokens']==4,'Model/EOS/budget differs across runs')
        test_path=bind(directory/'test.json',r['test_sha256'])
        need(str(test_path)==summary['test_file'] and summary['test_sha256']==r['test_sha256'],'Test artifact binding differs')
        test=read(test_path);bind(test['raw_file'],test['raw_sha256'])
        need(test['n']==len(test['rows'])==452 and test['raw_dtype']=='torch.float32','Complete V8 test archive required')
        selected_rows=[x for x in test['rows'] if x['cell'] in ('count_test_N32','count_test_N64')]
        need(len(selected_rows)==128 and {x['sid'] for x in selected_rows}==set(rows),'Unseen prediction inventory differs')
        inputs.append(dict(key=key,report_run=r,config=config,test=test,rows=selected_rows,manifest_rows=rows))
    plan=dict(schema_version=1,source_sha256=frozen,report_file=str(report_path),report_sha256=sha(report_path),
        report_completion_file=str(summary_path),report_completion_sha256=sha(summary_path),artifact_sha256=bindings,
        numeral_token_ids=tokenizations,native_eos=list(EOS),transformers_version=str(transformers.__version__),
        selected_models={x['key']:{k:x['report_run'][k] for k in ('run_directory','selected_step','selected_checkpoint','selected_checkpoint_sha256','selected_parameter_sha256')} for x in inputs},
        scope=__doc__,expected_examples=512,no_model_calls=True,no_fit=True)
    save(out/'plan.json',plan)
    traces=[]
    for item in inputs:
        test=item['test'];blob=torch.load(test['raw_file'],map_location='cpu',weights_only=True,mmap=True)
        need(blob['schema_version']==1 and len(blob['raw_logits'])==452,'Raw archive structure differs')
        seen=set()
        for row in item['rows']:
            sample=item['manifest_rows'][row['sid']];idx=row['raw_index']
            need(type(idx) is int and idx not in seen and 0<=idx<452 and test['rows'][idx]==row,'Raw trace index mismatch')
            seen.add(idx)
            need(all(row[k]==sample[k] for k in ('sid','gold','n_frames','content_sha256'))
                 and row['anchor_id']==sample['anchor_id'],'Manifest/prediction identity differs')
            need(row['metadata']['image_sha256']==[x['sha256'] for x in sample['image_files']]
                 and row['metadata']['question']==sample['question'],'Recorded visual/question evidence differs')
            ids=row['generated_ids'];x=blob['raw_logits'][idx]
            need(x.ndim==2 and x.shape[0]==len(ids) and x.dtype==torch.float32 and bool(torch.isfinite(x).all())
                 and torch.equal(x,x.half().float()) and x.argmax(-1).tolist()==ids,'Raw native greedy output differs')
            need(row['raw_text']==tokenizer.decode(ids,skip_special_tokens=False)
                 and row['text']==tokenizer.decode(ids,skip_special_tokens=True),'Natural decoded text differs')
            info=classify(ids,row['text'],row['gold'],tokenizations[str(row['gold'])])
            need(row['exact']==info['whole_answer_correct_with_eos'] and row['completed']==info['completed']
                 and row['truncated']==info['truncated'] and row['prediction']==info['parsed_numeral']
                 and row['parseable']==info['parseable'] and row['parsed_count_correct']==info['parsed_count_correct'],
                 'Original parser/completion accounting differs')
            traces.append(dict(condition=item['config']['condition'],seed=item['config']['seed'],
                sid=row['sid'],anchor_id=row['anchor_id'],n_frames=row['n_frames'],gold=row['gold'],
                count_family='K9' if row['gold']==9 else 'K10_16',generated_ids=ids,raw_text=row['raw_text'],text=row['text'],
                gold_numeral_ids=tokenizations[str(row['gold'])],raw_file=test['raw_file'],raw_sha256=test['raw_sha256'],raw_index=idx,
                selected_checkpoint_sha256=item['report_run']['selected_checkpoint_sha256'],**info))
        del blob
    need(len(traces)==512,'Incomplete all-model unseen trace coverage')
    groups=grouped(traces)
    save(out/'traces.json',traces);save(out/'groups.json',groups)
    need(sources()==frozen,'Diagnostic source changed during execution')
    for path,digest in bindings.items():need(sha(path)==digest,'Input changed during analysis: '+path)
    result=dict(schema_version=1,passed=True,diagnostic_only=True,no_fit=True,model_calls=0,examples=512,groups=groups,
        plan_file=str(out/'plan.json'),plan_sha256=sha(out/'plan.json'),source_sha256=frozen,
        traces_file=str(out/'traces.json'),traces_sha256=sha(out/'traces.json'),groups_sha256=sha(out/'groups.json'),
        seconds=time.perf_counter()-start,slurm_job_id=os.environ['SLURM_JOB_ID'],
        limitations=['First generated token is compared directly; no search for a later numeral.',
            'K9 immediate EOS can be correct; K10--16 immediate EOS after the first token is incomplete.',
            'Metrics overlap; malformed-or-truncated is a union and all rates use the full group denominator.',
            'Numeral histograms include parseable truncated outputs; completed histograms are also retained.',
            'Already verified natural outputs only; this does not prove presence or absence of latent count information.',
            'No new model call, prediction, fitted decoder, oracle prefix, or reasoning-composition claim.'])
    lines=['# V8 unseen-count natural traces','','All 512 original unseen-count outputs retained; no new model calls.','',
        '| Condition | Seed | N | K | n | First token correct | Whole + EOS | Correct first, immediate EOS | Wrong first | Malformed | Truncated |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        c=g['counts'];lines.append(f"| {g['condition']} | {g['seed']} | {g['n_frames']} | {g['count_family']} | {g['n']} | {c['first_numeral_token_correct']} | {c['whole_answer_correct_with_eos']} | {c['correct_first_token_then_immediate_eos']} | {c['wrong_first_token']} | {c['malformed']} | {c['truncated']} |")
    lines+=['','Each count uses the stated complete group denominator. Immediate EOS overlaps correct answers at K9.','',
        '[All traces](traces.json) · [Histograms and conditional denominators](groups.json) · [Provenance](plan.json)','']
    lines += ['- '+text for text in result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--report',type=Path,default=REPORT);args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'All diagnostic work requires CPU-only Slurm')
    out=OUT/f'{"selftest" if args.self_test else "run"}_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);frozen=snapshot(out)
    tests=self_test();save(out/'self_tests.json',tests)
    result=dict(tests,source_sha256=frozen) if args.self_test else analyze(args,out,frozen)
    save(out/'summary.json',result)
    print(json.dumps(dict(passed=True,directory=str(out))),flush=True)


if __name__=='__main__':main()
