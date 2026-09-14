"""Scoped enum compatibility for immutable learned-selection entry points.

Only the installed QuantizationMethod class is allowlisted. All original
loads retain weights_only=True. Numerical checks run exclusively in Slurm.
"""
from __future__ import annotations
import argparse
from enum import Enum
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
ROOT=REPO/'outputs/native_aggregation_vlm/identity_join_learned'
CKPT=Path('/mnt/ckpts/gabriele/gnn_transformer/identity_join_learned/training')
PROTOCOL='identity_join_exact_quantization_enum_compatibility'
OWN=('scripts/native_identity_join_serialization_compat.py',
     'slurm/native_identity_join_serialization_check.sbatch',
     'slurm/native_identity_join_serialization_train_profile.sbatch',
     'slurm/native_identity_join_serialization_train.sbatch',
     'slurm/native_identity_join_serialization_report.sbatch')
TARGETS={'train':'scripts.train_native_identity_join_learned',
         'report':'scripts.report_native_identity_join_learned'}
FAILED={'clip':'profile_clip_s22_443037','sigmoid':'profile_sigmoid_s22_443038',
        'softmax':'profile_softmax_s22_443036'}


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
def sources():return {p:sha(REPO/p) for p in OWN}
def bind(path,digest):need(sha(path)==digest,'Bound artifact changed: '+str(path))


def originals():
    training=importlib.import_module(TARGETS['train']);report=importlib.import_module(TARGETS['report'])
    return {'train':training.sources(),'report':report.source_hashes()}


def installed():
    import torch
    import transformers
    from transformers.utils.quantization_config import QuantizationMethod
    need(QuantizationMethod.__module__=='transformers.utils.quantization_config'
         and QuantizationMethod.__qualname__=='QuantizationMethod'
         and issubclass(QuantizationMethod,str) and issubclass(QuantizationMethod,Enum),
         'Only the exact installed string enum may be allowed')
    files=[Path(inspect.getfile(QuantizationMethod)).resolve(),Path(torch.serialization.__file__).resolve()]
    identity=dict(torch_version=torch.__version__,transformers_version=transformers.__version__,
        allowed_global='transformers.utils.quantization_config.QuantizationMethod',
        source_sha256={str(p):sha(p) for p in files},weights_only=True,
        mechanism='torch.serialization.safe_globals([QuantizationMethod])')
    return torch,QuantizationMethod,identity


def scope_call(torch,enum,callback):
    before=set(torch.serialization.get_safe_globals())
    need(enum not in before,'Exact enum must not already be globally allowlisted')
    argv=list(sys.argv)
    try:
        with torch.serialization.safe_globals([enum]):
            need(set(torch.serialization.get_safe_globals())==before|{enum},'Allowlist gained another object')
            return callback()
    finally:
        sys.argv=argv
        need(set(torch.serialization.get_safe_globals())==before,'Scoped allowlist was not restored')


def snapshot(out):
    frozen=sources();(out/'source').mkdir()
    for name,digest in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());bind(target,digest)
    save(out/'source_hashes.json',frozen);return frozen


def verify_proof(path,identity):
    path=Path(path).resolve();proof=read(path)
    need(proof['protocol']==PROTOCOL and proof['passed'] is True and proof['completed'] is True
         and proof['phase']=='check' and proof['no_model_loaded'] and proof['weights_only'] is True
         and proof['prior_safe_globals_restored'] and proof['argv_restored']
         and proof['exception_cleanup_passed'] and proof['source_sha256']==sources()
         and proof['installed_identity']==identity and proof['original_source_sha256']==originals(),
         'Completed exact-enum CPU compatibility proof differs')
    need({x['condition'] for x in proof['checkpoints']}==set(FAILED) and len(proof['checkpoints'])==3,
         'All three actual failed checkpoints required')
    for p,h in proof['artifact_bindings'].items():bind(p,h)
    for n,h in proof['source_sha256'].items():bind(path.parent/'source'/n.replace('/','_'),h)
    return proof


def check(out,frozen,torch,enum,identity):
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and not os.environ.get('SLURM_JOB_GPUS'),
         'Compatibility check requires CPU Slurm')
    torch.set_num_threads(4);old_argv=list(sys.argv);before=set(torch.serialization.get_safe_globals())
    training=importlib.import_module(TARGETS['train']);ledgers=originals();bindings={};rows=[]
    def remember(path):
        path=Path(path).resolve();bindings[str(path)]=sha(path);return path
    def load_all():
        for condition,runid in FAILED.items():
            directory=ROOT/'training'/runid;config=read(remember(directory/'config.json'))
            failure=read(remember(directory/'failure.json'));logs=read(remember(directory/'training.json'))
            need('transformers.utils.quantization_config.QuantizationMethod' in failure['message']
                 and 'Weights only load failed' in failure['message'] and len(logs)==32
                 and [r['step'] for r in logs]==list(range(1,33)), 'Actual failed profile evidence differs')
            need(config['run_id']==runid and config['condition']==condition and config['profile'] is True
                 and config['seed']==22 and config['policy']==training.POLICY
                 and config['source_sha256']==ledgers['train'] and Path(config['checkpoint_directory'])==CKPT/runid,
                 'Failed profile configuration/source differs')
            for name,digest in ledgers['train'].items():bind(directory/'code'/name.replace('/','_'),digest)
            file=remember(CKPT/runid/'profile.pt');packet=torch.load(file,map_location='cpu',weights_only=True)
            need(set(packet)=={'branch','step','config'} and packet['step']==32
                 and json.loads(json.dumps(packet['config']))==config
                 and type(packet['config']['hardware']['quantization']['quant_method']) is enum,
                 'Exact 32-step checkpoint/config enum differs')
            weights=packet['branch'];need(set(weights)==set(config['initialized'])
                 and sum(v.numel() for v in weights.values())==1041697,'Final core parameter inventory differs')
            for name,value in weights.items():
                need(value.device.type=='cpu' and value.dtype==torch.float32
                     and list(value.shape)==config['initialized'][name]['shape'] and bool(torch.isfinite(value).all()),
                     'Invalid actual core tensor: '+name)
            rows.append(dict(condition=condition,run_id=runid,step=32,checkpoint=str(file),checkpoint_sha256=sha(file),
                parameter_count=1041697,tensors={k:training.v7.tensor_info(v) for k,v in weights.items()},
                source_sha256=config['source_sha256'],weights_only=True))
    scope_call(torch,enum,load_all)
    need(set(torch.serialization.get_safe_globals())==before and sys.argv==old_argv,'Success cleanup differs')
    class ExpectedFailure(Exception):pass
    def fail():
        sys.argv=['deliberately temporary compatibility test'];raise ExpectedFailure()
    try:scope_call(torch,enum,fail)
    except ExpectedFailure:pass
    else:need(False,'Exception cleanup fixture did not raise')
    need(set(torch.serialization.get_safe_globals())==before and sys.argv==old_argv,'Exception cleanup differs')
    need(sources()==frozen and originals()==ledgers,'Source changed during compatibility check')
    return dict(protocol=PROTOCOL,passed=True,completed=True,phase='check',no_model_loaded=True,
        weights_only=True,source_sha256=frozen,original_source_sha256=ledgers,installed_identity=identity,
        checkpoints=rows,artifact_bindings=bindings,prior_safe_globals_restored=True,argv_restored=True,
        exception_cleanup_passed=True)


def verify_sidecar(directory,proof_path,identity,*,target,phase):
    directory=Path(directory).resolve();path=directory/'serialization_compatibility.json';value=read(path)
    need(value['protocol']==PROTOCOL and value['target']==target and value['phase']==phase
         and value['target_directory']==str(directory) and value['passed'] is True
         and value['completed'] is True and value['target_returned_normally'] is True
         and value['prior_safe_globals_restored'] and value['argv_restored']
         and value['weights_only'] is True and value['source_sha256']==sources()
         and value['original_source_sha256']==originals() and value['installed_identity']==identity
         and value['proof_file']==str(Path(proof_path).resolve()) and value['proof_sha256']==sha(proof_path),
         'Consumed target lacks exact successful scoped compatibility evidence')
    need('summary.json' in value['output_sha256'],'Compatibility target summary is absent')
    for name,digest in value['output_sha256'].items():bind(directory/name,digest)
    summary=read(directory/'summary.json');need(summary['passed'] is True and summary['completed'] is True,
        'Consumed target summary failed')
    if target=='report':need(summary['phase']==phase,'Consumed report phase differs')
    else:need(summary['profile'] is (phase=='profile'),'Consumed training phase differs')
    for item in value['prerequisites']:bind(item['file'],item['sha256'])
    return dict(file=str(path),sha256=sha(path),target=target,phase=phase)


def dispatch(args,remaining,torch,enum,identity):
    proof_path=args.proof.resolve();verify_proof(proof_path,identity)
    parse=argparse.ArgumentParser(add_help=False)
    for flag in ('profile','run','release','report'):parse.add_argument('--'+flag,action='store_true')
    parse.add_argument('--condition');parse.add_argument('--seed',type=int);parse.add_argument('--main-release',type=Path)
    parse.add_argument('--profile-directories',nargs='*',default=[]);parse.add_argument('--run-directories',nargs='*',default=[])
    options,_=parse.parse_known_args(remaining);phases=[p for p in ('profile','run','release','report') if getattr(options,p)]
    need(len(phases)==1 and phases[0] in (('profile','run') if args.target=='train' else ('release','report')),
         'Only original profile/run or release/report entry points are supported')
    phase=phases[0];job=os.environ['SLURM_JOB_ID']
    if args.target=='train':
        need(options.condition in ('clip','sigmoid','softmax') and options.seed in (22,23),'Original arm/seed required')
        out=ROOT/'training'/f'{phase}_{options.condition}_s{options.seed}_{job}'
    else:out=ROOT/'reporting'/f'{phase}_{job}'
    need(not out.exists(),'Never overwrite a prior target attempt')
    frozen=sources();original=originals();argv=list(sys.argv);safe=set(torch.serialization.get_safe_globals())
    prerequisites=[];returned=False;error=None;target_module=None;started=time.perf_counter()
    try:
        if args.target=='report' and phase=='release':
            need(len(options.profile_directories)==3,'Three scoped training profile proofs required')
            for directory in options.profile_directories:
                prerequisites.append(verify_sidecar(directory,proof_path,identity,target='train',phase='profile'))
        elif args.target=='train' and phase=='run':
            need(options.main_release is not None,'Exact scoped release required')
            side=verify_sidecar(options.main_release.parent,proof_path,identity,target='report',phase='release')
            record=read(side['file']);need(record['output_sha256'].get('release.json')==sha(options.main_release)
                and options.main_release.name=='release.json','Actual main release is not sidecar-bound')
            prerequisites.append(side)
        elif args.target=='report' and phase=='report':
            need(len(options.run_directories)==6,'Six scoped main proofs required')
            for directory in options.run_directories:
                prerequisites.append(verify_sidecar(directory,proof_path,identity,target='train',phase='run'))
        target_module=importlib.import_module(TARGETS[args.target])
        def invoke():
            sys.argv=[str(Path(target_module.__file__).resolve()),*remaining]
            target_module.main()
        scope_call(torch,enum,invoke);returned=True
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        argv_restored=sys.argv==argv;globals_restored=set(torch.serialization.get_safe_globals())==safe
        summary=read(out/'summary.json') if (out/'summary.json').exists() else None
        outputs={name:sha(out/name) for name in ('summary.json','failure.json','release.json','analysis.json',
            'final_endpoint.json','selection.json','config.json') if (out/name).is_file()}
        unchanged=sources()==frozen and originals()==original
        passed=bool(returned and summary and summary.get('passed') is True and summary.get('completed') is True
                    and argv_restored and globals_restored and unchanged)
        out.mkdir(parents=True,exist_ok=True)
        save(out/'serialization_compatibility.json',dict(protocol=PROTOCOL,target=args.target,phase=phase,
            target_directory=str(out),slurm_job_id=job,passed=passed,completed=returned,
            target_returned_normally=returned,error=error,weights_only=True,source_sha256=frozen,
            original_source_sha256=original,target_source_file=str(REPO/(TARGETS[args.target].replace('.','/')+'.py')),
            proof_file=str(proof_path),proof_sha256=sha(proof_path),installed_identity=identity,
            prior_safe_globals_restored=globals_restored,argv_restored=argv_restored,
            sources_unchanged=unchanged,prerequisites=prerequisites,output_sha256=outputs,seconds=time.perf_counter()-started))
        if returned:need(passed,'Compatibility target did not complete with restored state and bound sources')


def main():
    need(os.environ.get('SLURM_JOB_ID'),'Slurm required; no numerical execution on login')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    parser.add_argument('--target',choices=TARGETS);parser.add_argument('--proof',type=Path)
    args,remaining=parser.parse_known_args();need(args.check != (args.target is not None),'Choose compatibility check or one target')
    if remaining and remaining[0]=='--':remaining=remaining[1:]
    if args.check:
        need(not remaining and args.proof is None,'CPU compatibility check takes no target arguments')
        out=ROOT/'serialization'/f'check_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
        frozen=snapshot(out);started=time.perf_counter()
        try:
            torch,enum,identity=installed();result=check(out,frozen,torch,enum,identity)
            result.update(slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started)
            save(out/'summary.json',result);print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
        except BaseException as exc:
            save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen));raise
    else:
        need(args.proof is not None,'Passed exact-enum CPU proof required')
        torch,enum,identity=installed();dispatch(args,remaining,torch,enum,identity)


if __name__=='__main__':main()
