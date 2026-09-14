"""Held CPU-only fixed native semantic gates from every V17 training readout.

No head/model forward or fitting. Original empty-prefix image/question features
own the gates; every later prefix reuses that owner. QA is provenance only.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts import audit_native_vision_v17_local_readout as v17
need,read,sha,object_sha,save=v17.need,v17.read,v17.sha,v17.oid,v17.save
DATA=Path('/mnt/data/gabriele/gnn_transformer/v18_semantic_gates')
OUT=REPO/'outputs/native_aggregation_vlm/v18/gate_staging'
CACHE=DATA/'feature_cache.json'
PARENT=DATA.parent/'v10_parallel_local/feature_cache.json'
REPORT=REPO/'outputs/native_aggregation_vlm/v17/local_readout/report_442798/analysis.json'
REPORT_SHA='c59a178d6192623d6c277e8d32cfc51a9fad8f6782c8471ad82fd00f411b06cb'
PROTOCOL='v18_fixed_native_empty_prefix_gate_cache'
GATE_RULE='max(0,full_vocabulary_p1-full_vocabulary_p0); FP64 probabilities then detached FP32'
OWN=('scripts/stage_native_vision_v18_gates.py','slurm/native_vision_v18_gate_stage.sbatch',
     'tests/test_native_vision_v18_gates.py','gnnformer/parallel_local_semantic_gate.py',
     'gnnformer/parallel_local_semantic_aggregation.py')
IDENTITY_KEYS={'model','runtime','processor','native_api','native_dtypes','norm_weight','head_weight',
               'norm_source_sha256','rms_norm_eps','count_token_ids'}


def sources():return {**v17.sources(),**{name:sha(REPO/name) for name in OWN}}


def snapshot(out):
    (out/'source').mkdir();frozen=sources()
    for name,h in frozen.items():
        target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes())
        need(sha(target)==h,'Source changed during snapshot')
    save(out/'source_hashes.json',frozen)
    (out/'INDEX.md').write_text('# V18 fixed semantic gates\n\n[Summary](summary.json) · [Proof](proof.json) · [Sources](source_hashes.json). No head/model calls or fitting.\n')
    return frozen


def native_identity_payload(**metadata):
    need(set(metadata)==IDENTITY_KEYS,'Exact native identity fields required')
    value=json.loads(json.dumps(metadata,sort_keys=True,allow_nan=False))
    need(value['count_token_ids']=={'0':15,'1':16} and value['native_dtypes']=={'norm':'torch.float16','lm_head':'torch.float16'},
         'Native gate token/dtype identity changed')
    for key,shape in [('norm_weight',[3584]),('head_weight',[152064,3584])]:
        need(value[key]['shape']==shape and value[key]['dtype']=='torch.float16' and len(value[key]['sha256'])==64,
             'Native norm/head tensor identity differs')
    return dict(schema_version=1,gate_rule=GATE_RULE,**value)


def gates_from_probabilities(torch,p0,p1):
    need(isinstance(p0,torch.Tensor) and isinstance(p1,torch.Tensor) and p0.dtype==p1.dtype==torch.float64
         and p0.ndim==1 and p0.shape==p1.shape and p0.device==p1.device,'Aligned FP64 full-vocabulary probabilities required')
    need(bool((torch.isfinite(p0)&torch.isfinite(p1)&(p0>=0)&(p1>=0)&(p0<=1)&(p1<=1)&(p0+p1<=1+1e-12)).all()),
         'Invalid full-vocabulary probability mass')
    with torch.no_grad():return (p1-p0).clamp_min(0).float().detach()


def broadcast_gates(torch,gates,n_positions,*,padded_items=None,mode='native_gate'):
    need(isinstance(gates,torch.Tensor) and gates.dtype==torch.float32 and gates.ndim==1
         and bool((torch.isfinite(gates)&(gates>=0)&(gates<=1)).all()),'Finite FP32 origin gates required')
    n=len(gates);width=n if padded_items is None else padded_items
    need(type(n_positions) is int and n_positions>0 and type(width) is int and width>=n
         and mode in ('native_gate','all_open'),'Invalid causal-query or valid/padded item layout')
    # Both conditions receive the same computed origin gates; padding is always0.
    applied=gates.detach() if mode=='native_gate' else torch.ones_like(gates)
    result=torch.zeros((width,n_positions),dtype=torch.float32,device=gates.device)
    result[:n]=applied[:,None]
    return result.detach()


def build_origin_map(features,scenes):
    local={fid:f for fid,f in features.items() if f['kind']=='local'};by_pair={}
    for fid,f in local.items():
        need(fid==object_sha(['local',f['pair_id'],f['prefix_ids']]),'Local feature key includes different inputs')
        if not f['prefix_ids']:
            need(f['pair_id'] not in by_pair,'Duplicate original empty feature owner');by_pair[f['pair_id']]=fid
    mapping={}
    for fid,f in local.items():
        need(f['pair_id'] in by_pair,'Continuation lacks original empty feature')
        owner=by_pair[f['pair_id']];origin=local[owner]
        need(f['question_sha256']==origin['question_sha256'],'Continuation changed the question')
        mapping[fid]=owner
    result={}
    for sid,s in scenes.items():
        rows=s['local_feature_ids'];need(rows and all(row for row in rows),'Scene has empty feature inventory')
        need(all(len(row)==len(rows[0]) for row in rows),'Scene causal positions differ between image streams')
        owners=[]
        for row in rows:
            need(row[0] in local and local[row[0]]['prefix_ids']==[],'First feature must be original empty prefix')
            need(all(fid in mapping and mapping[fid]==row[0] for fid in row),'One image stream changes gate owner')
            owners.append(row[0])
        result[sid]=dict(local_empty_feature_ids=owners,local_feature_ids=rows)
    return mapping,result


def bind(path,bindings,expected=None):
    path=Path(path).resolve();h=sha(path);need(expected is None or h==expected,'Bound artifact changed: '+str(path))
    need(str(path) not in bindings or bindings[str(path)]==h,'Conflicting artifact identity');bindings[str(path)]=h
    return dict(file=str(path),sha256=h)


def verify_cache(path=CACHE,*,ancestors=True,tensors=False):
    path=Path(path).resolve();cache=read(path)
    need(path==CACHE and path.with_suffix('.sha256').read_text().strip()==sha(path)
         and cache['schema_version']==1 and cache['protocol']==PROTOCOL and cache['complete'] is True
         and cache['training_only'] is True and cache['gate_rule']==GATE_RULE,'Complete canonical gate cache required')
    need(cache['source_sha256']==sources(),'Gate-cache source changed')
    directory=Path(cache['proof_directory']);proof=read(cache['proof_file'])
    need(Path(cache['proof_file'])==directory/'proof.json' and sha(cache['proof_file'])==cache['proof_sha256']
         and proof['passed'] is True and proof['source_sha256']==cache['source_sha256'],'Frozen gate CPU proof changed')
    for name,h in cache['source_sha256'].items():need(sha(directory/'source'/name.replace('/','_'))==h,'Archived source changed')
    need(cache['native_identity']==native_identity_payload(**{k:cache['native_identity'][k] for k in IDENTITY_KEYS})
         and cache['native_identity_sha256']==object_sha(cache['native_identity'])
         and proof['native_identity']==cache['native_identity'] and proof['native_identity_sha256']==cache['native_identity_sha256'],
         'Native identity payload differs')
    need(cache['parent_cache']['file']==str(PARENT) and sha(PARENT)==cache['parent_cache']['sha256']
         and proof['parent_cache']==cache['parent_cache'] and proof['artifact_bindings']==cache['artifact_bindings'],'Parent V10/proof binding changed')
    need(Path(cache['tensor_file'])==DATA/'gates.pt' and sha(cache['tensor_file'])==cache['tensor_sha256']
         and proof['tensor_file']==cache['tensor_file'] and proof['tensor_sha256']==cache['tensor_sha256']
         and proof['tensor_info']==cache['tensor_info'],'Consumed gate tensor archive changed')
    need(cache['feature_ids']==sorted(cache['gates']) and len(cache['gates'])==9512 and len(cache['scenes'])==1782
         and len(cache['prefix_to_empty_feature_id'])==32686,'Fixed training-only gate coverage changed')
    for i,fid in enumerate(cache['feature_ids']):need(cache['gates'][fid]['row']==i,'Gate row ownership changed')
    need(proof['gate_metadata_sha256']==object_sha(cache['gates'])
         and proof['broadcast_mapping_sha256']==object_sha(dict(scenes=cache['scenes'],prefix_to_empty_feature_id=cache['prefix_to_empty_feature_id'])),
         'Gate metadata or fixed-prefix mapping changed')
    if ancestors:
        for filename,h in cache['artifact_bindings'].items():need(sha(filename)==h,'Gate source artifact changed: '+filename)
        parent=read(PARENT);feature_plan=read(parent['plan_file'])
        mapping,scenes=build_origin_map(feature_plan['features'],feature_plan['scenes'])
        need(mapping==cache['prefix_to_empty_feature_id'] and scenes==cache['scenes'],'Parent fixed-origin coverage differs')
    if tensors:
        import torch
        packet=torch.load(cache['tensor_file'],map_location='cpu',weights_only=True)
        need(packet['schema_version']==1 and packet['feature_ids']==cache['feature_ids'],'Gate tensor identity/order differs')
        for key,info in cache['tensor_info'].items():need(v17.tensor_info(torch,packet[key])==info,'Gate tensor bytes changed')
        need(torch.equal(packet['gates'],gates_from_probabilities(torch,packet['p0'],packet['p1'])),'Stored gate formula differs')
        for i,fid in enumerate(packet['feature_ids']):
            need(all(float(packet[k][i])==cache['gates'][fid]['gate' if k=='gates' else k] for k in ('gates','p0','p1')),'Tensor/metadata scalar mismatch')
    return cache


def stage(out,frozen,tests):
    import torch
    torch.set_num_threads(4);started=time.perf_counter();bindings={}
    need(not CACHE.exists() and not (DATA/'gates.pt').exists(),'Preserve existing or partial gate-cache publication')
    need(sha(REPORT)==REPORT_SHA,'Only the completed fixed V17 analysis is authorized')
    analysis=read(REPORT);report_summary=read(REPORT.parent/'summary.json')
    need(analysis['passed'] is True and analysis['completed'] is True and report_summary['passed'] is True
         and report_summary['analysis_sha256']==REPORT_SHA and analysis['local_rows']==22568,'Completed V17 report required')
    v17.check_source_copy(analysis['source_sha256'],REPORT.parent)
    plan=v17.verify_plan(analysis['plan_file'],ancestors=True)
    for filename,h in plan['artifact_bindings'].items():bindings[filename]=h
    bind(REPORT,bindings,REPORT_SHA);bind(REPORT.parent/'summary.json',bindings)
    bind(analysis['plan_file'],bindings,analysis['plan_sha256']);bind(Path(analysis['plan_file']).with_suffix('.sha256'),bindings)
    bind(analysis['gpu_summary_file'],bindings,analysis['gpu_summary_sha256']);gpu=read(analysis['gpu_summary_file'])
    need(gpu['passed'] is True and gpu['numerical_gate_passed'] is True and gpu['head_calls']==432
         and gpu['initial_weight_identity']==gpu['final_weight_identity'],'Frozen complete native head execution required')
    bind(gpu['records_file'],bindings,gpu['records_sha256']);records=read(gpu['records_file'])
    bind(analysis['rescored_file'],bindings,analysis['rescored_sha256']);rescored=read(analysis['rescored_file'])
    original={r['feature_id']:r for r in rescored if r['dataset']=='v10'}
    need(len(original)==9512,'All training local readouts required; no filtering')
    parent_binding=bind(PARENT,bindings,plan['artifact_bindings'][str(PARENT)]);parent=read(PARENT)
    bind(parent['plan_file'],bindings,parent['plan_sha256']);feature_plan=read(parent['plan_file'])
    need(parent['complete'] is True and parent['training_only'] is True,'Complete frozen training features required')
    mapping,scenes=build_origin_map(feature_plan['features'],feature_plan['scenes'])
    need(len(mapping)==32686 and len(scenes)==1782,'All original training prefixes/scenes required')
    identity=native_identity_payload(model=plan['model'],runtime=plan['runtime'],processor=plan['processor'],native_api=parent['native_api'],
        native_dtypes=plan['native_dtypes'],norm_weight=gpu['final_weight_identity']['norm'],head_weight=gpu['final_weight_identity']['head'],
        norm_source_sha256=plan['norm_source_sha256'],rms_norm_eps=plan['rms_norm_eps'],count_token_ids=plan['count_token_ids'])
    gates={};calls=[];max_scalar_difference=0.
    for item,record in zip(plan['calls'],records):
        if item['kind']!='v10':continue
        need(item['call']==record['call'] and record['kind']=='v10' and record['input_sha256']==item['sha256'],'Original head-call ownership changed')
        bind(record['input_file'],bindings,item['sha256']);bind(record['raw_file'],bindings,record['raw_sha256'])
        blob=torch.load(record['raw_file'],map_location='cpu',weights_only=True);logits=blob['logits']
        need(blob['schema_version']==1 and logits.dtype==torch.float16 and logits.shape==(item['batch_size'],1,152064)
             and v17.tensor_info(torch,logits)==record['logits'],'Original full native FP16 logits changed')
        probabilities=v17.row_probabilities(torch,logits[:,0],plan['count_token_ids'])
        need(v17.close_values(probabilities,record['probabilities']),'Independent native probability replay differs')
        p0=torch.tensor([r['p0'] for r in probabilities],dtype=torch.float64);p1=torch.tensor([r['p1'] for r in probabilities],dtype=torch.float64)
        values=gates_from_probabilities(torch,p0,p1)
        for j,(meta,prob) in enumerate(zip(item['rows'],probabilities)):
            fid=meta['feature_id'];f=feature_plan['features'][fid];prior_row=original[fid];loc=parent['features'][fid]
            need(meta['role']=='actual' and fid not in gates and f['phase']=='local_empty' and f['prefix_ids']==[]
                 and fid==mapping[fid] and f['pair_id']==meta['pair_id']==prior_row['pair_id']
                 and loc['state_sha256']==meta['state_sha256']==prior_row['state_sha256']
                 and prior_row['call']==item['call'] and prior_row['row']==j,'Original feature/state ownership differs')
            need(v17.close_values(prob,{k:prior_row[k] for k in prob}),'Independent report probability differs')
            max_scalar_difference=max(max_scalar_difference,abs(prob['p0']-prior_row['p0']),abs(prob['p1']-prior_row['p1']))
            pair=feature_plan['pairs'][f['pair_id']]
            need(pair['image_sha256']==prior_row['image_sha256'] and pair['question']==f['question']
                 and f['question_sha256']==object_sha(f['question']),'Image/question origin differs')
            gates[fid]=dict(pair_id=f['pair_id'],image_sha256=pair['image_sha256'],question_sha256=f['question_sha256'],
                hidden_sha256=meta['state_sha256'],logits_sha256=v17.tensor_info(torch,logits[j,0])['sha256'],
                raw_file=record['raw_file'],raw_sha256=record['raw_sha256'],raw_row=j,head_call=item['call'],
                hidden_bundle_file=item['file'],hidden_bundle_sha256=item['sha256'],
                gate=float(values[j]),p0=float(p0[j]),p1=float(p1[j]))
        calls.append(item['call']);del blob,logits
    need(len(calls)==152 and set(gates)==set(feature_plan['groups']['local_empty'])==set(original),'All9512 original readouts required')
    feature_ids=sorted(gates)
    for i,fid in enumerate(feature_ids):gates[fid]['row']=i
    packet=dict(schema_version=1,feature_ids=feature_ids,
        gates=torch.tensor([gates[fid]['gate'] for fid in feature_ids],dtype=torch.float32),
        p0=torch.tensor([gates[fid]['p0'] for fid in feature_ids],dtype=torch.float64),
        p1=torch.tensor([gates[fid]['p1'] for fid in feature_ids],dtype=torch.float64))
    info={k:v17.tensor_info(torch,packet[k]) for k in ('gates','p0','p1')}
    need(torch.equal(packet['gates'],gates_from_probabilities(torch,packet['p0'],packet['p1'])),'Final fixed-gate precision differs')
    need(sources()==frozen,'Gate source changed before publication')
    destination=DATA/out.name;destination.mkdir(parents=True,exist_ok=False);torch.save(packet,destination/'gates.pt')
    os.link(destination/'gates.pt',DATA/'gates.pt');tensor_hash=sha(DATA/'gates.pt')
    proof=dict(schema_version=1,protocol=PROTOCOL,passed=True,tests=tests,source_sha256=frozen,
        tensor_file=str(DATA/'gates.pt'),tensor_sha256=tensor_hash,tensor_info=info,source_head_calls=calls,
        feature_count=9512,training_scenes=1782,local_prefix_features=32686,
        max_report_probability_difference=max_scalar_difference,gate_metadata_sha256=object_sha(gates),
        broadcast_mapping_sha256=object_sha(dict(scenes=scenes,prefix_to_empty_feature_id=mapping)),
        report_file=str(REPORT),report_sha256=REPORT_SHA,parent_cache=parent_binding,
        native_identity=identity,native_identity_sha256=object_sha(identity),artifact_bindings=bindings,
        original_empty_prefix_only=True,all_prefixes_reuse_origin=True,padding_gate=0,all_open_valid_gate=1,
        labels_used_in_gate_calculation=False,no_fit=True,head_calls=0,VLM_calls=0,vision_calls=0)
    save(out/'proof.json',proof)
    cache=dict(schema_version=1,protocol=PROTOCOL,complete=True,training_only=True,gate_rule=GATE_RULE,
        source_sha256=frozen,proof_directory=str(out),proof_file=str(out/'proof.json'),proof_sha256=sha(out/'proof.json'),
        parent_cache=parent_binding,tensor_file=str(DATA/'gates.pt'),tensor_sha256=tensor_hash,tensor_info=info,
        feature_ids=feature_ids,gates=gates,scenes=scenes,prefix_to_empty_feature_id=mapping,
        native_identity=identity,native_identity_sha256=object_sha(identity),artifact_bindings=bindings)
    save(CACHE,cache);CACHE.with_suffix('.sha256').write_text(sha(CACHE)+'\n')
    verify_cache(CACHE,ancestors=False,tensors=True)
    save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,tests_passed=True,tests=tests,
        cache_file=str(CACHE),cache_sha256=sha(CACHE),proof_file=str(out/'proof.json'),proof_sha256=sha(out/'proof.json'),
        source_sha256=frozen,feature_count=9512,training_scenes=1782,head_calls=0,VLM_calls=0,vision_calls=0,
        no_fit=True,seconds=time.perf_counter()-started,slurm_job_id=os.environ['SLURM_JOB_ID']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--stage',action='store_true');args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','1'))<=4,'CPU-only Slurm4 required')
    out=OUT/f'{"selftest" if args.self_test else "stage"}_{os.environ["SLURM_JOB_ID"]}';out.mkdir(parents=True,exist_ok=False)
    frozen=snapshot(out)
    result=subprocess.run([sys.executable,str(REPO/'tests/test_native_vision_v18_gates.py')],cwd=REPO,text=True,capture_output=True)
    (out/'tests.txt').write_text(result.stdout+result.stderr);need(result.returncode==0,'Gate cache selftests failed; see tests.txt')
    tests=dict(passed=True,log_file=str(out/'tests.txt'),log_sha256=sha(out/'tests.txt'))
    if args.stage:stage(out,frozen,tests)
    else:save(out/'summary.json',dict(passed=True,completed=True,protocol=PROTOCOL,tests_passed=True,tests=tests,source_sha256=frozen,no_fit=True,head_calls=0,VLM_calls=0,vision_calls=0))
    need(sources()==frozen,'Source changed during gate CPU work');print(json.dumps(dict(passed=True,output=str(out))),flush=True)


if __name__=='__main__':main()
