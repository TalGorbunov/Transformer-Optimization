"""CPU-only diagnosis of the preserved native-memory report443139 failure."""
from pathlib import Path
import hashlib,json,os,sys,time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts import profile_native_learned_memory as original
from scripts.stage_native_vision_v6_teacher import need,read,save,sha
RUNS=('profile_443135_qwen','profile_443134_cosmos')
ORIGINAL_SHA='74f92f1f916e4573e00e3b17b5b241984b3ee41266992e541df98436ac0f3898'
OUT=REPO/'outputs/native_aggregation_vlm/learned_memory/software'

def main():
 import torch
 torch.set_num_threads(4);started=time.perf_counter()
 out=OUT/('report_diagnosis_'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
 names=('scripts/diagnose_native_learned_memory_report.py','slurm/native_learned_memory_report_diagnosis.sbatch')
 sources={**original.sources(),**{n:sha(REPO/n) for n in names}};(out/'source').mkdir()
 for n,h in sources.items():
  need(sha(REPO/n)==h,'Source changed');(out/'source'/n.replace('/','_')).write_bytes((REPO/n).read_bytes())
 save(out/'source_hashes.json',sources)
 try:
  need(sha(REPO/'scripts/profile_native_learned_memory.py')==ORIGINAL_SHA,'Original reporter changed')
  failure=OUT/'report_443139/failure.json';need(read(failure)['message']=='Fixed prefix/write coverage differs','Different original failure')
  results=[]
  for name in RUNS:
   run=OUT/name;summary=read(run/'summary.json');need(summary['passed'] and summary['completed'],'GPU profile incomplete')
   records=read(run/'forwards.json');saved=read(run/'comparisons.json');bylabel={r['label']:r for r in records}
   need(len(records)==43 and len(saved)==328 and len(bylabel)==43,'Original inventory changed')
   differences=[];recomputed=[];raw_bindings={};field_max={};flags_equal=True
   groups={}
   for row in saved:groups.setdefault((row['cached_label'],row['full_label']),[]).append(row)
   need(len(groups)==8,'Eight paired histories required')
   for (cached,full),rows in sorted(groups.items()):
    tensors=[]
    for label in (cached,full):
     r=bylabel[label];need(sha(r['path'])==r['sha256'],'Original raw changed');raw_bindings[r['path']]=r['sha256']
     value=torch.load(r['path'],map_location='cpu',weights_only=True);need(value['label']==label,'Raw label differs')
     tensors.append(value['native_logits'].clone());del value
    for row in rows:
     m=original.v11.old.metric(torch,tensors[0][row['row'],0],tensors[1][row['row'],0]);new={k:v for k,v in row.items() if k not in m};new.update(m);recomputed.append(new)
     flags_equal &= all(m[k]==row[k] for k in ('top1_equal','numerical_rule_passed'))
     for field,value in m.items():
      old=row[field]
      if value!=old:
       delta=None if isinstance(value,bool) else abs(value-old)
       rel=None if delta is None else delta/max(abs(value),abs(old),1e-300)
       differences.append(dict(n_frames=row['n_frames'],placement=row['placement'],step=row['step'],row=row['row'],field=field,saved=old,recomputed=value,absolute=delta,relative=rel))
       if delta is not None:
        maxima=field_max.setdefault(field,dict(absolute=0.,relative=0.,count=0));maxima['absolute']=max(maxima['absolute'],delta);maxima['relative']=max(maxima['relative'],rel);maxima['count']+=1
    del tensors
   result=dict(run=str(run),summary_sha256=sha(run/'summary.json'),saved_sha256=sha(run/'comparisons.json'),raw_bindings=raw_bindings,
      comparisons=len(recomputed),all_flags_equal=flags_equal,original_descriptive_failures=sum(not r['numerical_rule_passed'] for r in saved),
      recomputed_descriptive_failures=sum(not r['numerical_rule_passed'] for r in recomputed),differences=differences,field_maxima=field_max)
   save(out/(summary['model_key']+'_diagnosis.json'),result);results.append({k:v for k,v in result.items() if k not in ('raw_bindings','differences')})
  save(out/'summary.json',dict(passed=True,completed=True,diagnostic_only=True,no_threshold_or_source_change=True,source_sha256=sources,
      original_failure_file=str(failure),original_failure_sha256=sha(failure),runs=results,slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))
  print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
 except Exception as e:
  save(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise
if __name__=='__main__':main()
