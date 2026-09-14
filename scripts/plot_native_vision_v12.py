"""Post-result visualization of the independently audited fixed V12 study."""
import os,json,hashlib
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
report=root/'outputs/native_aggregation_vlm/v12/reference_study/report_442359/analysis.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(report)=='6ef7b842f0da4f35b526af2c2870a634086e5c16347842527b4437302b3e855b'
a=json.loads(report.read_text());assert a['passed'] and a['records']==408
out=root/'outputs/native_aggregation_vlm/v12'/('figure_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
fig,axes=plt.subplots(1,3,figsize=(13.8,4.8),sharey=True)
colors=['#8d99ae','#007f86','#d99a42'];modes=['base','background','sham'];labels=['Baseline','Reference centering','Orthogonal control']
keys=a['policy']['core_keys'];names=['CE\nseed 14','CE\nseed 15','Consistency\nseed 14','Consistency\nseed 15']
points=[]
for ax,n,scope,title in zip(axes,[32,64,64],['all','all','K10_16'],['32 frames · counts 0–16','64 frames · counts 0–16','64 frames · counts 10–16']):
 for mi,(mode,color,label) in enumerate(zip(modes,colors,labels)):
  values=[a['cells'][f'{key}/N{n}/{mode}'][scope] for key in keys]
  xpos=[i+(mi-1)*.24 for i in range(4)];heights=[100*v['accuracy'] for v in values]
  bars=ax.bar(xpos,heights,width=.22,color=color,label=label,zorder=3)
  for key,bar,v in zip(keys,bars,values):
   assert v['n']==(7 if scope=='K10_16' else 17) and abs(bar.get_height()-100*v['correct']/v['n'])<1e-10
   ax.annotate(str(v['correct']),xy=(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8,color=color)
   points.append(dict(core=key,n_frames=n,scope=scope,mode=mode,correct=v['correct'],denominator=v['n'],bar_height=bar.get_height()))
 ax.set_title(title,fontsize=12,pad=13);ax.set_xticks(range(4),names,fontsize=9);ax.set_ylim(0,107);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.2,zorder=0)
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Complete integer + EOS accuracy (%)',fontsize=10)
fig.suptitle('Reference centering improves native answers; higher counts remain difficult',fontsize=15,y=.98)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.89),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.015,'17 reused paired families across four frozen models. Numbers above bars are correct answers. Every condition processes N + 25 streams.',ha='center',fontsize=9,color='#475569')
fig.subplots_adjust(top=.73,bottom=.19,left=.065,right=.99,wspace=.12)
for ext in ('png','pdf'):fig.savefig(out/('native_vision_v12.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)
source=Path(__file__);(out/source.name).write_bytes(source.read_bytes())
summary=dict(passed=True,report_file=str(report),report_sha256=sha(report),source_sha256=sha(source),bars=points,figure_hashes={e:sha(out/('native_vision_v12.'+e)) for e in ('png','pdf')},post_result_visualization=True,slurm_job_id=os.environ['SLURM_JOB_ID'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'INDEX.md').write_text('# V12 audited figure\n\n[PNG](native_vision_v12.png) · [PDF](native_vision_v12.pdf) · [Binding](summary.json).\n');print(json.dumps(dict(passed=True,directory=str(out),bars=len(points))))
