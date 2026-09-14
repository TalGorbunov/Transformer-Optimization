"""Plot verified V15 cells; no new estimates, fitting or selection. CPU Slurm only."""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--analysis',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
        raise SystemExit('CPU Slurm required')
    analysis=args.analysis.resolve();data=json.loads(analysis.read_text());summary=json.loads((analysis.parent/'summary.json').read_text())
    assert data['passed'] is True and data['audit_passed'] is True and summary['passed'] is True
    assert summary['analysis_sha256']==digest(analysis) and Path(summary['analysis_file']).resolve()==analysis
    assert data['records']==272 and data['no_practical_milestone_claim'] is True
    out=ROOT/'outputs/native_aggregation_vlm/v15'/('figure_'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['centered_s18','centered_s19','offset_s18','offset_s19'];labels=['Centered 18','Centered 19','Offset 18','Offset 19']
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True);rows=[]
    for row,n in enumerate((32,64)):
        for col,(scope,label,total) in enumerate((('all','All K0–16',17),('K9_16','K9–16',8))):
            ax=axes[row,col]
            for mode,shift,color,title in [('learned',-.19,'#718096','Learned mean'),('bank',.19,'#1565b3','Empirical bank mean')]:
                values=[data['cells'][f'{name}/N{n}/{mode}'][scope] for name in names]
                assert all(v['n']==total and type(v['correct']) is int and 0<=v['correct']<=total for v in values)
                bars=ax.bar([i+shift for i in range(4)],[100*v['correct']/v['n'] for v in values],width=.36,color=color,label=title)
                for name,bar,v in zip(names,bars,values):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{v["correct"]}/{v["n"]}',ha='center',fontsize=9,color=color)
                    rows.append(dict(model=name,n_frames=n,mode=mode,scope=scope,correct=v['correct'],n=v['n']))
            ax.set_title(f'{n} frames · {label}',loc='left',fontsize=13)
            ax.set_xticks(range(4),labels,fontsize=10);ax.set_ylim(0,113);ax.set_yticks(range(0,101,20))
            ax.set_axisbelow(True);ax.grid(axis='y',color='#e0e5eb');ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Complete integer + EOS (%)')
    fig.suptitle('MMReD Vision: does the empirical mean repair native answers?',x=.06,ha='left',fontsize=18)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.945),ncol=2,frameon=False)
    fig.text(.06,.02,'Exploratory reuse: 17 paired families, four frozen models, maximum training length 16.\nBoth conditions use N+24 reference-image streams plus one global stream; only the chosen mean differs.',fontsize=10,color='#505965')
    fig.subplots_adjust(left=.07,right=.985,top=.84,bottom=.13,hspace=.32,wspace=.17)
    files=[]
    for suffix in ('png','pdf'):
        p=out/f'native_vision_v15.{suffix}';fig.savefig(p,dpi=180);files.append(dict(path=str(p),sha256=digest(p)))
    plt.close(fig)
    (out/'plot_input.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(passed=True,analysis_file=str(analysis),analysis_sha256=digest(analysis),source_sha256=digest(__file__),plot_input_sha256=digest(out/'plot_input.json'),files=files,no_new_estimates=True,no_practical_milestone_claim=True),indent=2)+'\n')
    (out/'INDEX.md').write_text('# V15 existing-cell figure\n\n![Native learned/bank accuracy](native_vision_v15.png)\n\n[PDF](native_vision_v15.pdf) · [Provenance](summary.json).\n')
    print(json.dumps(dict(passed=True,out=str(out))))


if __name__=='__main__':main()
