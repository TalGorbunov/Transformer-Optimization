"""Plot all six fixed learned-selection models from a completed CPU report.

This consumer reads JSON only and never imports a model, core or tensor cache.
Plotting is restricted to CPU Slurm; frozen report sources remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

REPO=Path(__file__).resolve().parents[1]
OWN=('scripts/plot_native_identity_join_learned.py','slurm/native_identity_join_learned_plot.sbatch')
MODES=('clip','sigmoid','softmax');SEEDS=(22,23);LENGTHS=(32,64)
REPORT_SOURCE='scripts/report_native_identity_join_learned_v2.py'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def report_counts(directory):
    summary_path=directory/'summary.json';summary=read(summary_path)
    need(summary['passed'] is True and summary['completed'] is True and summary['phase']=='report',
         'A completed independent result report is required, regardless of efficacy')
    need(Path(summary['analysis_file']).resolve()==directory/'analysis.json'
         and sha(directory/'analysis.json')==summary['analysis_sha256'],'Analysis binding differs')
    analysis=read(directory/'analysis.json')
    need(analysis['passed'] is True and analysis['completed'] is True
         and analysis['protocol']=='identity_join_learned_selection_result'
         and analysis['source_sha256']==summary['source_sha256'] and analysis['criteria']==summary['criteria'],
         'Independent report/source/decision ownership differs')
    need(REPORT_SOURCE in summary['source_sha256'],'The v2 independent reporter must own this result')
    for name,digest in summary['source_sha256'].items():
        need(sha(REPO/name)==digest and sha(directory/'source'/name.replace('/','_'))==digest,
             'Frozen independent report source differs: '+name)
    outcomes_path=directory/'outcomes.json'
    need(Path(analysis['outcomes_file']).resolve()==outcomes_path
         and sha(outcomes_path)==analysis['outcomes_sha256'],'All-outcome binding differs')
    rows=read(outcomes_path)['rows'];runs=analysis['runs'];matrix={(m,s) for m in MODES for s in SEEDS}
    need(len(runs)==6 and {(r['mode'],r['seed']) for r in runs}==matrix and len(rows)==2592,
         'All three modes, two seeds and 432 outcomes per model are required')
    groups=defaultdict(list);identities=set()
    for row in rows:
        key=(row['mode'],row['seed']);cell=row['cell'];n=row['n_frames']
        need(key in matrix and n in LENGTHS and cell in ('test_seen_N'+str(n),'test_held_N'+str(n))
             and type(row['exact']) is bool,'Malformed plotted endpoint')
        identity=(*key,cell,row['sid']);need(identity not in identities,'Duplicate plotted outcome');identities.add(identity)
        groups[(*key,cell)].append(row)
    counts=[]
    for run in runs:
        for regime in ('seen','held'):
            for n in LENGTHS:
                cell=f'test_{regime}_N{n}';selected=groups[(run['mode'],run['seed'],cell)]
                actual=sum(row['exact'] for row in selected);record=run['statistics']['cells'][cell]
                need(len(selected)==record['examples']==108 and actual==record['correct']
                     and abs(record['accuracy']-actual/108)<1e-12,'Plot counts differ from all retained outcomes')
                counts.append(dict(mode=run['mode'],seed=run['seed'],regime=regime,n_frames=n,
                                   correct=actual,examples=108,accuracy_percent=100*actual/108))
    bindings={str(path):sha(path) for path in (summary_path,directory/'analysis.json',outcomes_path)}
    return counts,bindings,analysis['criteria']


def draw(counts,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'clip':'#0072B2','sigmoid':'#D55E00','softmax':'#009E73'}
    styles={22:dict(linestyle='-',marker='o'),23:dict(linestyle='--',marker='s')}
    lookup={(r['mode'],r['seed'],r['regime'],r['n_frames']):r['accuracy_percent'] for r in counts}
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'}):
        figure,axes=plt.subplots(1,2,figsize=(10.8,5.5),sharey=True)
        for ax,regime,title,target_count in zip(axes,('seen','held'),('Seen room pairs','Held room pairs'),(98,87)):
            ax.set_title(title,fontsize=13,pad=11)
            for mode in MODES:
                for seed in SEEDS:
                    ax.plot(LENGTHS,[lookup[(mode,seed,regime,n)] for n in LENGTHS],color=colors[mode],
                        linewidth=2,markersize=6,alpha=.9,**styles[seed])
            target=100*target_count/108;chance=100/9
            ax.axhline(target,color='#666666',linestyle='-.',linewidth=1,zorder=0)
            ax.text(.02,target+1.3,f'Absolute target: {target_count}/108 ({target:.1f}%)',
                    transform=ax.get_yaxis_transform(),color='#555555',fontsize=8.5)
            ax.axhline(chance,color='#888888',linestyle=':',linewidth=1,zorder=0)
            ax.text(.02,chance+1.3,'Uniform over 9 names: 11.1%',transform=ax.get_yaxis_transform(),
                    color='#666666',fontsize=8.5)
            ax.set_xticks(LENGTHS);ax.set_xlim(28,68);ax.set_ylim(-2,105)
            ax.set_xlabel('Number of images (N)');ax.grid(axis='y',color='#eeeeee',linewidth=.7,zorder=-1)
        axes[0].set_ylabel('Exact name + EOS (%)')
        handles=[Line2D([0],[0],color=colors[m],linewidth=2,label=m.capitalize()) for m in MODES]
        handles += [Line2D([0],[0],color='#444444',linewidth=1.5,label=f'Seed {s}',**styles[s]) for s in SEEDS]
        figure.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=5,frameon=False)
        figure.suptitle('Learned selection on the identity join',fontsize=16,y=.98)
        figure.text(.5,.025,'108 contexts per point. Absolute target lines do not establish the relative criterion.',
                    ha='center',fontsize=9,color='#555555')
        figure.subplots_adjust(left=.085,right=.98,top=.86,bottom=.24,wspace=.12)
        paths=[out/'accuracy_by_length.png',out/'accuracy_by_length.pdf']
        for path in paths:figure.savefig(path,dpi=220,bbox_inches='tight')
        plt.close(figure)
    return {str(path):sha(path) for path in paths},matplotlib.__version__


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-directory',type=Path,required=True)
    args=parser.parse_args()
    need(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_PARTITION')=='cpu'
         and not os.environ.get('SLURM_JOB_GPUS'),'Plotting requires CPU Slurm')
    directory=args.report_directory.resolve();out=directory/'plots'/f'plot_{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();frozen={p:sha(REPO/p) for p in OWN}
    (out/'source').mkdir()
    for name in OWN:(out/'source'/name.replace('/','_')).write_bytes((REPO/name).read_bytes())
    try:
        counts,bindings,criteria=report_counts(directory);outputs,version=draw(counts,out)
        need(all(sha(path)==digest for path,digest in bindings.items()) and {p:sha(REPO/p) for p in OWN}==frozen,
             'Plot inputs or source changed during execution')
        save(out/'summary.json',dict(passed=True,completed=True,protocol='identity_join_learned_selection_plot',
            report_directory=str(directory),input_sha256=bindings,source_sha256=frozen,output_sha256=outputs,
            counts=counts,report_criteria=criteria,all_modes_and_seeds_retained=True,denominator_per_point=108,
            chance_baseline=1/9,baseline_interpretation='uniform over nine names; not an optimal task-aware chance bound',absolute_targets=dict(seen=98/108,held=87/108),
            targets_do_not_imply_relative_success=True,no_model_or_tensor_loads=True,
            matplotlib_version=version,slurm_job_id=os.environ['SLURM_JOB_ID'],seconds=time.perf_counter()-started))
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),source_sha256=frozen));raise


if __name__=='__main__':main()
