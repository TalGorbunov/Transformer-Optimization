"""CPU-only labelled-table repair of the unchanged feature assay443231.

Reconstruct labels and every numeric score from bound saved predictions. No
numerical library, tensor deserialization, model/head call, solve or refit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/native_aggregation_vlm/identity_join_feature_sufficiency'
PARENT=ROOT/'443231'
SUMMARY_SHA='01f5ebbbb991c7169f3c23e5a8acaf4014d936beed0e63e692157cc4a1b597e1'
METRICS_SHA='9b3ce53b5a303d3fc0a5de6ae4201e111a24c92c370d148dfb9ac1492825d201'
DATA=Path('/mnt/data/gabriele/gnn_transformer/identity_join_feature_sufficiency/443231')
OWN=('scripts/report_native_identity_join_feature_sufficiency_labels.py',
     'slurm/native_identity_join_feature_sufficiency_labels.sbatch')
FIELDS=(('question',),('person',),('room',),('person','room'),('question','person','room'),('step',))
METHODS=('ridge','global_majority','question_majority')


def need(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def groups(rows,fields):
    result=defaultdict(list)
    for row in rows:result[tuple(row[key] for key in fields)].append(row)
    return result


def summarize(rows,method,fields=()):
    result=[]
    for key,values in sorted(groups(rows,fields).items()):
        labels=dict(zip(fields,key));metrics=dict(rows=len(values),occurrences=sum(r['occurrence_weight'] for r in values))
        for target in ('person','room','joint'):
            correct=sum(r[method][target+'_correct'] for r in values)
            weighted=sum(r['occurrence_weight']*r[method][target+'_correct'] for r in values)
            metrics[target]=dict(correct=correct,total=len(values),accuracy=correct/len(values),
                occurrence_correct=weighted,occurrence_total=metrics['occurrences'],occurrence_accuracy=weighted/metrics['occurrences'])
        result.append(dict(labels=labels,metrics=metrics))
    return result


def old_projection(row):
    # Reproduce the original serialization collision ONLY for exact comparison.
    # Published repaired tables retain these namespaces separately.
    return {**row['labels'],**row['metrics']}


def fit_controls(rows,people,rooms,questions):
    fit=[row for row in rows if row['partition']=='fit'];result={}
    for key,values in [('global',fit)]+[(q,[r for r in fit if r['question']==q]) for q in questions]:
        predictions={};counts={};ties={}
        for label,order in (('person',people),('room',rooms)):
            frequencies=Counter(row[label] for row in values);count=[frequencies[c] for c in order];maximum=max(count)
            predictions[label]=order[count.index(maximum)];counts[label]=count;ties[label]=count.count(maximum)
        result[key]=dict(predictions=predictions,fit_counts=counts,tied_maxima=ties,fit_rows=len(values))
    return result


def validate_predictions(inventory,saved,occurrences,controls):
    rows=inventory['rows'];people=inventory['people'];rooms=inventory['rooms'];questions=inventory['questions']
    need(len(rows)==len(saved)==9007 and len(people)==len(set(people))==9 and len(rooms)==len(set(rooms))==6
         and len(questions)==len(set(questions))==12 and not any(inventory['metadata_errors'].values()),'Original row/classes/error inventory differs')
    need([r['feature_id'] for r in rows]==sorted({r['feature_id'] for r in rows}),'Exact unique feature order differs')
    by_id={r['feature_id']:r for r in rows};weights=Counter()
    need(len(occurrences)==72576 and len({(r['sid'],r['position']) for r in occurrences})==72576,'Occurrence ownership differs')
    for entry in occurrences:
        row=by_id[entry['feature_id']];weights[entry['feature_id']]+=1
        need(entry['image_sha256']==row['image_sha256'] and entry['step']==row['step']==entry['position']+1,
             'Occurrence/image/Step identity differs')
    need(all(weights[row['feature_id']]==row['occurrence_weight'] for row in rows),'Original occurrence weights differ')
    computed_controls=fit_controls(rows,people,rooms,questions);need(computed_controls==controls,'Fit-only frequency control differs')
    rebuilt=[]
    for row,pred in zip(rows,saved):
        need({key:pred[key] for key in row}==row and set(pred)==set(row)|{'scores','person_tied_maxima','room_tied_maxima',*METHODS},
             'Prediction input/label/ownership schema differs')
        need(row['person'] in people and row['room'] in rooms and row['question'] in questions
             and row['partition']==('fit' if row['step']%2 else 'evaluate') and 1<=row['step']<=16,'Original label/split differs')
        scores=pred['scores'];need(len(scores)==15 and all(type(x) is float and math.isfinite(x) for x in scores),'Saved scores malformed')
        a,b=scores[:9],scores[9:];pa,pr=people[a.index(max(a))],rooms[b.index(max(b))]
        need(pred['person_tied_maxima']==a.count(max(a)) and pred['room_tied_maxima']==b.count(max(b)), 'Saved score tie count differs')
        current=dict(row,person_tied_maxima=a.count(max(a)),room_tied_maxima=b.count(max(b)))
        for method,labels in [('ridge',dict(person=pa,room=pr)),('global_majority',computed_controls['global']['predictions']),
                              ('question_majority',computed_controls[row['question']]['predictions'])]:
            pc,rc=labels['person']==row['person'],labels['room']==row['room']
            current[method]=dict(**labels,person_correct=pc,room_correct=rc,joint_correct=pc and rc)
            need(pred[method]==current[method] and all(type(pred[method][key]) is bool for key in ('person_correct','room_correct','joint_correct')),
                 'Saved forced-choice prediction or correctness differs')
        rebuilt.append(current)
    return rebuilt


def reconstruct_coverage(rows,people,rooms,questions):
    result={};images={}
    for part,expected in (('fit',4472),('evaluate',4535)):
        values=[r for r in rows if r['partition']==part];images[part]={r['image_sha256'] for r in values}
        need(len(values)==expected and len(images[part])==432 and sum(r['occurrence_weight'] for r in values)==36288,'Original split count differs')
        result[part]=dict(features=expected,images=432,occurrences=36288,tables={})
        for fields in FIELDS:
            result[part]['tables']['/'.join(fields)]=[
                dict(zip(fields,key),features=len(v),images=len({r['image_sha256'] for r in v}),
                     steps=sorted({r['step'] for r in v}),occurrences=sum(r['occurrence_weight'] for r in v))
                for key,v in sorted(groups(values,fields).items())]
        cells=groups(values,('question','person','room'))
        need(set(cells)==set(itertools.product(questions,people,rooms)) and len(cells)==648
             and all(3<=len({r['step'] for r in v})<=8 for v in cells.values()),'Complete three-way support differs')
    need(not images['fit']&images['evaluate'] and len(images['fit']|images['evaluate'])==864,'Held-image disjointness differs')
    return result


def rebuild_metrics(rows,old):
    result={};tables=0
    for part in ('fit','evaluate'):
        values=[row for row in rows if row['partition']==part];result[part]={}
        for method in METHODS:
            overall=summarize(values,method)[0]
            need(old_projection(overall)==old[part][method]['overall'],'Original overall numeric scores differ')
            repaired={}
            for fields in FIELDS:
                name='/'.join(fields);table=summarize(values,method,fields)
                need([old_projection(r) for r in table]==old[part][method]['tables'][name],
                     'Original ordered stratum numeric scores differ: '+part+'/'+method+'/'+name)
                repaired[name]=table;tables+=len(table)
            result[part][method]=dict(overall=overall,tables=repaired)
        ties=dict(person_rows=sum(r['person_tied_maxima']>1 for r in values),room_rows=sum(r['room_tied_maxima']>1 for r in values))
        need(ties==old[part]['ties'],'Original aggregate tie count differs');result[part]['ties']=ties
    need(tables==4422,'Complete all-method/all-partition stratum coverage differs')
    return result,tables


def decisions(metrics):
    e=metrics['evaluate']['ridge'];overall=e['overall']['metrics'];base=metrics['evaluate']['question_majority']['overall']['metrics']['joint']['accuracy']
    result=dict(person=overall['person']['correct']>=4490,room=overall['room']['correct']>=4490,
        joint=overall['joint']['correct']>=4490,
        every_question=all(r['metrics']['joint']['correct']*100>=98*r['metrics']['rows'] for r in e['tables']['question']),
        every_person_room=all(r['metrics']['joint']['correct']*100>=95*r['metrics']['rows'] for r in e['tables']['person/room']),
        question_majority_advantage=overall['joint']['accuracy']-base>=.5)
    result['strong_linear_accessibility']=all(result.values());result.update(diagnostic_only=True,native_training_released=False)
    return result


def selftest():
    rows=[dict(person='Alpha',room='Kitchen',occurrence_weight=3,m=dict(person_correct=True,room_correct=False,joint_correct=False)),
          dict(person='Beta',room='Kitchen',occurrence_weight=1,m=dict(person_correct=False,room_correct=True,joint_correct=False))]
    table=summarize(rows,'m',('person','room'))
    need(table[0]['labels']==dict(person='Alpha',room='Kitchen') and table[1]['labels']['person']=='Beta'
         and table[0]['metrics']['person']['correct']==1 and table[0]['metrics']['room']['correct']==0,
         'Separate labels/metrics fixture failed')
    legacy=old_projection(table[0]);need(isinstance(legacy['person'],dict) and isinstance(legacy['room'],dict)
         and legacy==table[0]['metrics'],'Original collision reproduction fixture failed')
    total=summarize(rows,'m')[0]['metrics']
    need(total['person']['accuracy']==.5 and total['person']['occurrence_accuracy']==.75,'Numeric weighting fixture failed')
    return dict(passed=True,tests=['person_room_label_collision','legacy_numeric_projection','unique_occurrence_weights'])


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    need(os.environ.get('SLURM_JOB_PARTITION')=='cpu' and os.environ.get('SLURM_JOB_ID')
         and not os.environ.get('SLURM_JOB_GPUS') and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4,'CPU4 Slurm only')
    job=os.environ['SLURM_JOB_ID'];need(job.isdigit(),'Concrete job required')
    start=time.perf_counter();out=ROOT/('label_report_'+job);out.mkdir(parents=True,exist_ok=False);sources={};bindings={}
    try:
        need(sha(PARENT/'summary.json')==SUMMARY_SHA and sha(PARENT/'metrics.json')==METRICS_SHA,'Exact original result changed')
        parent=read(PARENT/'summary.json');sources={**parent['source_sha256'],**{n:sha(REPO/n) for n in OWN}};(out/'source').mkdir()
        for name,digest in sources.items():
            need(sha(REPO/name)==digest,'Frozen current source differs')
            target=out/'source'/name.replace('/','_');target.write_bytes((REPO/name).read_bytes());need(sha(target)==digest,'New source snapshot differs')
            if name in parent['source_sha256']:need(sha(PARENT/'source'/name.replace('/','_'))==digest,'Original source copy differs')
        save(out/'source_hashes.json',sources)
        save(out/'request.json',dict(job_id=job,parent_summary=str(PARENT/'summary.json'),parent_sha256=SUMMARY_SHA,
            source_sha256=sources,report_only=True,maximum_seconds=60))
        error=dict(type='reporting_label_key_collision',original_file=str(PARENT/'metrics.json'),original_sha256=METRICS_SHA,
            original_source='scripts/diagnose_native_identity_join_feature_sufficiency.py',
            cause='score_table person/room label keys were overwritten by same-named numeric metric dictionaries',
            correction='Reconstruct every stratum from unchanged saved predictions; publish separate labels and metrics objects',
            original_fit_and_results_preserved=True,no_refit=True,no_threshold_change=True)
        save(out/'original_report_error.json',error)
        need(parent['passed'] is True and parent['completed'] is True and parent['job_id']=='443231'
             and parent['calls']==dict(model=0,vision=0,head=0,gpu=0),'Original completed CPU assay differs')
        bindings[str(PARENT/'summary.json')]=SUMMARY_SHA
        for filename,digest in parent['files'].items():need(sha(filename)==digest,'Original bound output changed');bindings[filename]=digest
        need(bindings[str(PARENT/'metrics.json')]==METRICS_SHA,'Original metric binding differs')
        roundtrip=read(PARENT/'checkpoint_roundtrip.json')
        need(roundtrip==parent['checkpoint_roundtrip'] and roundtrip['passed'] is True
             and roundtrip['predictions_use_reloaded_tensors'] is True
             and roundtrip['sha256']==parent['ridge_checkpoint_sha256']==bindings[parent['ridge_checkpoint_file']]
             and roundtrip['tensors']==parent['model_tensors'],'Original checkpoint/prediction identity differs')
        resolved=read(PARENT/'resolved_inputs.json');inventory=read(DATA/'input_inventory.json')
        need(resolved['inventory_sha256']==bindings[str(DATA/'input_inventory.json')]
             and resolved['occurrence_inventory_sha256']==bindings[str(DATA/'occurrence_inventory.json')], 'Original resolved inventory binding differs')
        tests=selftest();save(out/'selftest.json',tests)
        predictions=read(DATA/'predictions.json');controls=read(PARENT/'controls.json')
        rows=validate_predictions(inventory,predictions,read(DATA/'occurrence_inventory.json'),controls)
        coverage=reconstruct_coverage(rows,inventory['people'],inventory['rooms'],inventory['questions'])
        need(coverage==read(PARENT/'coverage.json')==parent['coverage'],'Original coverage labels/counts differ')
        old=read(PARENT/'metrics.json');corrected,table_rows=rebuild_metrics(rows,old);decision=decisions(corrected)
        need(decision==parent['decisions'],'Original diagnostic decision changed')
        need(corrected['fit']['ridge']['overall']['metrics']==parent['fit_metrics']
             and corrected['evaluate']['ridge']['overall']['metrics']==parent['evaluation_metrics']
             and corrected['evaluate']['question_majority']['overall']['metrics']==parent['question_majority_evaluation'],
             'Original summary scores differ')
        save(out/'labelled_metrics.json',dict(schema_version=2,partitions=corrected,labels_and_metrics_separate=True))
        save(out/'input_bindings.json',bindings)
        need(all(sha(REPO/n)==digest for n,digest in sources.items()),'Source changed during report')
        files={name:dict(file=str(out/name),sha256=sha(out/name)) for name in
               ('labelled_metrics.json','input_bindings.json','selftest.json','original_report_error.json')}
        elapsed=time.perf_counter()-start;need(elapsed<=60,'Fixed CPU report cap exceeded')
        summary=dict(passed=True,completed=True,protocol='native_identity_join_feature_sufficiency_label_repair',job_id=job,
            parent_summary_file=str(PARENT/'summary.json'),parent_summary_sha256=SUMMARY_SHA,source_sha256=sources,
            files=files,decisions=decision,tests=tests,seconds=elapsed,all_original_numeric_metrics_exact=True,
            all_original_predictions_and_ties_exact=True,all_original_coverage_exact=True,
            unique_feature_rows=9007,fit_rows=4472,evaluation_rows=4535,three_way_cells_per_partition=648,
            all_method_partition_stratum_rows=table_rows,original_checkpoint_roundtrip=roundtrip,
            evaluation_metrics=parent['evaluation_metrics'],no_refit=True,no_new_predictions=True,
            no_tensor_deserialization=True,no_threshold_change=True,original_results_preserved=True,
            calls=dict(model=0,vision=0,head=0,gpu=0,solver=0),native_training_released=False)
        save(out/'summary.json',summary)
        (out/'REPORT.md').write_text('# Feature sufficiency: labelled-table repair\n\n'
            'All original counts, occurrence weights, ties, controls and decisions were independently reproduced from saved scores and predictions. '
            'Person/room stratum identities now occupy a separate labels object. The original fit and reports remain unchanged.\n\n'
            'Held even-Step accuracy remains person 4535/4535; room and joint 4379/4535. The strict diagnostic still fails. '
            'No model, head, tensor deserialization or solver was executed.\n\n'
            '[Summary](summary.json) · [Labelled metrics](labelled_metrics.json) · [Recorded reporting defect](original_report_error.json)\n')
        print(json.dumps(dict(passed=True,directory=str(out))),flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),job_id=job,source_sha256=sources,
            input_bindings=bindings,seconds=time.perf_counter()-start,original_results_preserved=True,no_refit=True));raise


if __name__=='__main__':main()
