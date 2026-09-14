"""Analyze the preregistered clean-vision attribution block on CPU Slurm."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

ARMS = ('global', 'hidden', 'middle_lora', 'lora')
LABELS = dict(global_='Global read', hidden='Hidden only', middle_lora='Middle + upper LoRA', lora='Upper LoRA')
LABELS['global'] = LABELS.pop('global_')
CELLS = ('length_N16', 'length_N32', 'length_N64', 'unseen_count_N32', 'unseen_count_N64')
DATA_ROOT = Path('/mnt/data/gabriele/gnn_transformer/v2_clean')
CHECKPOINT_ROOT = Path('/mnt/ckpts/gabriele/gnn_transformer/native_aggregation_vlm_v2')
CELL_SPECS = {
    **{f'train_N{n}': (90, range(9), 10) for n in (8, 16)},
    **{f'dev_N{n}': (36, range(9), 4) for n in (8, 16)},
    **{f'length_N{n}': (108, range(9), 12) for n in (16, 32, 64)},
    **{f'unseen_count_N{n}': (64, range(9, 17), 8) for n in (32, 64)},
}


def verify_manifests(run, config, manifest):
    """Bind selected records to byte-verified staged and frozen source manifests."""
    sources = {}
    for copied, configured, hash_field, published, family in (
        ('staged_manifest.json', 'manifest', 'manifest_sha256', 'main_manifest.json', 'length'),
        ('staged_count_manifest.json', 'count_manifest', 'count_manifest_sha256', 'count_manifest.json', 'unseen_count'),
    ):
        external = DATA_ROOT / published
        if Path(config[configured]).resolve() != external.resolve():
            raise ValueError('Manifest path differs from the registered source')
        raw = (run / copied).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != config[hash_field] or hashlib.sha256(external.read_bytes()).hexdigest() != sha:
            raise ValueError('Copied or frozen source manifest hash differs')
        source = json.loads(raw)
        if (source.get('schema_version') != 1 or not isinstance(source.get('splits'), dict)
                or Path(source['dataset_root']).resolve() != DATA_ROOT.resolve()):
            raise ValueError('Invalid registered manifest schema or dataset root')
        for key, cell in source['splits'].items():
            mapped = key.replace('test_', family + '_', 1) if key.startswith('test_') else key
            if mapped in sources:
                raise ValueError('Duplicate source cell across manifests')
            sources[mapped] = cell['samples']
    if set(sources) != set(CELL_SPECS) or set(manifest) != set(CELL_SPECS):
        raise ValueError('Unexpected or missing registered data cells')
    fields = ('path', 'sid', 'n_frames', 'gold', 'qa_sha256', 'pair_id', 'anchor_id',
              'test_family', 'parent_n_frames', 'parent_positions', 'anchor_positions')
    seen_paths = set()
    for cell, (count, golds, per_gold) in CELL_SPECS.items():
        expected, selected = sources[cell], manifest[cell]['samples']
        if len(expected) != count or len(selected) != count or manifest[cell]['n'] != count:
            raise ValueError(f'Incorrect sample count in {cell}')
        if Counter(r['gold'] for r in expected) != Counter({k: per_gold for k in golds}):
            raise ValueError(f'Incorrect gold support or balance in {cell}')
        n_frames = int(cell.rsplit('_N', 1)[1])
        for source, actual in zip(expected, selected):
            if any(actual.get(field) != source.get(field) for field in fields):
                raise ValueError(f'Selected records differ from staged order/content in {cell}')
            if source['n_frames'] != n_frames or source['path'] in seen_paths:
                raise ValueError('Invalid frame count or duplicated selected path')
            seen_paths.add(source['path'])
            if len(source['image_files']) != n_frames or len(actual['image_files']) != n_frames:
                raise ValueError('Manifest image count differs')
            for source_image, actual_image in zip(source['image_files'], actual['image_files']):
                if any(actual_image.get(field) != source_image.get(field)
                       for field in ('path', 'bytes', 'sha256')):
                    raise ValueError('Selected image provenance differs from staged manifest')


def verify_anchors(rows, family, cells, count):
    anchors = defaultdict(dict)
    for row in rows:
        if row['cell'] not in cells:
            continue
        anchor = row['pair_id']
        if not isinstance(anchor, str) or not anchor or row['cell'] in anchors[anchor]:
            raise ValueError('Missing anchor ID or duplicate anchor within a cell')
        if row['test_family'] != family or row['n_frames'] != int(row['cell'].rsplit('_N', 1)[1]):
            raise ValueError('Prediction family or frame count differs from its cell')
        anchors[anchor][row['cell']] = row
    if len(anchors) != count:
        raise ValueError('Incorrect registered anchor count')
    for paired in anchors.values():
        if set(paired) != set(cells) or len({row['gold'] for row in paired.values()}) != 1:
            raise ValueError('Incomplete anchor coverage or changing gold across extensions')


def summarize(rows):
    parsed = [r for r in rows if r['prediction'] is not None]
    return dict(n=len(rows), correct=sum(r['exact'] for r in rows), exact=mean(r['exact'] for r in rows),
                parse_rate=len(parsed)/len(rows),
                mae_parsed=mean(abs(r['prediction']-r['gold']) for r in parsed) if parsed else None,
                bias_parsed=mean(r['prediction']-r['gold'] for r in parsed) if parsed else None,
                first_token_nll=mean(r['gold_first_token_nll'] for r in rows),
                model_seconds=mean(r['model_seconds'] for r in rows),
                total_seconds=mean(r['model_seconds']+r['preprocessing_seconds'] for r in rows))


def compare(candidate, control, cells, np, rng):
    aa = {(r['cell'], r['sid']): r for r in candidate if r['cell'] in cells}
    bb = {(r['cell'], r['sid']): r for r in control if r['cell'] in cells}
    if not aa or aa.keys() != bb.keys():
        raise ValueError('Mismatched paired examples')
    anchors = defaultdict(dict)
    for key, a in aa.items():
        b = bb[key]
        if (a['path'], a['gold'], a['pair_id']) != (b['path'], b['gold'], b['pair_id']):
            raise ValueError('Paired identity mismatch')
        if a['cell'] in anchors[a['pair_id']]:
            raise ValueError('Duplicate anchor/cell in paired comparison')
        anchors[a['pair_id']][a['cell']] = int(a['exact'])-int(b['exact'])
    if any(set(values) != set(cells) for values in anchors.values()):
        raise ValueError('Incomplete anchor coverage')
    values = np.array([mean(anchors[k].values()) for k in sorted(anchors)])
    bootstrap = values[rng.integers(0, len(values), size=(10000,len(values)))].mean(axis=1)
    return dict(n=len(aa), anchors=len(values), exact_gain=float(values.mean()),
                bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
                candidate_only=sum(a['exact'] and not bb[k]['exact'] for k,a in aa.items()),
                control_only=sum(bb[k]['exact'] and not a['exact'] for k,a in aa.items()))


def extension(rows, family, low, high):
    aa={r['pair_id']:r for r in rows if r['cell']==f'{family}_N{low}'}
    bb={r['pair_id']:r for r in rows if r['cell']==f'{family}_N{high}'}
    if aa.keys()!=bb.keys() or not aa:
        raise ValueError('Extension pairs incomplete')
    parsed=[k for k in aa if aa[k]['prediction'] is not None and bb[k]['prediction'] is not None]
    return dict(n=len(aa), both_parsed=len(parsed),
                same_parsed_prediction=sum(aa[k]['prediction']==bb[k]['prediction'] for k in parsed),
                same_parsed_prediction_all_denominator=sum(aa[k]['prediction']==bb[k]['prediction'] for k in parsed)/len(aa),
                both_correct=sum(aa[k]['exact'] and bb[k]['exact'] for k in aa),
                correct_to_incorrect=sum(aa[k]['exact'] and not bb[k]['exact'] for k in aa),
                incorrect_to_correct=sum(not aa[k]['exact'] and bb[k]['exact'] for k in aa),
                mean_prediction_change_parsed=mean(bb[k]['prediction']-aa[k]['prediction'] for k in parsed) if parsed else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/native_aggregation_vlm/v2'))
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run analysis on CPU Slurm')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoTokenizer

    expected_hashes=json.loads((a.root/'main/source_hashes.json').read_text())
    predictions,results,runs={},{},{}
    common_hashes=set()
    for arm in ARMS:
        candidates=list((a.root/'main'/arm).glob('*/summary.json'))
        if len(candidates)!=1:
            raise ValueError(f'Need exactly one completed run for {arm}: {candidates}')
        run=candidates[0].parent
        summary=json.loads(candidates[0].read_text())
        config=json.loads((run/'config.json').read_text())
        rows=json.loads((run/'predictions.json').read_text())
        manifest=json.loads((run/'data_manifest.json').read_text())
        if config['code_sha256']!=expected_hashes:
            raise ValueError('Training code provenance mismatch')
        for name,sha in expected_hashes.items():
            if hashlib.sha256((run/'code'/name.replace('/','_')).read_bytes()).hexdigest()!=sha:
                raise ValueError('Source snapshot changed')
        if config['arm']!=arm or config['seed']!=0 or config['epochs']!=9 or len(summary['training'])!=9:
            raise ValueError('Registered training configuration mismatch')
        if config['generation_policy']!=dict(do_sample=False,repetition_penalty=1.0,max_new_tokens=4,output_logits=True):
            raise ValueError('Generation policy differs')
        verify_manifests(run, config, manifest)
        common_hashes.add((config['manifest_sha256'],config['count_manifest_sha256']))
        expected={(cell,r['sid']):(r['path'],r['gold'],r['pair_id']) for cell in CELLS for r in manifest[cell]['samples']}
        actual={(r['cell'],r['sid']):(r['path'],r['gold'],r['pair_id']) for r in rows}
        if len(rows)!=452 or len(actual)!=452 or actual!=expected:
            raise ValueError('Incomplete or mismatched held-out predictions')
        for r in rows:
            if r['mode']!='all' or r['tag']!='test' or r['exact']!=(r['prediction']==r['gold']):
                raise ValueError('Invalid prediction bookkeeping')
        verify_anchors(rows, 'length', CELLS[:3], 108)
        verify_anchors(rows, 'unseen_count', CELLS[3:], 64)
        if [e['epoch'] for e in summary['training']] != list(range(1, 10)):
            raise ValueError('Training epochs are incomplete or reordered')
        if any(e['n'] != 180 or e['step'] != 45 * e['epoch']
               or {m['cell']: m['n'] for m in e['dev']} != {'dev_N8': 36, 'dev_N16': 36}
               or len(e['dev']) != 2 for e in summary['training']):
            raise ValueError('Registered training/development coverage differs')
        best=max(summary['training'],key=lambda e:(sum(m['correct'] for m in e['dev'])/sum(m['n'] for m in e['dev']),
                 -sum(m['gold_first_token_nll']*m['n'] for m in e['dev'])/sum(m['n'] for m in e['dev'])))
        checkpoint = Path(summary['selected_checkpoint']).resolve()
        if not checkpoint.is_relative_to(CHECKPOINT_ROOT) or checkpoint.name != 'best.pt':
            raise ValueError('Selected checkpoint is outside the registered checkpoint root')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (saved['epoch'] != best['epoch'] or saved['step'] != best['step']
                or saved['dev'] != best['dev'] or saved['architecture'] != config['architecture']
                or saved['config']['run_id'] != config['run_id']
                or saved['config']['code_sha256'] != expected_hashes
                or saved['config']['manifest_sha256'] != config['manifest_sha256']
                or saved['config']['count_manifest_sha256'] != config['count_manifest_sha256']):
            raise ValueError('Selected checkpoint metadata does not match the dev-selected run')
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        del saved
        info=dict(parameters=summary['parameters'],selected_epoch=best['epoch'],selected_checkpoint=summary['selected_checkpoint'],
                  selected_checkpoint_sha256=checkpoint_sha256,
                  training=summary['training'],cells={c:summarize([r for r in rows if r['cell']==c]) for c in CELLS},
                  familiar_ood=summarize([r for r in rows if r['cell'] in ('length_N32','length_N64')]),
                  unseen_count=summarize([r for r in rows if r['cell'].startswith('unseen_count')]),
                  per_k={f'{c}_K{k}':summarize([r for r in rows if r['cell']==c and r['gold']==k])
                         for c in CELLS for k in sorted({r['gold'] for r in rows if r['cell']==c})},
                  count_answer_support={name:summarize([r for r in rows if r['cell'].startswith('unseen_count') and predicate(r)])
                      for name,predicate in [('K9',lambda r:r['gold']==9),('K10_16',lambda r:r['gold']>=10)]},
                  extensions={f'{family}_{lo}_to_{hi}':extension(rows,family,lo,hi)
                      for family,lo,hi in [('length',16,32),('length',32,64),('length',16,64),('unseen_count',32,64)]})
        # Verify published cell aggregates against raw predictions.
        for m in summary['results']:
            if info['cells'][m['cell']]['correct']!=m['correct'] or info['cells'][m['cell']]['n']!=m['n']:
                raise ValueError('Summary differs from saved predictions')
        predictions[arm],results[arm],runs[arm]=rows,info,str(run)
    if len(common_hashes)!=1:
        raise ValueError('Arms used different data')
    comparisons={}
    for control in ARMS[1:]:
        comparisons[control]={name:compare(predictions['global'],predictions[control],cells,np,np.random.default_rng(20260910))
            for name,cells in [('familiar_ood',('length_N32','length_N64')),('N16',('length_N16',)),
                               ('unseen_count',('unseen_count_N32','unseen_count_N64'))]}
    criteria=dict(V2_1=bool(comparisons['hidden']['familiar_ood']['exact_gain']>=.05-1e-12
                          and comparisons['hidden']['N16']['exact_gain']>=-.05-1e-12),
                  V2_2=bool(all(comparisons[c]['familiar_ood']['exact_gain']>=.05-1e-12 for c in ('middle_lora','lora'))))
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',local_files_only=True)
    tokenization={str(k):tokenizer(str(k),add_special_tokens=False).input_ids for k in range(17)}
    notes=['One training seed; bootstrap resamples paired test anchors and does not quantify training variability.',
           'Clean V2 changes the generator for every arm; V1-to-V2 accuracy changes are not a controlled comparison.',
           'Hidden and global approximately match parameters but differ in nonlinear width/input variance; this is not a pure input-only ablation.',
           'Middle LoRA and adapters use their respective preregistered learning rates; optimization policy remains a comparison factor.',
           'K9..16 changes answer support; K10..16 are multi-digit. First-token NLL is not whole-answer likelihood.',
           'Global reconstructs an existing attention read redundantly. Timing under concurrent jobs is descriptive.',
           'These runs use short direct answers. Reasoning-token composition is untested.']
    analysis=dict(criteria=criteria,arms=results,comparisons_global_minus_control=comparisons,runs=runs,
                  answer_tokenization=tokenization,notes=notes,analysis_job=os.environ['SLURM_JOB_ID'])
    (a.root/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    lines=['# Clean MMReD Vision attribution: V2','',
           '| Arm | N16 K0–8 | N32 K0–8 | N64 K0–8 | Familiar OOD | N32 K9–16 | N64 K9–16 | Selected epoch |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        c=results[arm]
        values=[c['cells'][x]['exact'] for x in CELLS[:3]]+[c['familiar_ood']['exact']]+[c['cells'][x]['exact'] for x in CELLS[3:]]
        lines.append('| '+LABELS[arm]+' | '+' | '.join(f'{100*v:.1f}%' for v in values)+f" | {c['selected_epoch']} |")
    lines+=['','Each familiar cell has108 examples; each unseen-count cell has64. Lengths within each family share anchors.',
            '',f"Registered screens: V2.1={criteria['V2_1']}; V2.2={criteria['V2_2']}.",'']
    for control in ARMS[1:]:
        x=comparisons[control]['familiar_ood'];lo,hi=x['bootstrap95']
        lines.append(f"- Global minus {LABELS[control]}: {100*x['exact_gain']:+.1f}pp; paired-anchor 95% interval [{100*lo:+.1f}, {100*hi:+.1f}]pp.")
    lines+=['','## Interpretation limits','']+['- '+n for n in notes]+['']
    (a.root/'REPORT.md').write_text('\n'.join(lines))
    colors=dict(zip(ARMS,['#0072B2','#D55E00','#009E73','#333333']))
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    for arm in ARMS:
        c=results[arm]
        axes[0,0].plot([16,32,64],[c['cells'][f'length_N{n}']['exact'] for n in [16,32,64]],marker='o',label=LABELS[arm],color=colors[arm])
        axes[0,1].plot([32,64],[c['cells'][f'unseen_count_N{n}']['exact'] for n in [32,64]],marker='o',color=colors[arm])
        epochs=c['training']
        axes[1,0].plot([e['epoch'] for e in epochs],[sum(m['correct'] for m in e['dev'])/sum(m['n'] for m in e['dev']) for e in epochs],color=colors[arm])
        axes[1,1].plot(range(9,17),[mean(c['per_k'][f'unseen_count_N{n}_K{k}']['exact'] for n in [32,64]) for k in range(9,17)],marker='.',color=colors[arm])
    axes[0,0].set(title='Distractor extensions · familiar counts',xlabel='Frames',ylabel='Exact answer',xticks=[16,32,64])
    axes[0,1].set(title='Unseen counts 9–16',xlabel='Frames',ylabel='Exact answer',xticks=[32,64])
    axes[1,0].set(title='In-range development',xlabel='Epoch',ylabel='Exact answer')
    axes[1,1].set(title='Unseen count support · pooled N32/64',xlabel='Gold count',ylabel='Exact answer',xticks=list(range(9,17)))
    for ax in axes.flat:
        ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False)
    fig.suptitle('Native MMReD Vision · clean generator · seed 0',y=.945)
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(a.root/'comparison.png',dpi=170);fig.savefig(a.root/'comparison.pdf')
    print('\n'.join(lines[:18]),flush=True)


if __name__=='__main__':
    main()
