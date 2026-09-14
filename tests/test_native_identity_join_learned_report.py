"""Independent decisions, denominator, paired-bootstrap and accounting fixtures."""
import copy
import os
from pathlib import Path
import sys
import unittest
if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION')!='cpu' or os.environ.get('SLURM_JOB_GPUS'):
    raise SystemExit('Numerical tests require CPU Slurm')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import report_native_identity_join_learned as report


class Tokenizer:
    all_special_ids=[100,151645,151643]
    table={1:'Mary',2:'Sand',3:'ra',4:'John',5:' ',6:' extra',7:'mary',8:'Ｍary',100:'<bos>',151645:'<eos>',151643:'<end>'}
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(self.table[i] for i in ids if not skip_special_tokens or i not in self.all_special_ids)


class ReporterTests(unittest.TestCase):
    def test_absolute_and_relative_screens_are_distinct(self):
        counts={(m,s):{c:108 for c in report.CELLS} for m in report.MODES for s in report.SEEDS}
        tied=report.criteria(counts);self.assertTrue(tied['absolute_accuracy_target']);self.assertFalse(tied['primary']);self.assertFalse(tied['practical'])
        for (m,s),row in counts.items():
            if m!='clip':
                for c in report.CELLS:row[c]=102 if c.endswith('64') else 108
        self.assertTrue(report.criteria(counts)['practical'])
        counts[('sigmoid',23)]['test_held_N64']=103
        self.assertFalse(report.criteria(counts)['primary'])

    def test_n32_loss_exact_integer_boundary(self):
        counts={(m,s):{c:100 if m=='clip' else 94 for c in report.CELLS} for m in report.MODES for s in report.SEEDS}
        counts[('softmax',22)]['test_seen_N32']=103;self.assertTrue(report.criteria(counts)['primary'])
        counts[('softmax',22)]['test_seen_N32']=104;self.assertFalse(report.criteria(counts)['primary'])

    def test_whole_native_name_and_termination(self):
        tok=Tokenizer()
        for ids in ([1,151645],[1,151643],[5,1,5,151645]):self.assertTrue(report.score(tok,list(ids),'Mary')['exact'])
        for ids in ([2,151645],[1,6,151645],[100,1,151645],[7,151645],[8,151645],[5,5,5,1]):self.assertFalse(report.score(tok,list(ids),'Mary')['exact'])
        for ids in ([1],[1,151645,1,151645]):
            with self.assertRaises(ValueError):report.score(tok,list(ids),'Mary')

    @staticmethod
    def rows():
        result=[]
        for regime in ('seen','held'):
            for room in range(3):
                for family in range(12):
                    for variant in range(3):
                        for n in (32,64):
                            result.append(dict(cell=f'test_{regime}_N{n}',regime=regime,room_pair=[str(room),str(room+3)],
                                contrast_id=f'{regime}/{room}/{family}',pair_id=f'{regime}/{room}/{family}/{variant}',variant=variant,
                                n_frames=n,gold=report.NAMES[(family*3+variant)%9],exact=family%2==0,completed=True,truncated=False,
                                parseable=True,parsed_name_correct=family%2==0))
        return result

    def test_family_denominators_and_shared_bootstrap_draws(self):
        rows=self.rows();stats=report.summarize(rows)
        self.assertEqual(stats['cells']['test_seen_N32']['all_three_correct'],18)
        all_rows={(m,s):copy.deepcopy(rows) for m in report.MODES for s in report.SEEDS}
        boot=report.bootstrap(all_rows)
        for interval in boot['intervals']:
            if interval['kind'].startswith('clip_gain'):self.assertEqual((interval['lower'],interval['upper']),(0.,0.))
        with self.assertRaises(ValueError):report.summarize(rows[:-1])

    def test_failed_zero_accounting_and_double_count_protection(self):
        name='identity_join_learned_feature_profile'
        raw=f'1|{name}|gpu|FAILED|1:0|7|gres/gpu=1,gres/gpu:b200=1|2026-09-11T10:00:00|2026-09-11T10:00:07'
        zero=f'2|{name}|gpu|CANCELLED|0:0|0||Unknown|Unknown'
        value=report.accounting_rows(raw+'\n'+zero);self.assertEqual(value['allocated_gpu_seconds'],7);self.assertEqual(len(value['rows']),2)
        for bad in (raw+'\n'+raw,raw.replace('|7|','|301|'),raw.replace('|FAILED|','|RUNNING|'),raw.replace('gpu:b200=1','gpu:b200=2')):
            with self.assertRaises(ValueError):report.accounting_rows(bad)
        overlap='\n'.join(raw.replace('1|',str(i)+'|',1).replace(name,'identity_join_learned_main') for i in range(1,6))
        with self.assertRaises(ValueError):report.accounting_rows(overlap)


if __name__=='__main__':unittest.main()
