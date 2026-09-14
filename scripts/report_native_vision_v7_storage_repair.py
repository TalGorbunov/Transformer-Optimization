"""Explicit V7 report repair: native FP16 logits are archived as exact FP32 copies.

Original frozen reporter, predictions, selection, scoring and tolerance unchanged.
Installed Transformers promotes native logits before returning generation logits.
"""
from pathlib import Path
import inspect
import hashlib
import sys
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from scripts import report_native_vision_v7 as base
ORIGINAL=dict(zip(base.OWN,(
    '69b5129b2b4743cf1f55c1fef057c725efdfdc575c119d9bd40202d267a8ad78',
    '63915ef12a650da60871a6dafb18a87e7f02152c2999e86530ce32f9c9a634d1')))
OWN=('scripts/report_native_vision_v7_storage_repair.py','slurm/native_vision_v7_storage_repair.sbatch')
OLD='x.dtype==torch.float16 and bool(torch.isfinite(x).all())'
NEW='stored_native_fp16_logits(x)'

def stored_native_fp16_logits(x):
    import torch
    return x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and torch.equal(x,x.half().float())

def main():
    base.ledger(ORIGINAL)
    source=inspect.getsource(base.audit_eval)
    base.need(source.count(OLD)==1,'Original storage assertion differs')
    effective=source.replace(OLD,NEW)
    base.stored_native_fp16_logits=stored_native_fp16_logits
    exec(compile(effective,str(Path(__file__).resolve())+'::effective_audit_eval','exec'),base.__dict__)
    base.OWN=tuple(ORIGINAL)+OWN
    old_snapshot=base.snapshot
    repair_record={}
    def snapshot(out):
        hashes=old_snapshot(out)
        path=out/'effective_audit_eval.py';path.write_text(effective)
        generation=Path(sys.prefix)/'lib/python3.10/site-packages/transformers/generation/utils.py'
        promotion='next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)'
        base.need(promotion in generation.read_text(),'Installed generation promotion differs')
        artifacts={}
        for label,p in [('generation_source',generation),('failed_report_log',REPO/'logs/native_v7_report-441893.out'),('schema_inspection_log',REPO/'logs/v7_raw_schema-441904.out')]:
            artifacts[label]=dict(path=str(p),sha256=base.sha(p))
        repair=dict(original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),generation_promotion=promotion,evidence=artifacts,scope='FP32 archival of native FP16 logits, exact roundtrip required',
            original_report_source_sha256=ORIGINAL,repair_source_sha256={p:hashes[p] for p in OWN},
            replaced_assertion=OLD,replacement=NEW,effective_function_file=str(path),effective_function_sha256=base.sha(path),
            original_failed_report=str(base.OUT/'report_441893'),schema_inspection_job=441904,
            no_model_predictions_selection_scoring_threshold_change=True)
        base.save(out/'storage_repair.json',repair)
        repair_record.update(file=str(out/'storage_repair.json'),sha256=base.sha(out/'storage_repair.json'))
        return hashes
    base.snapshot=snapshot
    old_verify=base.verify_main_release
    def verify(config,data,frozen):
        base.need(set(frozen)==set(ORIGINAL)|set(OWN) and {p:frozen[p] for p in ORIGINAL}==ORIGINAL,'Original source ledger differs')
        result=old_verify(config,data,ORIGINAL)
        result['storage_schema_repair']=dict(artifact=dict(repair_record),original_report_source_sha256=ORIGINAL,
            repair_source_sha256={p:frozen[p] for p in OWN},exact_fp16_roundtrip_required=True,
            original_failed_report=str(base.OUT/'report_441893'))
        return result
    base.verify_main_release=verify
    old_test=base.self_test
    def self_test():
        import torch
        tests=old_test()
        good=torch.tensor([0.,1.,-7.5],dtype=torch.float16).float()
        base.need(stored_native_fp16_logits(good),'Lossless promoted logits rejected')
        base.need(not stored_native_fp16_logits(good.half()),'Wrong archive dtype accepted')
        base.need(not stored_native_fp16_logits(good+0.000001),'Non-native FP32 detail accepted')
        base.need(not stored_native_fp16_logits(torch.tensor([float('nan')])),'NaN accepted')
        return tests+['exact_native_fp16_to_fp32_archive_contract']
    base.self_test=self_test
    base.main()

if __name__=='__main__':main()
