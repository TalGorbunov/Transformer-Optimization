"""Strict typed full-response answer contract for the original MMReD pilot.

Pure stdlib. Callers decode actual generated IDs after removing only a final
native EOS, with skip_special_tokens=False and cleanup disabled. Other special
tokens remain visible and cannot be silently repaired into a JSON answer.
"""
from __future__ import annotations
import json

ROOMS=('Kitchen','Bathroom','Garden','Office','Bedroom','Hallway')
PEOPLE=('Sandra','Mary','John','Daniel','Michael','Nobody')
EOS_IDS=(151645,151643)
MAX_TOKENS=50
VOCAB_SIZE=152064


def _typed(value,atype):
    if atype=='number':return type(value) is int and value>=0
    if atype=='room':return type(value) is str and value in ROOMS
    if atype=='person':return type(value) is str and value in PEOPLE
    raise ValueError('Unknown official answer type: '+str(atype))


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def _constant(value):raise ValueError('Nonfinite JSON constant: '+value)


def parse_answer(text,atype):
    """Accept one complete JSON object, exactly the answer key and declared type."""
    if atype not in ('number','room','person'):raise ValueError('Unknown official answer type')
    if not isinstance(text,str):raise TypeError('Actual decoded text must be a string')
    try:value=json.loads(text,object_pairs_hook=_object,parse_constant=_constant)
    except (ValueError,TypeError,RecursionError):return dict(format_valid=False,value=None,error='invalid_full_json')
    if type(value) is not dict or set(value)!={'answer'}:
        return dict(format_valid=False,value=None,error='exact_answer_object_required')
    answer=value['answer']
    if not _typed(answer,atype):return dict(format_valid=False,value=None,error='declared_type_or_vocabulary_mismatch')
    return dict(format_valid=True,value=answer,error=None)


def score_answer(text,atype,typed_gold,generated_ids):
    """Score an immutable natural trajectory; never repairs/retries generation."""
    if not _typed(typed_gold,atype):raise ValueError('Prepared gold must have the canonical JSON type/value')
    ids=list(generated_ids)
    if not (1<=len(ids)<=MAX_TOKENS and all(type(v) is int and 0<=v<VOCAB_SIZE for v in ids)
            and not any(v in EOS_IDS for v in ids[:-1])):
        raise ValueError('Invalid native greedy output-token history')
    completed=ids[-1] in EOS_IDS
    if not completed and len(ids)!=MAX_TOKENS:raise ValueError('Incomplete trajectory stopped before its fixed token budget')
    parsed=parse_answer(text,atype);semantic_match=parsed['format_valid'] and type(parsed['value']) is type(typed_gold) and parsed['value']==typed_gold
    return dict(**parsed,completed=completed,truncated=not completed,semantic_match=semantic_match,
        correct=completed and semantic_match,output_tokens=len(ids),primary_requires_native_eos=True)


def self_test():
    """Small prospective fixtures; call only inside an authorized CPU Slurm job."""
    valid=[('{"answer":"Kitchen"}','room','Kitchen'),(' { "answer": "Nobody" }\n','person','Nobody'),
           ('{"answer":0}','number',0),('{"answer":128}','number',128),('{"answer":"\\u004aohn"}','person','John')]
    for text,atype,value in valid:
        got=parse_answer(text,atype)
        if got!=dict(format_valid=True,value=value,error=None):raise AssertionError(('valid',text,got))
    invalid=[('{"answer":true}','number'),('{"answer":1.0}','number'),('{"answer":"1"}','number'),
             ('{"answer":-1}','number'),('{"answer":NaN}','number'),('{"answer":Infinity}','number'),
             ('{"answer":"kitchen"}','room'),('{"answer":"Park"}','room'),('{"answer":"Emma"}','person'),
             ('{"answer":"Kitchen","extra":1}','room'),('{"answer":"Kitchen","answer":"Kitchen"}','room'),
             ('[{"answer":"Kitchen"}]','room'),('Answer: Kitchen','room'),('Kitchen','room'),
             ('```json\n{"answer":"Kitchen"}\n```','room'),('{"answer":"Kitchen"} prose','room'),
             ('{"answer":"Kitchen"}{"answer":"Kitchen"}','room'),('<|im_start|>{"answer":"Kitchen"}','room'),
             ('{"Answer":"Kitchen"}','room'),('{"answer":null}','room')]
    for text,atype in invalid:
        if parse_answer(text,atype)['format_valid']:raise AssertionError(('invalid',text))
    for eos in EOS_IDS:
        got=score_answer('{"answer":"Kitchen"}','room','Kitchen',[1,eos])
        if not got['correct'] or got['truncated']:raise AssertionError('Native EOS scoring')
    wrong=score_answer('{"answer":"Garden"}','room','Kitchen',[1,151645])
    truncated=score_answer('{"answer":"Kitchen"}','room','Kitchen',[1]*50)
    if wrong['correct'] or not wrong['format_valid'] or truncated['correct'] or not truncated['semantic_match'] or not truncated['truncated']:
        raise AssertionError('Format/semantic/EOS decisions must remain separate')
    return dict(passed=True,valid_cases=len(valid),invalid_cases=len(invalid),native_eos_cases=2,
                wrong_valid_answer=True,truncated_semantic_match=True,model_calls=0,head_calls=0)
