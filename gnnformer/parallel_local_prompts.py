"""Fixed set-size-independent global prompt for the V7 MMReD experiment.

The original question is preserved. Local rows separately use the unchanged
canonical one-image prompt. This interface invariant is not an efficacy claim.
"""

def build_set_count_prompt(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    return (
        "You will be shown frames describing steps in a house.\n"
        "Respond with a single non-negative integer (0 is allowed). Output only the integer.\n"
        f"Question: {question}\n"
        "Answer: "
    )
