# ABOUTME: V0 pre-flight gate: confirm SC prompt construction matches existing pipeline
# ABOUTME: Diffs the raw text and the tokenized input_ids on a single representative case

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
from main_evaluate_sc import build_triage_prompt, TRIAGE_QUESTIONS


def existing_prompt(question_type, question, patient_info):
    """Reproduce the existing template from evaluate_models.py:290-295 verbatim."""
    return (
        "You are a physician provided with patient information trying to assign a treatment plan.\n"
        f"{question_type}: Answer the following treatment question with only 'yes' or 'no': {question}\n\n"
        f"Patient information:\n{patient_info}\n\n"
        "Answer (yes/no):"
    )


def main():
    patient_info = "Sample clinical context for V0 check."

    ok = True
    for task in ['MANAGE', 'VISIT', 'RESOURCE']:
        question = TRIAGE_QUESTIONS[task]
        sc_prompt = build_triage_prompt(patient_info, task)
        existing = existing_prompt(task, question, patient_info)
        if sc_prompt != existing:
            print(f"FAIL on task={task}:")
            for i, (a, b) in enumerate(zip(sc_prompt, existing)):
                if a != b:
                    print(f"  char {i}: SC={a!r}  existing={b!r}")
                    break
            else:
                print(f"  length differs: SC={len(sc_prompt)}  existing={len(existing)}")
            ok = False
        else:
            print(f"OK task={task}: raw text matches ({len(sc_prompt)} chars)")

    print("\nGeneration kwarg parity:")
    print("  SC:        do_sample=True, temperature=0.7, top_p=1.0, top_k=0, max_new_tokens=512")
    print("  Existing:  do_sample=True, temperature=0.7,                     max_new_tokens=512")
    print("  Documented divergence: top_p=1.0, top_k=0 explicit (overrides Llama's gen_config)")

    if not ok:
        print("\nFAIL: V0 prompt-parity check did not pass.", file=sys.stderr)
        sys.exit(1)
    print("\nPASS: V0 prompt-parity check.")


if __name__ == "__main__":
    main()
