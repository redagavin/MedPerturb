# Design: Codebase Review Fixes

## Context

A paranoid code review of the original MedPerturb codebase revealed three issues that need tests:

1. **C2**: No chat template in `_call_model()` and `_extract_binary_answer()` — model and extractor produce unreliable output
2. **I4**: Extractor fallback uses substring match (`"yes" in text`) and defaults to 0 — biased and fragile
3. **I3+C3**: `baseline_analysis.py` doesn't filter conversational data, and `(conversational, 211)` is duplicated in did=2/6, causing alignment corruption in MI analysis

All evaluations (dataset_id 1-9) will be re-run after these fixes.

## Fix C2: Chat Template

**File:** `MedPerturb/code/evaluate_models.py`

### `_call_model()` (HuggingFace path only)

Wrap prompt in chat message format and apply chat template before tokenization:

```python
messages = [{"role": "user", "content": prompt}]
formatted = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
```

Output slicing unchanged (already correct at token level). GPT-4 path unchanged (already uses API message format).

### `_extract_binary_answer()`

Same pattern with extractor tokenizer:

```python
messages = [{"role": "user", "content": prompt}]
formatted = self.extractor_tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = self.extractor_tokenizer(formatted, return_tensors="pt").to(self.device)
```

### Generation parameters

No changes. `max_new_tokens=512` for model, `max_new_tokens=128` for extractor.

## Fix I4: Extractor Fallback

**File:** `MedPerturb/code/evaluate_models.py`

Four-layer extraction chain in `_extract_binary_answer()`:

1. **LLM integer parse**: Parse extractor output as int. If 0 or 1, return it. Extraction method: `"integer_parse"`.
2. **LLM text match**: Word-boundary regex (`\byes\b`, `\bno\b`, `\b1\b`, `\b0\b`) on extractor output. Extraction method: `"extractor_text_match"`.
3. **Regex on original model response**: Same word-boundary regex on the original model response passed in as `response` parameter. Extraction method: `"model_response_regex"`.
4. **Default**: Log warning with sample details. Return 0. Extraction method: `"default"`.

Method signature change: `_extract_binary_answer` needs to return both the binary answer AND the trace data (extractor raw output, extraction method). Options:
- Return a dict `{"answer": int, "extractor_output": str, "extraction_method": str}`
- Or return a tuple `(int, str, str)`

Since `evaluate_triage` currently returns `Dict[str, List[int]]`, it will need to return the trace data too. Callers (`evaluate_baselines.py`, `sanity_check_evaluate.py`) will need to handle the richer return value.

## Intermediate Data Capture

Save non-reconstructable intermediate data for inspection:

- **Raw model response** (per seed)
- **Raw extractor output** (per seed)
- **Extraction method** (per seed)
- **Final binary answer** (per seed)

### Storage

**Baseline evaluation** (`evaluate_baselines.py`): CSV checkpoint unchanged for binary results. Parallel JSON file (`baseline_trace_gpu{gpu_id}.json`) stores the trace data keyed by Index.

**Sanity check** (`sanity_check_evaluate.py`): Trace data included in the existing JSON output alongside binary answers.

### Data shape per sample

```json
{
  "Index": 800,
  "MANAGE": {
    "seeds": [0, 1, 42],
    "model_responses": ["Yes.", "Yes.", "Yes."],
    "extractor_outputs": ["1", "1", "1"],
    "extraction_methods": ["integer_parse", "integer_parse", "integer_parse"],
    "binary_answers": [1, 1, 1]
  },
  "VISIT": { ... },
  "RESOURCE": { ... }
}
```

## Fix I3+C3: Conversational Filter

**File:** `MedPerturb/case_studies/baseline_analysis.py`

Add one line after CSV load in `run_analysis()`:

```python
df = pd.read_csv(dataset_path)
df = df[df['dataset'] != 'conversational']
```

This matches the original MedPerturb case study notebook methodology and eliminates the duplicate `(conversational, 211)` alignment bug.

## Tests

### C2 tests
- `_call_model` formats prompt with chat template (verify formatted string contains chat template markers)
- `_extract_binary_answer` formats extraction prompt with chat template

### I4 tests
- Layer 1: clean "0" → 0, "1" → 1
- Layer 2: extractor outputs "yes" → 1, "no" → 0; "yesterday" does NOT match
- Layer 3: extractor fails, regex on model response "Yes." → 1, "No, I don't recommend." → 0
- Layer 4: both fail → 0 + warning logged
- Edge cases: empty string, out-of-range integer ("2"), mixed content

### I3+C3 tests
- `run_analysis` excludes conversational rows
- Non-conversational data preserved

### Intermediate data tests
- Trace JSON contains expected fields
- Raw model responses and extractor outputs are strings
- Extraction method is one of the four valid values

## Files Modified

- `MedPerturb/code/evaluate_models.py` — chat template + fallback + trace data
- `MedPerturb/code/evaluate_baselines.py` — handle richer return, save trace JSON
- `MedPerturb/code/sanity_check_evaluate.py` — handle richer return, include trace in JSON
- `MedPerturb/case_studies/baseline_analysis.py` — conversational filter
- `MedPerturb/tests/unit/test_evaluate_models.py` — new test file for C2 + I4
- `MedPerturb/tests/unit/test_baseline_analysis.py` — add conversational filter test
