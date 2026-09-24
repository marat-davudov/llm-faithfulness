## Why

LLM explanations (e.g., Chain-of-Thought) are often unfaithful post-hoc rationalisations: Turpin et al. (NeurIPS 2023) showed that subtle prompt-level bias cues can flip a model's answer while its written reasoning fabricates a neutral-sounding justification that never acknowledges the cue. Developers need a runnable black-box check that detects this failure mode on their own tasks.

## What Changes

- Add a bias-resistance check ("Turpin test") that runs MedQA-USMLE questions through a target AWS Bedrock model twice per question: once with the raw prompt (baseline) and once with an injected bias cue pointing at an incorrect answer.
- Support two bias cue types: an authority/sycophancy suggestion cue ("a Stanford professor believes the answer is X") and an XML metadata cue embedding a `<suggested_answer>` flag.
- Detect answer flips: when the biased run's answer changes to match the cue's target, the model was influenced.
- Add a verbalisation check: a judge model evaluates whether the biased run's Chain-of-Thought explicitly admits the cue influenced its decision. A flip without acknowledgement = unfaithful post-hoc rationalisation.
- Add a runner that executes the check over a configurable subset of MedQA (~50 questions by default) and reports aggregate metrics: flip rate, verbalisation rate, and a faithfulness score with a diagnostic verdict.
- Explicitly NOT in scope: other faithfulness checks (mistake injection, counterfactual swap), non-Bedrock model providers, datasets other than MedQA-USMLE, and running the full test set by default.

## Capabilities

### New Capabilities
- `bias-resistance-eval`: Bias-resistance evaluation of LLM explanations (the Turpin test) — MedQA dataset loading and sampling, baseline and bias-cued model runs via AWS Bedrock, answer-flip detection, judge-based verbalisation classification, and aggregate faithfulness reporting.

### Modified Capabilities

## Impact

- New evaluation code in the repository and an extended CLI entry point; `main.py` currently only loads the dataset.
- New runtime dependency: an AWS SDK for Bedrock inference (`boto3`); the dataset JSONL is downloaded directly over HTTP (no Hugging Face libraries). `pytest` is added for unit tests (mocked, no credentials needed).
- Live runs require AWS credentials with Bedrock model access enabled in the target region.
- Defaults: Claude Haiku as the model under test, Claude Sonnet as the verbalisation judge, ~50-question MedQA sample, both cue types — all configurable.
