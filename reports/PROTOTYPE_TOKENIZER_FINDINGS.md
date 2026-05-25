# Prototype Tokenizer Findings v0.1

Across 12 natural semantic-equivalent conversational prompt groups in Hindi,
Tamil and Hinglish, Sarvam-30B produced fewer raw user-prompt tokens than
Krutrim-2-Instruct and BharatGen Param2 in all three language categories.

The largest raw-token gaps observed were:

- Sarvam vs Krutrim on Tamil: 38.9% fewer tokens
- Sarvam vs BharatGen on Hinglish: 34.2% fewer tokens
- Sarvam vs Krutrim on Hindi: 31.1% fewer tokens

Token-fragment inspection suggests that Sarvam preserves larger native-script
and conversational subword units in selected examples.

These results are preliminary, based on a small prototype prompt set, and do
not establish lower serving latency or higher model quality. GPU experiments
will test whether natural-tokenisation differences affect streaming TTFT.

## Global Multilingual MoE Control: Qwen

Qwen is added as a strong multilingual MoE control with a similar
total-parameter scale to Sarvam.

In the four-model `tokenizer_diagnostic_natural_v2_qwen` run, all tokenizer
and chat-template applications succeeded for Qwen across the frozen 12
semantic-equivalent prompt groups.

Observed Qwen mean raw user-prompt token counts were:

- Hindi: 66.7 tokens
- Tamil: 81.0 tokens
- Hinglish: 22.3 tokens

Observed Qwen mean formatted input token counts were:

- Hindi: 214.5 tokens
- Tamil: 250.2 tokens
- Hinglish: 73.1 tokens

Compared with Qwen, Sarvam-30B produced fewer raw user-prompt tokens and fewer
formatted input tokens in all three prototype language categories. The largest
observed Qwen gaps were in Hindi and Tamil, where Qwen's tokenizer fragmented
native-script text much more heavily in the selected token-fragment examples.
Hinglish was closer, but Sarvam still produced fewer tokens in this prototype
set.

This remains a tokenizer and chat-template observation only. It does not
establish lower inference latency or higher model quality.

## English Reference Control

English was added as a reference-control language to help distinguish whether
the preliminary Sarvam compact-token signal is strongest for native-script
Hindi and Tamil or appears uniformly across languages. English is not a primary
Indic workload for this project; Hindi, Tamil, and Hinglish remain the primary
benchmark targets.

In `tokenizer_diagnostic_natural_v3_english_control`, all four configured
tokenizers and chat templates succeeded across 48 natural prompt records.

Observed English mean raw user-prompt token counts were:

- Sarvam-30B: 14.4 tokens
- Qwen3-30B-A3B-Instruct-2507-FP8: 14.8 tokens
- Krutrim-2-Instruct: 14.8 tokens
- BharatGen Param2: 15.8 tokens

Observed English mean formatted input token counts were:

- Qwen3-30B-A3B-Instruct-2507-FP8: 57.8 tokens
- Sarvam-30B: 60.2 tokens
- Krutrim-2-Instruct: 61.2 tokens
- BharatGen Param2: 62.4 tokens

For Qwen, the mean raw user-prompt token count was much higher on native-script
Hindi and Tamil than on English:

- Hindi: 66.7 raw tokens, about 4.5x Qwen's English raw-token count
- Tamil: 81.0 raw tokens, about 5.5x Qwen's English raw-token count
- Hinglish: 22.3 raw tokens, about 1.5x Qwen's English raw-token count
- English: 14.8 raw tokens

This suggests the large Qwen gap in the prototype is concentrated in
native-script Hindi and Tamil rather than being a general token-count gap across
all languages. This is tokenizer/template evidence only. It does not establish
lower inference latency or higher model quality. GPU experiments are still
required to test latency impact.
