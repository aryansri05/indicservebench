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
