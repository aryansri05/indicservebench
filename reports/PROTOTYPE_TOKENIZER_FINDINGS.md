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
