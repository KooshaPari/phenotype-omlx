# OMLX NIAH 32k API smoke (Qwen3.5)

Call the OpenAI-compatible chat completions endpoint at `OPENAI_BASE_URL`
with the needle-in-haystack prompt from the oracle script. The 32k variant
fills the prompt to exactly 32768 tokens (vs. 8192 for the 8k gate) to
exercise Qwen3.5 long-context retrieval.

Write:
- `/app/niah_answer.txt` — model reply (must contain the secret code `42-alpha`)
- `/app/niah_result.json` — structured smoke result

`OPENAI_BASE_URL` is required. Model must be Qwen3.5 (SSOT).
`NIAH_CONTEXT_TOKENS_32K=32768` (set via Infisical + portage.env) drives the
oracle's `build_prompt()` to construct an exact 32768-token prompt.
