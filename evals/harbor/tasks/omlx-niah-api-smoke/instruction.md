# OMLX NIAH API smoke (Qwen3.5)

Call the OpenAI-compatible chat completions endpoint at `OPENAI_BASE_URL`
with the needle-in-haystack prompt from the oracle script.

Write:
- `/app/niah_answer.txt` — model reply (must contain the secret code `42-alpha`)
- `/app/niah_result.json` — structured smoke result

`OPENAI_BASE_URL` is required. Model must be Qwen3.5 (SSOT).
