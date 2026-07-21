# L6 RLVR-AF smoke

**Verdict:** `PASS` · mean_reward=0.5 · passed=2/5

| task | passed | reward | reason |
|------|--------|--------|--------|
| schema-json-ok | True | 1.0 | expected string found |
| needle-exact | True | 1.0 | expected string found |
| empty-fail | False | 0.0 | empty or error completion |
| error-prefix-fail | False | 0.0 | empty or error completion |
| partial-credit | False | 0.5 | expected 'green' not found in 'Red and blue are important colors in the palette.' |
