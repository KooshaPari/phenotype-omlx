# OMLX Qwen3.5 acceptance policy

Write `/app/policy_ok.txt` containing exactly:

```
qwen35-ssot-ok
```

This task gates Harbor/Langfuse operator runs on the Phenotype rule that
acceptance models must be **Qwen3.5** (see `config/smoke_models.json`).
The oracle solution stamps the marker; live agent runs should only proceed
when the SSOT defaults contain Qwen3.5 and not Qwen2.5.
