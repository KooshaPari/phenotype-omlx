# OMLX TurboQuant SSOT gate

Write `/app/turbo_ssot_ok.txt` containing exactly:

```
turbo-qwen35-ssot-ok
```

Live TurboQuant+ MLX (Metal) cannot run inside apple-container. Full host
proof remains `scripts/phenotype_omlx_ready.py` check 12 / `perf_turboquant.py`
against Qwen3.5 SSOT. This Harbor task gates operator JobConfigs on that policy.
