# SSD Readiness Research

## Local Implementation Evidence

The surviving readiness entrypoint is `scripts/phenotype-omlx-ready`.

- Lines 15-20 derive the repository, perf-core, oMLX, and GUI paths.
- Lines 75-93 implement the SSD check.
- Lines 81-83 add the external SSD checkout to `sys.path` and default
  `SSD_HF_CACHE` to `~/.cache/huggingface/hub` and `SSD_DATASET_DIR` to
  `~/.cache/ssd/datasets`.
- Lines 84-92 import `ssd.config` and `ssd.llm`; only import errors mentioning FlashInfer or
  CUDA are treated as an expected Apple-Silicon unavailability.
- Lines 124-131 aggregate checks and fail if a result does not start with `ok`.

This baseline has no `scripts/readiness_check.py` or `tests/test_readiness.py`. Those files
existed in the later FFI-correctness work but are not present in the recovered checkout.
Consequently, the surviving script couples dependency importability, platform availability,
and dataset configuration rather than reporting them as separate readiness dimensions.

An absorbed-tree variant, `scripts/phenotype_omlx_ready.py`, retains the same SSD defaults at
lines 91-102 and similarly considers only FlashInfer/CUDA import failures expected. It adds
per-check timing and timeout reporting, but still does not independently validate dataset
provenance or distinguish structural data readiness from release benchmark readiness.

## Expected Processed Dataset Layout

The prior FFI-correctness readiness run expected these paths below `SSD_DATASET_DIR`:

```text
humaneval/humaneval_data_10000.jsonl
alpaca/alpaca_data_10000.jsonl
c4/c4_data_10000.jsonl
gsm8k/gsm8k_data_10000.jsonl
ultrafeedback/ultrafeedback_data_10000.jsonl
```

The `10000` suffix denotes the requested processing cap. It must not be interpreted as a
guarantee that a source supplies 10,000 records; HumanEval's test split is smaller.

## Local Corpus Audit

On 2026-07-18, all five expected files above were absent from
`~/.cache/ssd/datasets`. A filename search under `~/.cache` also found no qualifying
`humaneval_data_*.jsonl`, `alpaca_data_*.jsonl`, `c4_data_*.jsonl`,
`gsm8k_data_*.jsonl`, or `ultrafeedback_data_*.jsonl` files.

The existing Hugging Face model cache does not qualify as a processed SSD dataset root.
No local corpus currently satisfies the full release-data gate.

## Upstream Source and Schema

The authoritative SSD repository is `tanishqkumar/ssd`. Its README defines:

- `SSD_HF_CACHE` as the Hugging Face hub directory containing `models--org--name/`;
- `SSD_DATASET_DIR` as the parent containing dataset subdirectories such as `humaneval/`
  and `alpaca/`; and
- `scripts/get_data_from_hf.py --num-samples 10000` as the preparation workflow, writing
  under `$HF_DATASETS_CACHE/processed_datasets`.

The upstream preparation script normalizes every output JSONL row to one key:

```json
{"text": "non-empty prompt text"}
```

Source mappings are:

| Output | Upstream dataset | Split/config | Normalized source field |
|---|---|---|---|
| `gsm8k/` | `openai/gsm8k` | `main`, `train` | `question` |
| `c4/` | `allenai/c4` | `en`, `train` streaming | `text` |
| `ultrafeedback/` | `openbmb/UltraFeedback` | `train` | `instruction` |
| `humaneval/` | `openai/openai_humaneval` | `test` | `prompt` |
| `alpaca/` | `tatsu-lab/alpaca` | `train` | `instruction`, plus non-empty `input` |

The upstream script skips an output merely when its filename already exists. Therefore a
release validator must independently parse rows and verify a pinned manifest, row count,
byte count, and SHA-256; filename presence alone is insufficient provenance.

## Primary Sources

- SSD repository and environment/data instructions:
  <https://github.com/tanishqkumar/ssd>
- SSD dataset preparation implementation:
  <https://github.com/tanishqkumar/ssd/blob/main/scripts/get_data_from_hf.py>
- ICLR 2026 SSD paper and evaluated dataset set:
  <https://openreview.net/pdf/072b017310e96164943846ae3bef1615d178e4d7.pdf>

No datasets were downloaded or generated during this research.
