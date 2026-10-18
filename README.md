# B1 vs. B2 FastV: Visual Token Pruning in LLaVA-Med for Medical VQA

> **Research status:** Completed B1 at 100%, 75%, 50%, and 25% visual-token retention and B2 FastV at 75%, 50%, and 25%. These are **experimental measurements on SLAKE**, not the numbers reported by the FastV authors. Primary accuracy scoring is matched between methods; **end-to-end timing boundaries require further harmonization** before any causal speedup claim.

## Overview

This repository documents two visual-token pruning experiments using frozen **Microsoft LLaVA-Med v1.5 Mistral-7B** on the English portion of the **SLAKE 1.0 official test split**. The goal is to evaluate how visual-token retention affects medical visual question answering (VQA), language-model prefill latency, full-response latency, generated answer length, and allocated GPU memory.

- **B1 — Baseline visual-token pruning:** existing earlier-stage pruning implementation; includes a **100% (unpruned) reference** plus 75%, 50%, and 25% retention.
- **B2 — FastV:** attention-guided visual-token ranking **after two complete decoder layers**, then physical removal of lower-ranked visual tokens in layers 2–31 (zero-based indexing). Implemented with frozen weights and heterogeneous KV caches.

Retention is the **percentage kept**; the FastV paper's *R* is the **percentage removed**. Thus, this project's 25% retention equals *R = 75%* in the paper.

### Highlights

- **1,061 official SLAKE English test questions**: 416 CLOSED and 645 OPEN; **no training or fine-tuning**.
- **576 visual tokens per image** (336 × 336 CLIP image input); 432 / 288 / 144 retained at 75% / 50% / 25%.
- **3,183 B2 predictions** (1,061 questions × 3 retentions); **4,244 B1 predictions** including the 100% reference.
- B2 25% retention: **46.63% CLOSED accuracy** vs. **47.60%** unpruned B1 (difference -0.96 percentage points).
- B2 25% retention: mean prefill **181.40 ms**, a **64.9% reduction in recorded prefill latency** against B1 100%; however, **total inference latency is not faster** in the current measurements.

## Model, data, and experimental environment

| Setting | Value |
|---|---|
| Model | `microsoft/llava-med-v1.5-mistral-7b` |
| Model checkpoint revision | `91bb16c122001ddc9cf1fd36ce1dae09448943a2` |
| Source repository | [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med) |
| Source code revision used for B2 | `30697ca50b5c29a8e955c99330b259776aef27b9` |
| Trainable parameters | 0 (frozen inference) |
| Dataset | [SLAKE 1.0](https://www.med-vqa.com/slake/) official `test.json`, English `q_lang == "en"` |
| Official test split (all languages) | 2,094 questions |
| Evaluation subset | 1,061 English questions; 416 CLOSED, 645 OPEN |
| Visual input | CLIP features from 336 × 336 images, 576 visual patch tokens |
| Decoder | 32-layer Mistral; hidden size 4,096; 32 attention heads; 8 KV heads |
| GPUs | **2 × NVIDIA Tesla T4**, ~14.56 GiB usable per GPU |
| B2 placement | GPU 0: CLIP/projector and decoder layers 0–15; GPU 1: layers 16–31, norm, LM head |
| Inference precision | FP16 weights/embeddings |
| Generation | Greedy decoding; maximum **64 new tokens**; stop on EOS |
| Software for B2 | Python 3.10.21, PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, Transformers 4.36.2 |

**Scope note:** This is a single-dataset extension/implementation study. It does not reproduce the original FastV paper's exact model architecture, benchmark mixture, batch regime, hardware or FLOPs analysis.

## Pruning strategies

### B1 — Earlier-stage baseline

B1 serves as the earlier-pruning reference and unpruned control (100%). It uses the same 576-token visual stream and the same nominal retention budgets as B2. In our experimental workflow, B1 prunes earlier than B2, around the first decoder-layer stage. For exact insertion semantics and ranking/selection details, consult the original B1 notebook rather than inferring them from the summary files:

- [B1 benchmark notebook](https://github.com/mashfiqrizvee/Pruning_/blob/main/baselines/b1_results/b1-benchmark-final.ipynb)

**Important:** Since the methods use different pruning depths, a direct B1/B2 comparison changes both **selection policy** and **pruning depth**. It is a comparison of configured systems, not an isolated ablation of attention-based token importance.

### B2 — Post-layer-2 FastV

The B2 configuration uses `K = 2` completed full-sequence decoder layers:

```text
Image (336x336)
  -> CLIP vision encoder -> projection -> 576 visual embeddings
  -> multimodal text + image sequence
  -> Mistral decoder layer 0: FULL sequence
  -> Mistral decoder layer 1: FULL sequence, collect attention
  -> rank visual keys by mean attention received
  -> keep top-k visual tokens in ORIGINAL sequence order
  -> Mistral decoder layers 2..31: REDUCED sequence
  -> norm -> LM head -> greedy autoregressive generation
```

For attention tensor `A` with shape `[batch, heads, query_positions, key_positions]`, B2 scores each visual-token key by its mean **received attention** across heads and query positions, selects the highest-ranked visual keys, and preserves **all text tokens**. The selected tokens retain their **original absolute RoPE position IDs**. B2 physically gathers the retained hidden states and causal mask, rather than merely suppressing attention to tokens.

**Heterogeneous KV cache:** During prefill, decoder layers 0–1 retain the entire original input in KV, while layers 2–31 retain the shorter sequence. Every subsequent generated token is appended to each layer's own cache with the unpruned absolute token position.

| B2 retention | FastV paper removal ratio *R* | Visual tokens kept | Visual tokens removed | Full decoder layers | Reduced decoder layers |
|---|---:|---:|---:|---|---|
| 75% | 25% | 432 | 144 | 0–1 | 2–31 |
| 50% | 50% | 288 | 288 | 0–1 | 2–31 |
| 25% | 75% | 144 | 432 | 0–1 | 2–31 |

## Evaluation protocol

1. Read official SLAKE `test.json`, keep records where `q_lang` is English, and preserve question IDs (`qid`) and reference answers.
2. Verify the **1,061 unique questions**, with **416 CLOSED** and **645 OPEN** answers. Check that each referenced image exists.
3. Generate one answer per `qid × retention` pair. B2's locked expanded manifest contains exactly **3,183 unique pairs**.
4. Use the same deterministic prompts and greedy generation parameters for the B2 retention settings. The real-model test uses the LLaVA-Med `mistral_instruct` conversation template.
5. Score B1 and B2 using **B1 Cell 11B's rule set**, implemented for B2 in `b2_metrics.py`.
6. Report question-level predictions, accuracy and token overlap metrics, latency, answer lengths, and PyTorch peak allocated memory. Do **not** interpret GPU memory as electrical power or FLOPs.

### Exact answer scoring

- **Normalization:** lowercase, trim, replace hyphens with spaces, remove punctuation using `[\W]`-equivalent behavior preserving word characters/whitespace, then collapse whitespace.
- **CLOSED yes/no:** if the gold reference is `yes`/`no`, extract a leading `yes` or `no` from the normalized model answer.
- **Other CLOSED:** **strict normalized exact match**; a full sentence that merely *contains* the expected word is not scored correct.
- **OPEN:** word-multiset overlap (`Counter` intersection) to compute token precision, recall, and F1. Aggregate scores are macro means over the **645 OPEN questions**.

For example, reference `CT` versus prediction `The image is taken using a computed tomography (CT) scan.` is considered **incorrect** for CLOSED exact matching in both methods, although a human could consider the answer semantically correct. This limitation is deliberate for B1/B2 metric compatibility. No LLM judge or supplementary semantic labeling is part of the primary results below.

## Full experimental results

The following entries were calculated from the attached/archived **B1 per-retention summaries and prediction JSONLs** and the **B2 performance summary**. B1 OPEN precision was computed from the OPEN per-question `scoring.open_precision` values. This section preserves measured values rather than filling gaps with estimates.

| Metric | B1 100% | B1 75% | B2 75% | B1 50% | B2 50% | B1 25% | B2 25% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Retained visual tokens | 576 | 432 | 432 | 288 | 288 | 144 | 144 |
| Removed visual tokens | 0 | 144 | 144 | 288 | 288 | 432 | 432 |
| CLOSED accuracy (%) | 47.60 | 46.63 | 47.36 | 46.88 | 46.88 | 44.95 | 46.63 |
| OPEN precision (%) | 4.29 | 4.28 | 4.32 | 4.25 | 4.34 | 4.17 | 4.47 |
| OPEN recall (%) | 39.07 | 39.37 | 38.86 | 39.25 | 39.30 | 38.38 | 39.51 |
| OPEN F1 (%) | 7.44 | 7.44 | 7.48 | 7.37 | 7.52 | 7.24 | 7.75 |
| Mean LLM prefill (ms) | 517.06 | 396.78 | 425.68 | 312.54 | 315.23 | 159.41 | 181.40 |
| Median LLM prefill (ms) | 520.10 | 397.93 | 426.52 | 312.45 | 317.09 | 159.12 | 182.92 |
| Mean reported total inference (ms) | 2117.37 | 1978.20 | 2441.15 | 1857.35 | 2336.49 | 1740.83 | 2241.93 |
| Mean reported vision encoding (ms)* | 26.70 | 26.86 | 50.37 | 27.17 | 50.37 | 26.77 | 50.37 |
| Mean generated tokens | 25.66 | 25.77 | 24.58 | 25.51 | 24.66 | 25.87 | 24.58 |
| Total generated tokens | 27228 | 27340 | 26075 | 27071 | 26164 | 27449 | 26077 |
| Mean summed peak GPU allocation (GiB)** | 14.930 | 14.845 | 14.831 | 14.769 | 14.782 | 14.708 | 14.756 |
| Generation limit reached (count) | 54 | 52 | 53 | 56 | 59 | 59 | 61 |


\* **Vision-timing caveat:** B1's `vision_encoding_ms` isolates its named vision-encoding stage, whereas the B2 field `image_encoding_ms` includes image loading/preprocessing and multimodal-embedding preparation. These are **not identical measurement boundaries**. Likewise, reported `total_inference_ms` is implementation-specific; see [Performance interpretation](#performance-interpretation-and-limitations).

\*\* **Memory caveat:** B1 reports `mean_peak_memory_sum_gib`; B2 reports per-GPU peak allocation. The B2 sum shown above adds **means of per-device peak allocations** and does **not** represent a synchronized system-wide peak.

### Difference between B2 and B1 at equal retention

| B2 minus B1 | 75% | 50% | 25% |
| --- | --- | --- | --- |
| CLOSED accuracy (pp) | +0.72 | +0.00 | +1.68 |
| OPEN recall (pp) | -0.51 | +0.05 | +1.13 |
| OPEN F1 (pp) | +0.04 | +0.15 | +0.51 |
| Prefill time change (%) | +7.29 | +0.86 | +13.80 |
| Reported total time change (%) | +23.40 | +25.80 | +28.79 |


For accuracy, positive numbers mean B2 performs better. For latency, positive numbers mean B2 took longer. `pp` means percentage points; relative time changes use `(B2/B1 - 1) × 100`.

### Key observations

1. **Accuracy preservation:** At **25% retention**, B2 CLOSED accuracy is **46.63%** vs. B1's unpruned **47.60%**, a difference of only **0.96 percentage points** while removing 75% of the visual tokens after layer 2.
2. **At equal retention:** B2 CLOSED accuracy is higher at 75% and 25%, tied at 50%. B2 OPEN F1 is higher at all three budgets, but some differences are small and have **not been tested for statistical significance**.
3. **Within-B2 prefill:** Mean prefill latency decreases from **425.68 ms (75%)** to **181.40 ms (25%)**, a **57.4%** reduction.
4. **Against unpruned B1:** B2 at 25% shows **64.9%** lower recorded mean prefill latency, but **5.9% higher** reported mean total inference latency. **Do not claim end-to-end acceleration** from these results.
5. **Memory:** At equal retention, the observed B1/B2 differences are on the order of hundredths of a GiB; B2 does not consistently use less memory than B1. Earlier pruning can reduce some layer-specific KV requirements, but parameter storage dominates a frozen 7B model's memory footprint.

## Real-model verification and correctness checks

The B2 implementation was validated progressively before the full experiment:

| Cell / phase | Validation | Result |
|---|---|---|
| 1–5 | Dataset/protocol, environment, checkpoint integrity, 2-GPU placement, model architecture | PASS |
| 6 | Attention-dependent token ranking, keep counts, all text preserved, causal mask gather | PASS |
| 7 | 4-layer Mistral; unpruned custom/native logits identical; pruned layer lengths and original RoPE positions | PASS |
| 8 | Native cached decoding equivalence (8 steps); heterogeneous KV growth at 75/50/25% | PASS |
| 9 | Frozen real 7B checkpoint; one SLAKE image; native/custom unpruned prefill and decode equivalence; three pruning levels | PASS |
| 10 | Full-answer greedy generation for sample `qid=11934`; 8 native-identity steps; EOS termination | PASS |
| 11 | Official 1,061-question manifest, 3,183-prediction manifest, B1 scorer unit tests | PASS |
| 12–13 | Six-prediction pilot, score and cache audits, immutable checkpoint after resume | PASS |
| 14A | Warm-up patch and syntax/resume checks | PASS (warm-up path enabled for full run) |
| 15–17 | Full inference and post-run results produced | Complete; archived B2 summary reports `PASS` |

Illustrative real-model validation on SLAKE QID `11934` (reference `CT`): B2 75% and 50% generated `The image is taken using a computed tomography (CT) scan.`; at 25% it generated `The image is taken using a chest computed tomography (CT) scan.` The scoring rule still marks each as incorrect for exact categorical `CT` matching, consistently with B1.

## Performance interpretation and limitations

**The latency measurements should not be over-interpreted.** B2 calls decoder layers individually, captures full attention at the selection layer, handles heterogeneous per-layer caches and performs token selection/gather. B1 uses its own inference path. As a result:

- The measurements are actual recorded execution timings, but the two methods' `total_inference_ms` fields have **not been demonstrated to share fully identical instrumentation boundaries**.
- The **B2 vision-preparation timing** is broader than B1's isolated vision-encoder timing, so the corresponding means cannot be compared as if they were the same operation.
- Answer lengths differ between methods and naturally affect autoregressive decoding time. One should normalize or stratify by token counts before assigning a total-time difference to pruning alone.
- The B2 custom decode loop can introduce implementation overhead; the large total-latency gap does **not** isolate the cost of its attention-guided selection.
- Pruning depth differs between B1 and B2; a matched-layer ablation would be needed to isolate the selection strategy.
- GPU memory numbers include persistent **model parameters**. Summed independent GPU peaks are not necessarily simultaneous peaks. Neither electrical **power**, **energy**, nor measured **FLOPs** were collected for either benchmark; none are claimed here.
- Scores are based on a deliberately strict textual scorer. OPEN F1 is low partly because verbose generated answers are compared with short references using token-overlap F1.
- No confidence intervals or paired significance tests are included here. Sub-one-percentage-point differences are descriptive, not proof of statistical improvement.
- Findings are restricted to this model, medical dataset, checkpoint, prompts, and two-T4 runtime.

### Alignment with the FastV paper

Original paper: [Chen et al., *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference*](https://arxiv.org/abs/2403.06764) (2024).

| Paper finding | This study |
|---|---|
| Importance ranking from average received attention | Implemented in B2 after layer 2 |
| Significant reduction of visual-token processing | Verified by physical sequence lengths in later layers |
| Much of task accuracy can survive 50–75% visual-token removal | Consistent with measured SLAKE CLOSED accuracy and OPEN F1 |
| Reduced computational workload / FLOPs in theory | Compatible with shorter later-layer sequences, **but FLOPs were not measured** |
| Reduced wall-clock inference time | **Not established** here; measured B2 total latency exceeds B1 in current runs |
| Lower GPU memory than fully unpruned processing | Small descriptive differences; **not a matched native-B2 unpruned memory benchmark** |

The original paper evaluates other models, datasets and GPU conditions. Its Table 4 latency experiment includes a *K=0* pruning setting on an A40 GPU, whereas B2 uses *K=2* and two T4 GPUs with free-form answers; direct numerical replication is not expected.

## Repository artifacts and data layout

An example layout when all exported files and source scripts are included (paths can be adapted to your GitHub repository):

```text
.
├── README.md
├── baselines/
│   └── b1_results/
│       ├── b1_100pct_predictions.jsonl
│       ├── b1_100pct_summary.json
│       ├── b1_75pct_predictions.jsonl
│       ├── b1_75pct_summary.json
│       ├── b1_50pct_predictions.jsonl
│       ├── b1_50pct_summary.json
│       ├── b1_25pct_predictions.jsonl
│       └── b1_25pct_summary.json
└── B2_FastV/
    ├── fastv_selection.py
    ├── fastv_prefill.py
    ├── fastv_cached.py
    ├── load_b2_model.py
    ├── b2_metrics.py
    ├── configs/
    ├── manifests/
    ├── scripts/
    │   └── b2_fastv_inference_runner.py
    ├── predictions/
    │   └── b2_fastv_full_predictions.jsonl
    └── results/
        ├── b2_performance_summary.csv
        ├── b2_performance_summary.json
        └── b2_question_level_metrics.csv
```

**Note:** This is a suggested organization, **not a claim that every listed file has already been committed to GitHub**. Kaggle's isolated `venv/`, downloaded model shards, image datasets and caches should not be pushed into version control. Keep public uploads consistent with the model/dataset licenses and applicable research data policies.

### Important saved outputs

| Artifact | Contents |
|---|---|
| `b1_{retention}pct_predictions.jsonl` | B1 question-level prediction, scoring, timings and GPU memory |
| `b1_{retention}pct_summary.json` | B1 1,061-question summary at that retention |
| `B2_FastV/manifests/b2_fastv_3183_manifest.jsonl` | Locked question × retention combinations |
| `B2_FastV/predictions/b2_fastv_full_predictions.jsonl` | All 3,183 full-response B2 predictions with token IDs, scoring and profiling |
| `B2_FastV/results/b2_performance_summary.csv` | Three B2 retention summaries, publication-table ready |
| `B2_FastV/results/b2_performance_summary.json` | Machine-readable B2 aggregate metrics |
| `B2_FastV/results/b2_question_level_metrics.csv` | One row per B2 question × retention |
| `B2_FastV/configs/` | Environment, model audit, pilot/resume and profiling configurations |

## Reproducing the B2 inference

This workflow was developed and verified in **Kaggle notebooks**, with model execution delegated to the Python 3.10 environment at `/kaggle/working/B2_FastV/venv/bin/python`. It uses the pinned LLaVA-Med source and checkpoint revisions above.

1. Provision **two T4-class GPUs** or sufficient comparable resources; attach SLAKE and obtain the licensed frozen model checkpoint.
2. Restore the archived `B2_FastV` source, manifests, configs, and scripts. Rebuild the isolated environment if a fresh runtime does not retain `venv/`.
3. Verify the pinned checkpoint and loader, the original model device mapping, and the unit/integration tests before enabling full inference.
4. From the LLaVA-Med source directory, launch the resumable runner:

```bash
cd /kaggle/working/LLaVA-Med
export HF_HOME=/tmp/b2_llava_med_hf_cache
export PYTHONPATH=/kaggle/working/LLaVA-Med:/kaggle/working/B2_FastV
/kaggle/working/B2_FastV/venv/bin/python \
  /kaggle/working/B2_FastV/scripts/b2_fastv_inference_runner.py \
  --mode pilot

# After verifying the six-prediction pilot:
/kaggle/working/B2_FastV/venv/bin/python \
  /kaggle/working/B2_FastV/scripts/b2_fastv_inference_runner.py \
  --mode full
```

The `--mode full` runner checkpoints **each successful prediction** to `predictions/b2_fastv_full_predictions.jsonl`, flushes and `fsync`s it, and skips valid `(qid, retention_pct)` combinations already present when resumed. The validated full-mode runner performs six unrecorded warm-up inferences (two passes over three retention settings) when predictions remain pending. It does **not** train or modify the model.

After completing all 3,183 predictions, run the archived notebook's integrity and metric-summary cells to regenerate the saved files; the summary CSV and JSON are provided for verification.

### Recompute aggregate accuracy from saved B2 question metrics

This short example uses no model checkpoint:

```python
import pandas as pd

df = pd.read_csv('B2_FastV/results/b2_question_level_metrics.csv')
for pct in (75, 50, 25):
    part = df[df['retention_pct'] == pct]
    closed = part[part['answer_type'] == 'closed']
    opened = part[part['answer_type'] == 'open']
    print(pct, {
        'closed_accuracy_pct': 100 * closed['closed_accuracy'].mean(),
        'open_recall_pct': 100 * opened['open_recall'].mean(),
        'open_f1_pct': 100 * opened['open_f1'].mean(),
    })
```

## Recommended follow-up experiments

- **Match pruning depth** between B1 and B2 to isolate token-ranking effects.
- **Match timing boundaries** and record native unpruned generation on the B2 implementation to isolate full-response overhead.
- Report paired **bootstrap confidence intervals** for CLOSED accuracy and OPEN token-overlap scores, using identical `qid` pairs.
- Break down decode time per generated token and investigate latency impact of attention extraction, masking, host synchronization and cache handling.
- Test additional image resolutions, datasets and generation lengths to understand the stability of the accuracy–latency trade-off.

## References

1. Chen, L. et al. (2024). **An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference.** [arXiv:2403.06764](https://arxiv.org/abs/2403.06764). [FastV implementation](https://github.com/pkunlp-icler/FastV).
2. Li, B. et al. / Microsoft. **LLaVA-Med: Large Language and Vision Assistant for BioMedicine.** [Repository](https://github.com/microsoft/LLaVA-Med).
3. Liu, H. et al. **Improved Baselines with Visual Instruction Tuning (LLaVA-1.5).** [arXiv:2310.03744](https://arxiv.org/abs/2310.03744).
4. **SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering.** [Project](https://www.med-vqa.com/slake/).

---

**Reporting guidance:** State clearly whether a percentage describes **tokens retained** (this README) or **tokens pruned** (FastV paper), whether a latency covers only **LLM prefill** or **complete answer generation**, and whether GPU memory represents **per-device peak allocations** or an aggregate. These distinctions are essential for a reproducible comparison.
