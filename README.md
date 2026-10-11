# moe-disk-stream-bench

**Before you spend a weekend wiring MoE expert streaming, predict whether it can work at all.** This ~250-line oracle models sparsity x bandwidth x cache and returns tokens/sec plus a streamable/not verdict. You need it if you plan to run sparse MoE models from disk on consumer hardware.

## Problem

Disk-streaming MoE inference (keeping only shared weights + activated experts in RAM, reading expert slabs from SSD on demand — the approach popularized by engines like [JustVugg/colibri](https://github.com/JustVugg/colibri), which runs a 744B-parameter model in ~9.9 GB RAM) is extremely sensitive to three numbers people usually guess at: activation rate (top-k of N), random-read bandwidth, and cache warmth. Most "poor-man's MoE" attempts die not on compute but on the disk wall — and the interesting B-side is that activation sparsity is measurable *before* building anything: it directly determines whether bandwidth becomes the bottleneck, and can even feed back into model design (choose N, k, and quantization so your target hardware streams).

The overlooked angle: nobody ships a pre-flight calculator. You either build the streaming engine first and discover infeasibility later, or you trust marketing numbers.

## How it works

- `ModelSpec` — experts N, top-k, bytes/expert (param count x quantization width), shared weights, layer count.
- `HardwareSpec` — sequential + random disk bandwidth, RAM, cold-cache penalty, optional measured compute-only tokens/sec.
- `predict` — per token, `k` expert slabs per layer must be read; cold reads use *random* bandwidth with a penalty factor. Batch reuse is modeled birthday-style: expected distinct experts touched per layer among `b` tokens is `N * (1 - (1 - k/N)^b)`, so disk reads amortize sublinearly with batch size. Output: predicted tokens/sec, bottleneck (disk vs compute), streamable verdict at ≥1 tps.
- `crossover_topk` — binary-searches the largest top-k that still streams on cold cache (useful for "how sparse must my model be for this SSD?").
- `sweep` — feasibility across cache warmth and across every top-k from 1..N.

## Quickstart (~2 min to first verdict)

```bash
python self_test.py                    # 29 checks, <1 s
python moe_disk_stream_bench.py demo   # canned NVMe case, ~0.1 s
python moe_disk_stream_bench.py predict \
  --experts 64 --topk 4 --expert-params 20e6 --shared-params 200e6 \
  --bytes-per-param 0.5 --layers 24 \
  --disk-seq 3500 --disk-rand 800 --ram 8 --batch 16
```

Measured demo output (repro: `python moe_disk_stream_bench.py demo`; Python 3.12, Windows; model math is deterministic — hardware numbers are *inputs*, not measurements; as of 2026-10-11):

| Scenario (4-bit, 64 experts, top-4, 24 layers, NVMe 800 MB/s random) | tokens/sec | Verdict |
|---|---|---|
| batch=1, cold cache | 0.87 | not streamable |
| batch=16, cold cache | 1.36 | **streamable** |
| batch=16, 50% warm | 2.71 | streamable |
| crossover: max top-k that streams cold (batch=1) | top-k ≤ 3 | — |

The flip from 0.87 to 1.36 is the expert-reuse effect, not noise — it is deterministic given the model. Confidence: **High (≥80%)** for internal math correctness (closed-form tests in `self_test.py` verify the arithmetic exactly). Confidence **Medium (60–79%)** that predictions track a real engine within 2x — the bandwidth model treats expert reads as fully random and assumes no prefetch overlap [estimated; validate against your runtime]. All absolute tps numbers assume the bandwidth inputs above; measure your own disk with a tool like `fio` first.

## Why not X?

| Alternative | Why this still wins |
|---|---|
| Just build the streaming engine and measure | Days of work to learn a number this tool gives in seconds; infeasible configs are rejected before any engineering |
| Full inference benchmark (llama.cpp `-ts` expert offload) | Requires the actual model weights + runtime + GPU/CPU setup (GBs of download, hours); answers "how fast" but not "why / what would fix it" |
| Analytical I/O math in a spreadsheet | Re-derivable, but easily wrong on batch reuse and cold-vs-random bandwidth; this encodes the model once, tested by closed-form unit checks |
| Roofline/analytic GPU tools | Compute-centric; MoE disk streaming is an I/O-first problem with sparse access patterns they don't model |

## Minimal Fallback (PLAN B)

No Python? The core estimate is one formula you can run in any calculator: `tps ≈ random_MBps / ( layers * k * expert_MB )` for batch=1 cold cache. If even that is too much: the qualitative rule is enough to say no — if `layers * k * expert_MB` exceeds ~1 second of your random-read bandwidth, streaming will feel unusable. And the pragmatic zero-code fallback is OS mmap: let the page cache do the streaming and accept the cold-start penalty; this tool tells you how bad that first token will be.

## Design Trade-offs

Priority order on conflict: **honest verdicts > model simplicity > prediction accuracy > feature count.** Deliberate consequences: (1) we use *random* bandwidth for cold reads even though expert slabs are individually sequential — pessimistic by design, false "go" signals are worse than false "no-go"; (2) no learned/I/O-trace simulation — the model must stay inspectable and testable in one file; (3) compute is optional — when unknown, the tool reports the pure I/O ceiling, clearly bounded.

## Limits & Second-order Effects

- **Bandwidth inputs are guesses until you measure.** A wrong `--disk-rand` flips the verdict; run `fio` with 4 KiB–1 MiB random reads first. All tps figures here assume the stated hardware.
- **The reuse model assumes independent expert routing.** Correlated routing (topics, languages) makes batch reuse *better* than modeled; adversarial routing could be worse. Second-order effect: teams may over-batch to hit the tps line, then discover latency (time-to-first-token) ballooning — the tool optimizes throughput, not interactivity.
- **It predicts, it does not run.** Real engines add prefetch threads, mmap heuristics, and expert caching policies that can beat the cold-cache bound; treat outputs as a lower bound and a no-go filter, not a speed guarantee.
- **Feedback loop risk.** If model designers start tuning sparsity to pass this bench on cheap disks, "streamable" could become a box-ticking metric that encourages brittle, bandwidth-marginal architectures.

## FAQ

**Q: Does this replace benchmarking llama.cpp / a real runtime?**
A: No — it decides *whether to bother*, and explains the levers (k, quantization, batch, warmth) if the answer is "almost."

**Q: Why random bandwidth instead of sequential?**
A: Expert slabs are contiguous (sequential per slab) but the access pattern across slabs is sparse; on consumer SSDs, deep-queue random reads are the realistic worst case. Optimistic modeling hides infeasibility.

**Q: When NOT to use this?**
A: (Socratic check.) Do **not** use it when (1) your experts fit in RAM anyway — mmap already solves you; (2) you have a datacenter NVMe array where nothing is disk-bound — the verdict is trivially yes; (3) you need interactive sub-100 ms latency — streaming MoE at these bandwidths is a throughput game, not a latency game; (4) you want to model expert *caching policies* — this tool's cache model is a single warmth scalar, deliberately crude.

**Q: What hardware numbers should I plug in?**
A: Measure, don't guess: sequential via a large-file read test, "random" via fio `rw=randread` at 1 MiB blocks, and use your QD32 result. RAM = what you can spare after OS + runtime.

## [Core Assumptions]

1. Expert activations across a batch are approximately independent, so distinct-expert counts follow the birthday-style `N(1-(1-k/N)^b)` model.
2. Cold expert reads perform at a fixed "random" bandwidth (no effective prefetch overlap) — pessimistic by design.
3. Compute cost per token is either negligible or supplied as a measured constant; the tool does not model CPU/GPU kernel details.
4. Feasibility threshold is ≥1 token/sec (interactive-ish); batch/eval workloads may accept slower rates and should read `predicted_tokens_per_sec` directly instead of the verdict.
5. Shared weights always reside in RAM; the tool checks this (`fits_ram`) and zeroes warmth if not.

---
Topics: `mixture-of-experts` `llm-inference` `disk-streaming` `benchmark` `memory-bound` `quantization` `hardware-feasibility`

Status: v0.1.1 reference oracle. Pure stdlib. MIT. All arithmetic reproducible via `python self_test.py`; demo scenarios via `python moe_disk_stream_bench.py demo`. Hardware bandwidth figures are illustrative inputs, not measurements; model architecture trends as of 2026-10-11.

Inspired by the disk-streaming MoE approach of JustVugg/colibri (https://github.com/JustVugg/colibri); no code or text from that repository is used here.
