# Alignment Audit (Rules 85-100 / 16 items) — moe-disk-stream-bench v0.1.1

Audited: 2026-10-11, against ALIGNMENT-16.md.

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Safety/Ethics section | N/A | Hardware feasibility math; no security-sensitive content |
| 2 | Privacy (no secrets/real names) | PASS | No keys/tokens/real names/real emails anywhere; LICENSE copyright "zhengming contributors" |
| 3 | Copyright boundary | PASS | JustVugg/colibri referenced by link + 1-sentence structural paraphrase (744B/9.9GB figure cited as its claim, marked as such); no >50-char contiguous borrowing; LICENSE (MIT) present |
| 4 | Fact-check annotations | PASS | All tps figures from `python moe_disk_stream_bench.py demo` (deterministic math, repro command given); hardware bandwidths explicitly declared as illustrative inputs; accuracy-vs-real-engine claim tagged [estimated] |
| 5 | Confidence grading | PASS | High (≥80%) for internal math via closed-form tests; Medium (60–79%) for 2x real-engine tracking, with rationale |
| 6 | Minimal Fallback (PLAN B) | PASS | Calculator one-formula fallback + qualitative no-go rule + OS mmap fallback |
| 7 | Quickstart timing | PASS | "~2 min to first verdict"; per-command durations (<1 s self-test, ~0.1 s demo) |
| 8 | Design trade-offs / priorities | PASS | "honest verdicts > simplicity > accuracy > features"; pessimistic random-bandwidth choice justified |
| 9 | CHANGELOG ≥2 entries | PASS | v0.1.0 + v0.1.1 with named improvements (batch amortization model, demo re-tuning, tolerance fixes) |
| 10 | Exemption (session rule 94) | N/A | Session-level rule; static repo content unaffected |
| 11 | Time-sensitive data dated | PASS | "as of 2026-10-11" in Quickstart table, Status footer; hardware numbers declared as inputs not measurements |
| 12 | Core Assumptions ≥3 | PASS | 5 assumptions under [Core Assumptions] (independent routing, no prefetch, constant compute, ≥1 tps threshold, shared-in-RAM) |
| 13 | Logic triple (functional/error/edge) | PASS | self_test.py: A functional (closed-form tps, warmth, batch amortization, crossover, CLI), B error paths (10 invalid-input cases), C extreme (dense top_k==N, 1-expert, 1024-expert frontier-scale, pathological penalty, 256-expert sweep) — 29/29 PASS |
| 14 | FAQ "When NOT to use" | PASS | 4 concrete anti-use cases incl. Socratic framing |
| 15 | ≥2 content iterations | PASS | v0.1.0 → v0.1.1; round-2 dimension "correctness": batch expert-reuse model added (batch=4 test went FAIL → PASS; demo verdict flips 0.87→1.36 tps), ~20% of self-test rewritten for numeric discipline |
| 16 | ≤60-word first screen | PASS | Opening paragraph is 40 words incl. verdict-judgment sentence |

Additional: **Limits & Second-order Effects** present (rule 16 second-order thinking): throughput-vs-latency trap from over-batching, sparsity-tuning feedback loop risk, lower-bound-only interpretation.
