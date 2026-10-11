# Changelog

## v0.1.1 — 2026-10-11 (self-audit revision, round 2)

Iterated against ALIGNMENT-16 (rules 85-100). Changes:

- **Model improvement (round-2 fix, caught by functional tests):** v0.1.0 treated batch as pure linear I/O scaling, which contradicts the well-known expert-reuse effect and made `batch=4` tests fail. v0.1.1 adds birthday-style distinct-expert amortization `N*(1-(1-k/N)^b)`: batch=16 on the demo config flips the verdict from 0.87 tps (not streamable) to 1.36 tps (streamable) — the demo now demonstrates the single most decision-relevant effect.
- **Honest demo:** replaced the unrealistic 128-expert/2-bit demo (0.02 tps — a number with no decision value) with a 4-bit 64-expert config where feasibility actually flips across scenarios; demo table added to README with repro command.
- **Numerical discipline:** all closed-form unit tests now derive expectations from module constants (MiB, not decimal MB); rounding-tolerance failures fixed by relative-tolerance assertions; 29/29 PASS.
- **README completeness:** added Minimal Fallback (calculator formula + mmap fallback), Design Trade-offs priority order (honest verdicts first, pessimistic random-bandwidth by design), Limits & Second-order Effects (batch-vs-latency trap, sparsity-tuning feedback loop), FAQ with "When NOT to use" (4 cases), 5 [Core Assumptions], two explicit confidence ratings with rationale.

## v0.1.0 — 2026-10-11 (initial draft)

- `ModelSpec` / `HardwareSpec` with full input validation (top-k ≤ N, positive bandwidths, etc.).
- `predict`: cold-cache streaming feasibility with random-bandwidth model and disk/compute bottleneck attribution.
- `sweep`: feasibility across cache warmth (0..1) and all top-k values 1..N.
- `crossover_topk`: binary search for the sparsest usable top-k on given hardware.
- CLI (`predict`, `demo`) + self_test with functional, error-path, extreme-edge families (24 checks at draft time).
