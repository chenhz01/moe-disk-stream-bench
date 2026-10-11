# -*- coding: utf-8 -*-
"""self_test.py for moe-disk-stream-bench.

Three test families (ALIGNMENT-16 rule 97):
  A. functional  -- closed-form math checks, warmth effect, sweep, crossover
  B. error paths -- zero/negative bandwidth, top_k > experts, bad warmth
  C. extreme edge -- top_k == experts (dense), 1-expert model, huge params,
                     empty sweep, pathological cold penalty

Run:  python self_test.py
"""
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_disk_stream_bench as mdb

PASS = 0
FAIL = 0
RESULTS = []


def check(name, cond, info=""):
    global PASS, FAIL
    tag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    RESULTS.append("%-4s %s %s" % (tag, name, info))


def section(t):
    RESULTS.append("")
    RESULTS.append("== %s ==" % t)


# ---------- A. functional ----------
section("A. functional")
m = mdb.ModelSpec(num_experts=100, top_k=10, expert_params=1e6,
                  shared_params=0, bytes_per_param=1, num_layers=1)
h = mdb.HardwareSpec(disk_seq_MBps=1000, disk_rand_MBps=1000, ram_GB=1e9)
r = mdb.predict(m, h, cache_warmth=0.0)
# closed form: 10 experts * 1e6 B = 10e6 bytes; tps = 1000 * MB / 10e6
expect = h.disk_rand_MBps * mdb.MB / (10 * 1e6)
check("closed-form tps", abs(r["predicted_tokens_per_sec"] - expect) < 0.01,
      "got %.2f want %.2f" % (r["predicted_tokens_per_sec"], expect))
check("activation rate recorded", r["model"]["activation_rate"] == 0.1)
check("streamable at 100 tps", r["streamable"] is True and r["bottleneck"] == "disk")

r2 = mdb.predict(m, h, cache_warmth=0.5)
check("warmth halves disk reads", abs(r2["per_token_disk_MB"] * 2 - r["per_token_disk_MB"]) < 0.01,
      "got %.3f MB vs %.3f" % (r2["per_token_disk_MB"], r["per_token_disk_MB"]))
check("warmth doubles tps", abs(r2["predicted_tokens_per_sec"] - 2 * r["predicted_tokens_per_sec"]) < 0.01)

# batch amortizes reads via expert reuse: tps rises, sublinearly (<4x)
r3 = mdb.predict(m, h, batch=4, cache_warmth=0.0)
check("batch=4 raises tps", r3["predicted_tokens_per_sec"] > r["predicted_tokens_per_sec"],
      "%.1f -> %.1f" % (r["predicted_tokens_per_sec"], r3["predicted_tokens_per_sec"]))
check("batch amortization is sublinear",
      r3["predicted_tokens_per_sec"] < 4 * r["predicted_tokens_per_sec"])
check("batch lowers per-token disk reads",
      r3["per_token_disk_MB"] < r["per_token_disk_MB"])

sw = mdb.sweep(m, h, warmth_steps=(0.0, 1.0))
check("sweep shape", len(sw["warmth_sweep"]) == 2 and len(sw["sparsity_sweep"]) == 100)
check("warmth=1 => zero disk reads", sw["warmth_sweep"][1]["per_token_disk_MB"] == 0.0)

x = mdb.crossover_topk(m, h)
check("crossover finds max feasible top_k", x is not None and x >= 10,
      "crossover_topk=%s" % x)

# compute-bound case: fast disk, slow compute
h_compute = mdb.HardwareSpec(1000, 1000, 1e9, compute_tokens_per_sec=50)
r4 = mdb.predict(m, h_compute, cache_warmth=0.0)
check("compute bottleneck detected", r4["bottleneck"] == "compute"
      and r4["predicted_tokens_per_sec"] <= 50.0)

# CLI smoke
cli = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "moe_disk_stream_bench.py")]
p1 = subprocess.run(cli + ["predict", "--experts", "128", "--topk", "8",
                           "--expert-params", "8e6", "--disk-seq", "500",
                           "--disk-rand", "80", "--ram", "2"],
                    capture_output=True, text=True)
check("cli predict exit 0 + json", p1.returncode == 0
      and json.loads(p1.stdout)["streamable"] is False, p1.stdout[:40].replace("\n", " "))
p2 = subprocess.run(cli + ["demo"], capture_output=True, text=True)
check("cli demo exit 0", p2.returncode == 0)

# ---------- B. error paths ----------
section("B. error paths")
for label, fn in [
    ("zero bandwidth", lambda: mdb.HardwareSpec(0, 100, 1)),
    ("negative rand bw", lambda: mdb.HardwareSpec(100, -1, 1)),
    ("cold penalty < 1", lambda: mdb.HardwareSpec(100, 100, 1, cold_cache_penalty=0.5)),
    ("top_k > experts", lambda: mdb.ModelSpec(8, 16, 1e6, 0)),
    ("zero expert params", lambda: mdb.ModelSpec(8, 2, 0, 0)),
    ("negative shared", lambda: mdb.ModelSpec(8, 2, 1e6, -5)),
    ("zero layers", lambda: mdb.ModelSpec(8, 2, 1e6, 0, num_layers=0)),
    ("batch 0", lambda: mdb.predict(m, h, batch=0)),
    ("warmth > 1", lambda: mdb.predict(m, h, cache_warmth=1.5)),
    ("negative ram", lambda: mdb.HardwareSpec(100, 100, -3)),
]:
    try:
        fn()
        check("rejects %s" % label, False, "no exception")
    except ValueError as exc:
        check("rejects %s" % label, True, str(exc)[:40])

# ---------- C. extreme edge ----------
section("C. extreme edge")
dense = mdb.ModelSpec(4, 4, 1e6, 0, num_layers=1)
rd = mdb.predict(dense, h, cache_warmth=0.0)
check("dense model (top_k==N) reads everything",
      abs(rd["per_token_disk_MB"] * mdb.MB - 4 * 1e6) < 1000.0,
      "got %.3f MB" % rd["per_token_disk_MB"])

tiny = mdb.ModelSpec(1, 1, 1e6, 0)
rt = mdb.predict(tiny, h)
check("single-expert model works", rt["streamable"] and rt["model"]["activation_rate"] == 1.0)

huge = mdb.ModelSpec(1024, 512, 1e12, 1e11, bytes_per_param=1, num_layers=120)
rh = mdb.predict(huge, mdb.HardwareSpec(500, 80, 2))
check("frontier-scale model does not stream on 2GB RAM",
      rh["streamable"] is False and rh["fits_ram"] is False)

r_warm_all = mdb.predict(m, mdb.HardwareSpec(500, 80, 1e9, cold_cache_penalty=1e9),
                         cache_warmth=1.0)
check("pathological cold penalty harmless when warm",
      r_warm_all["predicted_tokens_per_sec"] > 0)

t0 = time.perf_counter()
big_sweep = mdb.sweep(mdb.ModelSpec(256, 16, 1e7, 1e8), h)
dt = time.perf_counter() - t0
check("256-expert sweep under 2s", dt < 2.0 and len(big_sweep["sparsity_sweep"]) == 256,
      "%.2fs" % dt)

# ---------- report ----------
print("\n".join(RESULTS))
print("")
print("moe-disk-stream-bench self-test: %d/%d PASS (families A/B/C)" % (PASS, PASS + FAIL))
demo = mdb._demo()
print("demo(64e/4k/24L 4-bit, NVMe cold): b1=%.2f tps b16=%.2f tps b16warm50=%.2f tps crossover_topk=%s | repro: python moe_disk_stream_bench.py demo"
      % (demo["nvme_batch1_cold"]["predicted_tokens_per_sec"],
         demo["nvme_batch16_cold"]["predicted_tokens_per_sec"],
         demo["nvme_batch16_warm50"]["predicted_tokens_per_sec"],
         demo["crossover_topk_nvme"]))
sys.exit(0 if FAIL == 0 else 1)
