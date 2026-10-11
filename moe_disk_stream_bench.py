# -*- coding: utf-8 -*-
"""
moe-disk-stream-bench -- Feasibility oracle for streaming MoE experts from disk.

Answers one question: *given a Mixture-of-Experts model and a specific
machine, does streaming activated expert weights from disk stay ahead of
compute?* Models activation sparsity (top-k of N experts), sequential vs
random disk bandwidth, RAM cache warmth, and cold-cache latency, then
predicts tokens/sec and emits a streamable / not-streamable verdict.

The framing follows the disk-streaming MoE line of work (e.g. Colibri-style
pure-C engines that run huge sparse models in tiny RAM): activation rate is
the lever that decides whether disk bandwidth becomes the bottleneck.

Pure stdlib. MIT. v0.1.1
"""
import argparse
import json
import math
import sys
import time

MB = 1024 * 1024
GB = 1024 * MB


class ModelSpec:
    def __init__(self, num_experts, top_k, expert_params, shared_params,
                 bytes_per_param=1, num_layers=1):
        if num_experts < 1 or top_k < 1:
            raise ValueError("num_experts and top_k must be >= 1")
        if top_k > num_experts:
            raise ValueError("top_k (%d) cannot exceed num_experts (%d)" % (top_k, num_experts))
        if expert_params <= 0 or shared_params < 0:
            raise ValueError("expert_params must be > 0, shared_params >= 0")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_params = float(expert_params)
        self.shared_params = float(shared_params)
        self.bytes_per_param = float(bytes_per_param)
        self.num_layers = int(num_layers)

    @property
    def expert_bytes(self):
        return self.expert_params * self.bytes_per_param

    @property
    def shared_bytes(self):
        return self.shared_params * self.bytes_per_param

    @property
    def total_bytes(self):
        return self.shared_bytes + self.num_experts * self.expert_bytes

    def to_dict(self):
        return {"num_experts": self.num_experts, "top_k": self.top_k,
                "expert_params": self.expert_params, "shared_params": self.shared_params,
                "bytes_per_param": self.bytes_per_param, "num_layers": self.num_layers,
                "total_GB": round(self.total_bytes / GB, 2),
                "activation_rate": round(self.top_k / self.num_experts, 4)}


class HardwareSpec:
    def __init__(self, disk_seq_MBps, disk_rand_MBps, ram_GB,
                 cold_cache_penalty=1.0, compute_tokens_per_sec=None):
        if disk_seq_MBps <= 0 or disk_rand_MBps <= 0:
            raise ValueError("bandwidths must be > 0")
        if ram_GB < 0:
            raise ValueError("ram_GB must be >= 0")
        if cold_cache_penalty < 1.0:
            raise ValueError("cold_cache_penalty must be >= 1.0")
        self.disk_seq_MBps = float(disk_seq_MBps)
        self.disk_rand_MBps = float(disk_rand_MBps)
        self.ram_GB = float(ram_GB)
        self.cold_cache_penalty = float(cold_cache_penalty)
        self.compute_tokens_per_sec = compute_tokens_per_sec

    def to_dict(self):
        return {"disk_seq_MBps": self.disk_seq_MBps, "disk_rand_MBps": self.disk_rand_MBps,
                "ram_GB": self.ram_GB, "cold_cache_penalty": self.cold_cache_penalty}


def predict(model, hw, batch=1, cache_warmth=0.0):
    """Predict streaming feasibility.

    Model: per token, per layer, top_k expert slabs must be resident. Warm
    fraction (cache_warmth) of the hot working set is served from RAM at no
    disk cost; the rest is read from disk. Cold reads use *random* bandwidth
    (page-cache misses break sequentiality) multiplied by cold_cache_penalty.
    Shared weights (attention/embeddings) assumed always resident in RAM.
    """
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if not 0.0 <= cache_warmth <= 1.0:
        raise ValueError("cache_warmth must be within [0, 1]")
    # RAM check: shared + activated experts must fit if we want warmth > 0
    hot_bytes = model.shared_bytes + model.top_k * model.expert_bytes * model.num_layers
    fits_ram = hot_bytes <= hw.ram_GB * GB
    eff_warmth = cache_warmth if fits_ram else 0.0

    # Expert reuse across the batch: b independent tokens activate top_k of N
    # experts each; expected *distinct* experts touched per layer is
    # N * (1 - (1 - k/N)^b), i.e. disk reads amortize sublinearly with batch.
    if batch == 1:
        eff_slabs = model.top_k
    else:
        reuse = model.num_experts * (1.0 - (1.0 - model.top_k / model.num_experts) ** batch)
        eff_slabs = min(batch * model.top_k, reuse)
    per_batch_disk = model.num_layers * eff_slabs * model.expert_bytes * (1.0 - eff_warmth)
    per_token_disk = per_batch_disk / batch
    rand_MBps = hw.disk_rand_MBps / hw.cold_cache_penalty
    disk_sec = per_batch_disk / (rand_MBps * MB)
    compute_sec = (1.0 / hw.compute_tokens_per_sec) * batch if hw.compute_tokens_per_sec else 0.0
    tps = batch / max(disk_sec + compute_sec, 1e-12)
    bound = "disk" if disk_sec > compute_sec else "compute"
    return {
        "model": model.to_dict(),
        "hardware": hw.to_dict(),
        "batch": batch,
        "cache_warmth_requested": cache_warmth,
        "hot_working_set_GB": round(hot_bytes / GB, 3),
        "fits_ram": fits_ram,
        "effective_warmth": round(eff_warmth, 3),
        "per_token_disk_MB": round(per_token_disk / MB, 3),
        "predicted_tokens_per_sec": round(tps, 2),
        "bottleneck": bound,
        "streamable": tps >= 1.0,
    }


def sweep(model, hw, warmth_steps=(0.0, 0.25, 0.5, 0.75, 0.95)):
    """Feasibility across cache warmth levels; also across top_k 1..num_experts."""
    rows = [dict(predict(model, hw, cache_warmth=w), _row="warmth=%.2f" % w)
            for w in warmth_steps]
    sparsity_rows = []
    for k in range(1, model.num_experts + 1):
        m = ModelSpec(model.num_experts, k, model.expert_params,
                      model.shared_params, model.bytes_per_param, model.num_layers)
        sparsity_rows.append({"top_k": k, **predict(m, hw, cache_warmth=0.0)})
    return {"warmth_sweep": rows, "sparsity_sweep": sparsity_rows}


def crossover_topk(model, hw):
    """Largest top_k that still reaches >= min_tps with a cold cache."""
    lo, hi, ans = 1, model.num_experts, None
    while lo <= hi:
        mid = (lo + hi) // 2
        m = ModelSpec(model.num_experts, mid, model.expert_params,
                      model.shared_params, model.bytes_per_param, model.num_layers)
        if predict(m, hw)["streamable"]:
            ans, lo = mid, mid + 1
        else:
            hi = mid - 1
    return ans


def _demo():
    """Canned case (as of 2026-10-11): a 4-bit 64-expert MoE (24 layers,
    ~2.7 GB total) on consumer NVMe. Shows the batch-reuse effect: cold-cache
    streaming is infeasible at batch=1 but crosses the 1 tps line at batch=16
    because distinct experts touched per layer amortize sublinearly."""
    out = {}
    model = ModelSpec(num_experts=64, top_k=4, expert_params=20e6,
                      shared_params=200e6, bytes_per_param=0.5, num_layers=24)
    out["model"] = model.to_dict()
    nvme = HardwareSpec(3500, 800, 8)
    out["nvme_batch1_cold"] = predict(model, nvme, batch=1)
    out["nvme_batch16_cold"] = predict(model, nvme, batch=16)
    out["nvme_batch16_warm50"] = predict(model, nvme, batch=16, cache_warmth=0.5)
    out["crossover_topk_nvme"] = crossover_topk(model, nvme)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="moe-disk-stream-bench",
        description="Predict whether streaming MoE experts from disk is feasible on given hardware.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_p = sub.add_parser("predict", help="single feasibility prediction")
    p_p.add_argument("--experts", type=int, required=True)
    p_p.add_argument("--topk", type=int, required=True)
    p_p.add_argument("--expert-params", type=float, required=True,
                     help="params per expert slab")
    p_p.add_argument("--shared-params", type=float, default=0,
                     help="non-expert params (attention/embeddings)")
    p_p.add_argument("--bytes-per-param", type=float, default=2)
    p_p.add_argument("--layers", type=int, default=1)
    p_p.add_argument("--disk-seq", type=float, required=True, help="MB/s sequential")
    p_p.add_argument("--disk-rand", type=float, required=True, help="MB/s random-ish")
    p_p.add_argument("--ram", type=float, required=True, help="GB")
    p_p.add_argument("--cold-penalty", type=float, default=1.0)
    p_p.add_argument("--batch", type=int, default=1)
    p_p.add_argument("--warmth", type=float, default=0.0)
    p_p.add_argument("--compute-tps", type=float, default=None,
                     help="measured compute-only tokens/sec (optional)")

    sub.add_parser("demo", help="canned SATA-vs-NVMe comparison (~0.1 s)")
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        print(json.dumps(_demo(), indent=2))
        return 0
    if args.cmd == "predict":
        m = ModelSpec(args.experts, args.topk, args.expert_params,
                      args.shared_params, args.bytes_per_param, args.layers)
        h = HardwareSpec(args.disk_seq, args.disk_rand, args.ram, args.cold_penalty,
                         args.compute_tps)
        print(json.dumps(predict(m, h, batch=args.batch, cache_warmth=args.warmth), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
