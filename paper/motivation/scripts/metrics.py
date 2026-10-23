"""CPU/FP32 statistics; no training, masking of generation, or plot selection."""
import numpy as np
import torch

KS = (1, 2, 4, 8, 16, 32, 64, 128)
GAP_EDGES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf"))
GAP_LABELS = ("(0,.01]", "(.01,.05]", "(.05,.1]", "(.1,.25]",
              "(.25,.5]", "(.5,1]", "(1,2]", ">2")


def checked(logits):
    logits = torch.as_tensor(logits).detach().cpu().float()
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("Expected a finite one-dimensional logit vector")
    return logits


def concentration(logits):
    z = checked(logits)
    logp = z.log_softmax(-1)
    p = logp.exp()
    ordered = p.sort(descending=True).values.cumsum(0)
    entropy = float(-(p * logp).sum())
    return {"entropy_nats": entropy, "normalized_entropy": entropy / np.log(len(z)),
            "effective_candidates": float(np.exp(entropy)), "vocab_size": len(z),
            "mass": {str(k): float(ordered[k - 1]) for k in KS if k <= len(z)}}


def compare_logits(teacher, student, k=32):
    t, s = checked(teacher), checked(student)
    if t.shape != s.shape or 2 * k > len(t):
        raise ValueError("Mismatched vocabulary or insufficient candidates")
    # Input indices are sorted by token ID. Stable sorting makes tie-breaking explicit.
    order = torch.argsort(t, descending=True, stable=True)
    qorder = torch.argsort(s, descending=True, stable=True)
    positive, negative = order[:k], order[k:2*k]
    td = t[positive, None] - t[negative][None, :]
    sd = s[positive, None] - s[negative][None, :]
    bins = []
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        selected = (td > lo) & (td <= hi)
        bins.append({"count": int(selected.sum()),
                     "flips": int((selected & (sd < 0)).sum()),
                     "student_ties": int((selected & (sd == 0)).sum())})
    logp, logq = t.log_softmax(-1), s.log_softmax(-1)
    return {"kl": max(0.0, float((logp.exp() * (logp - logq)).sum())),
            "top32_retention": len(set(positive.tolist()) & set(qorder[:k].tolist())) / k,
            "tie_aware_top32_retention": float((t[qorder[:k]] >= t[order[k-1]]).float().mean()),
            "top1_agreement": int(order[0] == qorder[0]),
            "cutoff_gap": float(t[order[k-1]] - t[order[k]]),
            "teacher_zero_gap_pairs": int((td == 0).sum()),
            "teacher_near_tie_pairs": int(((td > 0) & (td <= .01)).sum()),
            "boundary_bins": bins}


def path_overlap(teacher_paths, student_paths):
    t, s = set(map(tuple, teacher_paths)), set(map(tuple, student_paths))
    return len(t & s) / len(t) if t else None


def bootstrap_mean(values, seed=42, repeats=2000):
    """Rows are independent queries/documents, not tokens or candidate pairs."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"mean": None, "low": None, "high": None, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(repeats)])
    return {"mean": float(x.mean()), "low": float(np.quantile(boot, .025)),
            "high": float(np.quantile(boot, .975)), "n": len(x)}
