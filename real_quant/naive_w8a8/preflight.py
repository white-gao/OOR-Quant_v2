from __future__ import annotations

import argparse
from typing import Any

import torch

from .modules import FP8_MAX, quantize_fp8


def _run_small_scaled_mm(device: torch.device) -> bool:
    m, k, n = 4, 16, 16
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    x_scale = (x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX)
    w_scale = (weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX)
    x_fp8 = quantize_fp8(x, x_scale, qmax=FP8_MAX)
    weight_fp8_t = quantize_fp8(weight, w_scale, qmax=FP8_MAX).t()
    y = torch._scaled_mm(
        x_fp8,
        weight_fp8_t,
        scale_a=x_scale,
        scale_b=w_scale.t().contiguous(),
        out_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    return tuple(y.shape) == (m, n) and y.dtype == torch.bfloat16


def run_preflight_checks(*, device: str = "cuda:0", run_cuda_gemm: bool = True) -> dict[str, Any]:
    device_obj = torch.device(device)
    report: dict[str, Any] = {
        "device": str(device_obj),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "has_float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
        "has_scaled_mm": hasattr(torch, "_scaled_mm"),
        "cuda_gemm_checked": False,
        "cuda_gemm_ok": False,
        "status": "NOT_RUN",
    }
    if device_obj.type != "cuda" or not torch.cuda.is_available():
        report["status"] = "SKIPPED_CUDA_GEMM"
        return report
    if not run_cuda_gemm:
        report["status"] = "SKIPPED_CUDA_GEMM"
        return report
    if not report["has_float8_e4m3fn"] or not report["has_scaled_mm"]:
        report["status"] = "MISSING_FP8_RUNTIME"
        return report

    report["cuda_gemm_checked"] = True
    try:
        report["gpu_name"] = torch.cuda.get_device_name(device_obj)
        report["cuda_gemm_ok"] = _run_small_scaled_mm(device_obj)
        report["status"] = "OK" if report["cuda_gemm_ok"] else "FAILED_CUDA_GEMM"
    except Exception as exc:  # pragma: no cover - exercised only on unsupported CUDA runtimes
        report["status"] = "FAILED_CUDA_GEMM"
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def format_preflight_report(report: dict[str, Any]) -> str:
    ordered_keys = (
        "status",
        "device",
        "gpu_name",
        "torch_version",
        "cuda_version",
        "cuda_available",
        "has_float8_e4m3fn",
        "has_scaled_mm",
        "cuda_gemm_checked",
        "cuda_gemm_ok",
        "error",
    )
    lines = []
    for key in ordered_keys:
        if key in report:
            lines.append(f"{key}: {report[key]}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for real naive W8A8 torch._scaled_mm inference.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_cuda_gemm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight_checks(device=args.device, run_cuda_gemm=not args.skip_cuda_gemm)
    print(format_preflight_report(report))
    if report["status"] not in ("OK", "SKIPPED_CUDA_GEMM"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
