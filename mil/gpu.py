"""GPU verification helpers used by the demo scripts and explorer app."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


PYTORCH_CU12X_INDEX_URL = "https://download.pytorch.org/whl/cu128"
MATMUL_SLOW_THRESHOLD_MS = 500.0


@dataclass
class GpuCheckResult:
    ok: bool
    fail_kind: str | None
    messages: list[str]
    elapsed_ms: float | None = None
    capability: tuple[int, int] | None = None


def blackwell_torch_remediation() -> str:
    return (
        "Remediation: reinstall PyTorch from the CUDA 12.8 wheel index rather than "
        f"the default pip index, for example:\n"
        f"  pip install torch torchvision torchaudio --index-url {PYTORCH_CU12X_INDEX_URL}\n"
        "Check https://pytorch.org/get-started/locally/ for the current CUDA 12.x "
        "wheel index before trusting this hardcoded URL long-term."
    )


def nvidia_smi_output() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found on PATH."
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - depends on host driver state.
        return False, f"nvidia-smi failed: {type(exc).__name__}: {exc}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, output or f"nvidia-smi exited {completed.returncode}."
    return True, output


def run_cuda_matmul_check(
    *,
    size: int = 2048,
    dtype_name: str = "float16",
    slow_threshold_ms: float = MATMUL_SLOW_THRESHOLD_MS,
) -> GpuCheckResult:
    messages: list[str] = []
    smi_ok, smi_text = nvidia_smi_output()
    messages.append("=== nvidia-smi ===")
    messages.append(smi_text)
    if not smi_ok:
        messages.append("FAIL[no_gpu_visible]: nvidia-smi is unavailable or failed.")
        return GpuCheckResult(ok=False, fail_kind="no_gpu_visible", messages=messages)

    try:
        import torch
    except ImportError as exc:
        messages.append(f"FAIL[torch_missing]: torch import failed: {exc}")
        messages.append(blackwell_torch_remediation())
        return GpuCheckResult(ok=False, fail_kind="torch_missing", messages=messages)

    messages.append("=== torch ===")
    messages.append(f"torch.__version__ = {torch.__version__}")
    messages.append(f"torch.version.cuda = {torch.version.cuda}")
    if torch.version.cuda is None:
        messages.append(
            "FAIL[torch_cpu_build]: nvidia-smi sees a GPU, but this is a CPU-only "
            "PyTorch build."
        )
        messages.append(blackwell_torch_remediation())
        return GpuCheckResult(ok=False, fail_kind="torch_cpu_build", messages=messages)

    available = torch.cuda.is_available()
    messages.append(f"torch.cuda.is_available() = {available}")
    if not available:
        messages.append(
            "FAIL[no_gpu_visible]: torch does not see CUDA. This points to WSL/driver "
            "configuration, not a Blackwell-specific PyTorch kernel build."
        )
        return GpuCheckResult(ok=False, fail_kind="no_gpu_visible", messages=messages)

    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    messages.append(f"torch.cuda.get_device_name() = {device_name}")
    messages.append(f"torch.cuda.get_device_capability() = {capability}")

    dtype = _resolve_dtype(torch, dtype_name)
    try:
        torch.cuda.empty_cache()
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        for _ in range(2):
            _ = a @ b
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
        _ = float(c[0, 0].item())
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        messages.append(f"FAIL[kernel_unavailable]: cuda matmul raised: {text}")
        if "no kernel image is available for execution on the device" in text.lower():
            messages.append("Detected Blackwell/sm_120 kernel-image failure.")
        messages.append(blackwell_torch_remediation())
        return GpuCheckResult(
            ok=False,
            fail_kind="kernel_unavailable",
            messages=messages,
            capability=capability,
        )

    best_ms = min(times)
    median_ms = sorted(times)[len(times) // 2]
    messages.append(
        f"cuda matmul: {size}x{size} @ {size}x{size}, dtype={dtype}, "
        f"best_ms={best_ms:.3f}, median_ms={median_ms:.3f}"
    )
    if median_ms > slow_threshold_ms:
        messages.append(
            f"FAIL[slow_kernel]: median matmul time {median_ms:.3f} ms exceeds "
            f"{slow_threshold_ms:.1f} ms. CUDA is visible, but this is suspiciously "
            "slow for optimized GPU kernels."
        )
        messages.append(blackwell_torch_remediation())
        return GpuCheckResult(
            ok=False,
            fail_kind="slow_kernel",
            messages=messages,
            elapsed_ms=median_ms,
            capability=capability,
        )

    messages.append("PASS: CUDA tensor matmul ran on the GPU with real timing.")
    return GpuCheckResult(
        ok=True,
        fail_kind=None,
        messages=messages,
        elapsed_ms=median_ms,
        capability=capability,
    )


def _resolve_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype for GPU check: {dtype_name!r}")
