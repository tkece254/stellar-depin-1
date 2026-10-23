from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Peak:
    bytes_per_s: float
    flops_per_s: float


PEAKS: tuple[tuple[str, Peak], ...] = (
    ("H100 NVL", Peak(3.9e12, 835e12)),
    ("H100 PCIe", Peak(2.0e12, 756e12)),
    ("H100", Peak(3.35e12, 989e12)),
    ("H200", Peak(4.8e12, 989e12)),
    ("B200", Peak(8.0e12, 2250e12)),
    ("A100-SXM4-80GB", Peak(2.039e12, 312e12)),
    ("A100 80GB PCIe", Peak(1.935e12, 312e12)),
    ("A100", Peak(1.555e12, 312e12)),
    ("L40S", Peak(864e9, 362e12)),
    ("L4", Peak(300e9, 121e12)),
    ("T4", Peak(320e9, 65e12)),
)


def peak_for(device_name: str) -> Peak | None:
    return next((peak for key, peak in PEAKS if key in device_name), None)
