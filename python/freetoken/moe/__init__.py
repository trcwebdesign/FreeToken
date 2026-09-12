from __future__ import annotations

from freetoken.utils import Registry

from .base import BaseMoeBackend

# Where routed experts live. The offload family (offload / cpu / hybrid) serves experts from
# pinned host banks through an ``OffloadMoeCache`` -- the GPU only holds the two-layer prefill
# double buffer -- and differs only in how decode gets the experts: ``offload`` streams the
# missing experts over PCIe into a GPU slot cache, ``cpu`` computes them on the CPU from the
# host banks, ``hybrid`` fetches at most K missing experts per layer and computes the rest on
# the CPU, overlapped. ``fused`` keeps every expert resident on the GPU.
MOE_STRATEGIES = ("fused", "offload", "cpu", "hybrid")
OFFLOAD_MOE_STRATEGIES = frozenset({"offload", "cpu", "hybrid"})
OFFLOAD_MOE_BACKENDS = OFFLOAD_MOE_STRATEGIES

SUPPORTED_MOE_BACKENDS = Registry("MoE Backend")


def is_offload_moe_strategy(strategy: str) -> bool:
    return strategy in OFFLOAD_MOE_STRATEGIES


def is_offload_moe_backend(backend: str) -> bool:
    return backend in OFFLOAD_MOE_BACKENDS


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    from .fused import FusedMoe

    return FusedMoe()


@SUPPORTED_MOE_BACKENDS.register("offload")
def create_offload_moe_backend():
    from .offload import OffloadMoeBackend

    return OffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("cpu")
def create_cpu_moe_backend():
    from .cpu_offload import CpuOffloadMoeBackend

    return CpuOffloadMoeBackend()


@SUPPORTED_MOE_BACKENDS.register("hybrid")
def create_hybrid_moe_backend():
    from .cpu_offload import HybridMoeBackend

    return HybridMoeBackend()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
    "OFFLOAD_MOE_BACKENDS",
    "is_offload_moe_backend",
    "MOE_STRATEGIES",
    "OFFLOAD_MOE_STRATEGIES",
    "is_offload_moe_strategy",
]
