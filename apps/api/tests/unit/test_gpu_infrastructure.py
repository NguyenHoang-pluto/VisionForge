"""GPU lease and model registry, exercised entirely with fakes.

No CUDA, no model weights, no network. The budget arithmetic, the eviction order
and the mutex are the parts most likely to be wrong and the least pleasant to
debug on real hardware, so they are tested where they can be driven directly.

A real-GPU smoke test lives in ``tests/integration/test_gpu_lane.py`` and is
marked ``gpu``.
"""

from __future__ import annotations

import threading
import time

import pytest

from visionforge.infra.ml.lease import (
    DEFAULT_VRAM_BUDGET_MB,
    GpuBudgetExceededError,
    GpuBusyError,
    GpuLeaseManager,
    LeaseReceipt,
    VramSnapshot,
)
from visionforge.infra.ml.registry import ModelRegistry


class FakeModel:
    def __init__(self, key: str) -> None:
        self.key = key
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


class FakeSpec:
    """A model that records how often it was actually built."""

    def __init__(self, key: str, vram_mb: int, version: str = "1") -> None:
        self.key = key
        self.version = version
        self.estimated_vram_mb = vram_mb
        self.load_count = 0

    def load(self) -> FakeModel:
        self.load_count += 1
        return FakeModel(self.key)


# ------------------------------------------------------------------- budget
class TestBudget:
    def test_default_budget_is_below_the_physical_card(self) -> None:
        """4096 MiB exists; roughly 3.3 GiB is reachable under WDDM.

        Budgeting the full card is how you get an OOM at 95% utilisation: the
        desktop compositor holds several hundred MiB and the caching allocator
        reserves more than it reports as allocated.
        """
        assert DEFAULT_VRAM_BUDGET_MB < 4096
        assert DEFAULT_VRAM_BUDGET_MB >= 2048

    def test_fits_when_there_is_room(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        assert manager.fits(900)

    def test_does_not_fit_once_committed(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        manager.register("a", 900)

        assert not manager.fits(900)
        assert manager.committed_mb == 900

    def test_registering_the_same_model_twice_does_not_double_count(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        manager.register("a", 400)
        manager.register("a", 400)

        assert manager.committed_mb == 400

    def test_release_frees_the_budget(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        manager.register("a", 900)
        manager.release("a")

        assert manager.committed_mb == 0
        assert manager.fits(900)

    def test_a_model_larger_than_the_whole_budget_is_rejected_up_front(self) -> None:
        """A clear error at the boundary beats a CUDA OOM three layers down."""
        manager = GpuLeaseManager(budget_mb=1000)

        with pytest.raises(GpuBudgetExceededError, match="whole budget"):
            manager.plan_eviction(4000)


# ----------------------------------------------------------------- eviction
class TestEviction:
    def test_nothing_is_evicted_when_it_already_fits(self) -> None:
        manager = GpuLeaseManager(budget_mb=3000)
        manager.register("a", 900)

        assert manager.plan_eviction(900) == ()

    def test_evicts_least_recently_used_first(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        manager.register("old", 400)
        manager.register("new", 400)

        assert manager.plan_eviction(400) == ("old",)

    def test_touching_a_model_makes_it_most_recently_used(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        manager.register("a", 400)
        manager.register("b", 400)
        manager.register("a", 400)  # a is used again

        assert manager.plan_eviction(400) == ("b",)

    def test_evicts_only_as_many_as_needed(self) -> None:
        manager = GpuLeaseManager(budget_mb=1000)
        for name in ("a", "b", "c", "d"):
            manager.register(name, 250)

        assert manager.plan_eviction(250) == ("a",)

    def test_evicts_several_when_one_is_not_enough(self) -> None:
        """Budget 1000, four residents of 250 (full). A 800 MiB model arrives.

        Evicting three frees 750 and leaves 250 committed: 250 + 800 = 1050,
        still over. All four must go.
        """
        manager = GpuLeaseManager(budget_mb=1000)
        for name in ("a", "b", "c", "d"):
            manager.register(name, 250)

        assert manager.plan_eviction(800) == ("a", "b", "c", "d")

    def test_evicts_exactly_enough_and_no_more(self) -> None:
        """Budget 1000, four residents of 250. A 500 MiB model needs two gone."""
        manager = GpuLeaseManager(budget_mb=1000)
        for name in ("a", "b", "c", "d"):
            manager.register(name, 250)

        assert manager.plan_eviction(500) == ("a", "b")


# -------------------------------------------------------------------- mutex
class TestLease:
    def test_lease_is_exclusive(self) -> None:
        """Two inferences at once on 4 GiB is an immediate OOM."""
        manager = GpuLeaseManager(budget_mb=3000, timeout_s=0.2)
        observed: list[str] = []

        def worker(tag: str) -> None:
            try:
                with manager.acquire(model=tag, estimated_mb=10):
                    observed.append(f"{tag}:in")
                    time.sleep(0.4)
                    observed.append(f"{tag}:out")
            except GpuBusyError:
                observed.append(f"{tag}:busy")

        first = threading.Thread(target=worker, args=("a",))
        second = threading.Thread(target=worker, args=("b",))
        first.start()
        time.sleep(0.05)
        second.start()
        first.join()
        second.join()

        assert observed[0] == "a:in"
        # The second thread cannot enter while the first holds the lease.
        assert "b:in" not in observed[:2]

    def test_timeout_surfaces_as_an_error_not_a_hang(self) -> None:
        """A deadlock must become a failed job, not a wedged worker."""
        manager = GpuLeaseManager(budget_mb=3000, timeout_s=0.1)
        blocked: list[bool] = []

        def hold() -> None:
            with manager.acquire(model="holder", estimated_mb=10):
                time.sleep(0.5)

        def contend() -> None:
            try:
                with manager.acquire(model="waiter", estimated_mb=10):
                    blocked.append(False)
            except GpuBusyError:
                blocked.append(True)

        holder = threading.Thread(target=hold)
        holder.start()
        time.sleep(0.05)
        waiter = threading.Thread(target=contend)
        waiter.start()
        waiter.join()
        holder.join()

        assert blocked == [True]

    def test_lease_is_released_even_when_the_body_raises(self) -> None:
        manager = GpuLeaseManager(budget_mb=3000, timeout_s=0.5)

        # Nesting is the assertion: the error must escape the lease context.
        with pytest.raises(RuntimeError):  # noqa: SIM117
            with manager.acquire(model="a", estimated_mb=10):
                raise RuntimeError("inference blew up")

        # Still acquirable, so the lock was not leaked.
        with manager.acquire(model="b", estimated_mb=10):
            pass

    def test_same_thread_can_reenter(self) -> None:
        """RLock: a nested lease inside one task must not self-deadlock."""
        manager = GpuLeaseManager(budget_mb=3000, timeout_s=0.5)
        # Nesting is the assertion: a re-entrant acquire must not self-deadlock.
        with manager.acquire(model="a", estimated_mb=10):  # noqa: SIM117
            with manager.acquire(model="a", estimated_mb=10):
                pass


# ----------------------------------------------------------------- registry
class TestModelRegistry:
    def test_unknown_model_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            ModelRegistry(GpuLeaseManager()).get("nope")

    def test_model_is_loaded_lazily(self) -> None:
        spec = FakeSpec("a", 100)
        registry = ModelRegistry(GpuLeaseManager(budget_mb=1000))
        registry.register(spec)

        assert spec.load_count == 0, "registering must not load"
        registry.get("a")
        assert spec.load_count == 1

    def test_model_stays_warm_across_calls(self) -> None:
        """The reason the registry exists: a cold CLIP load is seconds.

        Measured on this machine: 5140 ms cold versus 138 ms warm.
        """
        spec = FakeSpec("a", 100)
        registry = ModelRegistry(GpuLeaseManager(budget_mb=1000))
        registry.register(spec)

        first = registry.get("a")
        second = registry.get("a")

        assert first is second
        assert spec.load_count == 1

    def test_unload_releases_the_model_and_the_budget(self) -> None:
        spec = FakeSpec("a", 100)
        lease = GpuLeaseManager(budget_mb=1000)
        registry = ModelRegistry(lease)
        registry.register(spec)

        model = registry.get("a")
        registry.unload("a")

        assert model.unloaded is True
        assert lease.committed_mb == 0
        assert not registry.is_loaded("a")

    def test_unloading_an_absent_model_is_a_no_op(self) -> None:
        ModelRegistry(GpuLeaseManager()).unload("never-loaded")

    def test_loading_over_budget_evicts_first(self) -> None:
        lease = GpuLeaseManager(budget_mb=1000)
        registry = ModelRegistry(lease)
        big, other = FakeSpec("big", 700), FakeSpec("other", 700)
        registry.register(big)
        registry.register(other)

        first = registry.get("big")
        registry.get("other")

        assert first.unloaded is True, "the first model must be evicted, not leaked"
        assert registry.loaded_keys == ("other",)
        assert lease.committed_mb == 700

    def test_reload_after_eviction_rebuilds(self) -> None:
        lease = GpuLeaseManager(budget_mb=1000)
        registry = ModelRegistry(lease)
        a, b = FakeSpec("a", 700), FakeSpec("b", 700)
        registry.register(a)
        registry.register(b)

        registry.get("a")
        registry.get("b")
        registry.get("a")

        assert a.load_count == 2

    def test_unload_all(self) -> None:
        lease = GpuLeaseManager(budget_mb=3000)
        registry = ModelRegistry(lease)
        for name in ("a", "b"):
            registry.register(FakeSpec(name, 100))
        models = [registry.get("a"), registry.get("b")]

        registry.unload_all()

        assert all(m.unloaded for m in models)
        assert registry.loaded_keys == ()
        assert lease.committed_mb == 0

    def test_lease_context_yields_the_model_and_a_receipt(self) -> None:
        registry = ModelRegistry(GpuLeaseManager(budget_mb=1000))
        registry.register(FakeSpec("a", 100))

        leased = registry.lease("a")
        with leased as model:
            assert isinstance(model, FakeModel)
        assert leased.receipt is not None
        assert leased.receipt.model == "a@1"
        assert leased.receipt.duration_ms >= 0

    def test_leasing_an_unknown_model_fails_before_acquiring(self) -> None:
        with pytest.raises(KeyError):
            ModelRegistry(GpuLeaseManager()).lease("nope")


# ---------------------------------------------------------------- reporting
class TestMetricsShapes:
    def test_vram_snapshot_serialises_for_job_metrics(self) -> None:
        payload = VramSnapshot(
            total_mb=4096.0, allocated_mb=360.5, reserved_mb=634.0, free_mb=3305.7
        ).as_dict()

        assert set(payload) == {
            "vram_total_mb",
            "vram_allocated_mb",
            "vram_reserved_mb",
            "vram_free_mb",
        }

    def test_lease_receipt_records_what_phase3_asked_for(self) -> None:
        payload = LeaseReceipt(
            model="clip:vit-b-32@1",
            device="cuda",
            vram_before_mb=0.0,
            vram_peak_mb=584.1,
            vram_after_mb=360.5,
            duration_ms=138.1,
            evicted=("faces:yunet",),
        ).as_dict()

        assert payload["model"] == "clip:vit-b-32@1"
        assert payload["device"] == "cuda"
        assert payload["vram_peak_mb"] == 584.1
        assert payload["evicted"] == ["faces:yunet"]
