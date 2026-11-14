"""Keep job ownership after releasing inputs, and distinguish errors from cancellation."""
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.jobs as jobs
from app.cancellation import SimulationCancelled
from app.routers import runs


def test_input_released_at_start_and_owner_retained(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def execute(config, **kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(output={"status": "ok"}, warnings=[], warning_details=[])

    monkeypatch.setattr(jobs, "execute_simulation", execute)
    manager = jobs.RunManager()
    monkeypatch.setattr(runs, "get_run_manager", lambda: manager)
    try:
        job, _ = manager.submit({"config": {"large": [1]*10000}, "owner_key_id": "owner"})
        assert entered.wait(5)
        assert job.request == {} and job.owner_key_id == "owner"
        for status in ("running", "succeeded"):
            if status == "succeeded":
                release.set()
                job.future.result(timeout=5)
            for route in (runs.get_run_status, runs.get_run_result, runs.cancel_run):
                with pytest.raises(HTTPException) as error:
                    route(job.run_id, SimpleNamespace(scope={"vtsimnx_key_id": "other"}))
                assert error.value.status_code == 403
        assert job.status == "succeeded" and job.request == {}
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize("error, expected", [
    (RuntimeError("not cancelled: unrelated failure"), "failed"),
    (SimulationCancelled("requested stop"), "cancelled"),
])
def test_only_typed_cancellation_marks_job_cancelled(monkeypatch, error, expected):
    def execute(*args, **kwargs):
        raise error
    monkeypatch.setattr(jobs, "execute_simulation", execute)
    manager = jobs.RunManager()
    try:
        job, _ = manager.submit({"config": {"test": True}})
        job.future.result(timeout=5)
        assert job.status == expected
        assert job.request == {}
    finally:
        manager.shutdown()
