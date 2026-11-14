"""Recover read-only polling without duplicating job submissions."""
import json

import pytest
import requests

from vtsimnx.run_calc import wait_for_run, RunCalcAPIError
from vtsimnx.run_calc import _http


def response(status, body):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r._content_consumed = True
    return r


class Session:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def get(self, url, **kwargs):
        self.calls.append(url)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_submission_not_repeated_after_transient_poll_error(monkeypatch):
    session = Session([requests.ConnectionError("temporary disconnect"),
                       response(503, {}), response(200, {"status": "succeeded"}),
                       response(502, {}), response(200, {"result": {"status": "ok"}})])
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)
    posts = []
    def submit(*args, **kwargs):
        posts.append(args)
        return response(202, {"run_id": "existing", "status": "queued"}), {}
    monkeypatch.setattr(_http, "_post_json", submit)
    result = _http._submit_and_wait("http://test", payload={}, compress_request=False, timeout=10)
    assert result["result"]["status"] == "ok"
    assert len(posts) == 1 and len(session.calls) == 5 and session.closed


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_permanent_status_error_is_not_retried_and_retains_id(monkeypatch, status):
    session = Session([response(status, {"error": {"code": "denied"}})])
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    with pytest.raises(RunCalcAPIError) as error:
        wait_for_run("http://test", "saved-run")
    assert error.value.status_code == status
    assert error.value.detail["run_id"] == "saved-run"
    assert len(session.calls) == 1 and session.closed


def test_retry_exhaustion_preserves_id_and_cause(monkeypatch):
    session = Session([requests.ConnectionError("down")] * 4)
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    monkeypatch.setattr(_http.time, "sleep", lambda _: None)
    with pytest.raises(RunCalcAPIError) as error:
        wait_for_run("http://test", "saved-run")
    assert error.value.detail["run_id"] == "saved-run"
    assert isinstance(error.value.__cause__, requests.ConnectionError)
    assert len(session.calls) == 4 and session.closed


def test_retry_respects_overall_deadline(monkeypatch):
    now = [0.0]
    session = Session([requests.Timeout("read timeout")]*4)
    monkeypatch.setattr(_http.requests, "Session", lambda: session)
    monkeypatch.setattr(_http.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_http.time, "sleep", lambda seconds: now.__setitem__(0, now[0]+seconds))
    with pytest.raises(RunCalcAPIError) as error:
        wait_for_run("http://test", "saved-run", timeout=0.1)
    assert error.value.detail == {"code": "timeout", "run_id": "saved-run"}
    assert now[0] == 0.1 and len(session.calls) == 1
