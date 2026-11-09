"""Importing diff_stream sets the OpenMP wait policy, and leaves the user's alone."""

import os
import subprocess
import sys

import pytest

stream_limits = pytest.importorskip("stream_limits")
pytestmark = pytest.mark.skipif(
    not hasattr(stream_limits, "set_openmp_wait_policy"), reason="needs stream-limits 0.3"
)


def _policy(**env):
    e = {k: v for k, v in os.environ.items() if k != "OMP_WAIT_POLICY"}
    e.update(env)
    code = "import os, diff_stream; print(os.environ.get('OMP_WAIT_POLICY'))"
    r = subprocess.run(
        [sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_import_sets_passive():
    assert _policy() == "passive"


def test_a_policy_set_by_the_user_is_kept():
    assert _policy(OMP_WAIT_POLICY="active") == "active"
