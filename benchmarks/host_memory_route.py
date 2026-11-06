"""Record actual systemd memory-route behavior; run outside the execution sandbox.

Only short-lived services are created. The largest allocation is 128 MiB.
A successful systemd request is deliberately not classified as enforcement.
"""

import json
import os
import platform
import pwd
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/results/kernel_memory_checks"


def run(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return dict(command=command, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def main():
    account = pwd.getpwuid(os.getuid())
    code = (
        "import json,stream_limits; from pathlib import Path; "
        "print(json.dumps({'cgroup':Path('/proc/self/cgroup').read_text(),"
        "'detected_memory_gb':stream_limits.detect_memory_gb(environ={})}),flush=True); "
        "x=bytearray(128*1024*1024); print('allocation_succeeded',flush=True)"
    )
    common = [
        "--wait",
        "--pipe",
        "--collect",
        "--property=MemoryAccounting=yes",
        "--property=MemoryMax=64M",
        "--property=MemorySwapMax=0",
    ]
    user = run(
        [
            "systemd-run",
            "--user",
            f"--unit=diff-stream-route-user-{time.time_ns()}",
            *common,
            sys.executable,
            "-c",
            code,
        ]
    )
    system = run(
        [
            "systemd-run",
            "--no-ask-password",
            f"--unit=diff-stream-route-system-{time.time_ns()}",
            f"--property=User={account.pw_name}",
            *common,
            sys.executable,
            "-c",
            code,
        ]
    )
    report = dict(
        platform=platform.platform(),
        uid=os.getuid(),
        pid1=Path("/proc/1/comm").read_text().strip(),
        cgroup_mounts=[s for s in Path("/proc/mounts").read_text().splitlines() if " cgroup" in s],
        v2_controllers=Path("/sys/fs/cgroup/unified/cgroup.controllers").read_text().strip(),
        user_service=user,
        system_service=system,
        user_memory_limit_ignored=(
            user["returncode"] == 0 and "allocation_succeeded" in user["stdout"]
        ),
        requested_cap_bytes=64 * 2**20,
        allocation_bytes=128 * 2**20,
        note="Successful launch is not kernel enforcement. See cgroup_selfcheck.json for counter-based proof.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "host_routes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["pid1", "v2_controllers", "user_memory_limit_ignored"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
