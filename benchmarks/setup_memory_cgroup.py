"""One-time privileged delegation of a capped, empty v1 test parent.

Run from an ordinary host terminal with sudo. Estimation stays unprivileged;
only this script creates the fixed-purpose parent and delegates child creation.
It never moves a process, changes the hierarchy, or changes another group's cap.
"""

import argparse
import os
import pwd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("requires sudo from your host terminal; no password is read by this script")
    account = pwd.getpwnam(args.user)
    root = Path("/sys/fs/cgroup/memory")
    if not (root / "memory.limit_in_bytes").exists():
        parser.error("this setup command is for the host's cgroup v1 memory controller")
    parent = root / f"diff-stream-tests-{account.pw_uid}"
    if args.remove:
        parent.rmdir()  # Refuse if a test child or process still exists.
        print(f"Removed {parent}")
        return
    parent.mkdir()  # Refuse to change an existing group.
    try:
        (parent / "memory.use_hierarchy").write_text("1")
        (parent / "memory.limit_in_bytes").write_text(str(3 * 2**30))
        (parent / "memory.memsw.limit_in_bytes").write_text(str(3 * 2**30))
        (parent / "memory.swappiness").write_text("0")
        (parent / "memory.oom_control").write_text("0")
        # Delegate child creation and attachment, but leave the parent's hard
        # limit files owned by root. All descendants remain under its 3 GiB cap.
        os.chown(parent, account.pw_uid, account.pw_gid)
        os.chown(parent / "cgroup.procs", account.pw_uid, account.pw_gid)
        os.chown(parent / "tasks", account.pw_uid, account.pw_gid)
    except BaseException:
        parent.rmdir()
        raise
    print(f"Delegated {parent} to {account.pw_name}; parent limit 3 GiB, swappiness 0.")
    print("Run the self-check and fits as your ordinary user, outside the Codex sandbox.")


if __name__ == "__main__":
    main()
