"""Small standard-library process batch, intended for eventual shared core.

Threads only wait on child processes; numerical work belongs in the children.
On failure or interruption, stop and reap children before consumers delete
their scratch. Commands are argument vectors, never shell strings.
"""

import subprocess
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Lock


class ProcessBatch:
    def __init__(self, workers):
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._lock = Lock()
        self._children = set()
        self._closing = False

    def __enter__(self):
        return self

    def submit(self, command, log):
        return self._executor.submit(self._run, command, log)

    def _run(self, command, log):
        with open(log, "wb") as output:
            with self._lock:
                if self._closing:
                    raise CancelledError()
                child = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT
                )
                self._children.add(child)
            try:
                return child.wait()
            finally:
                with self._lock:
                    self._children.discard(child)

    def __exit__(self, exc_type, exc, tb):
        with self._lock:
            self._closing = True
            children = list(self._children)
        # On normal exit all submitted futures have already completed. A
        # consumer exception/interrupt must not leave numerical fits running.
        for child in children:
            try:
                child.terminate()
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass  # Exited between the timeout and kill.
                child.wait()
        self._executor.shutdown(wait=True, cancel_futures=True)
