"""Private fresh-process refit entry point; configure limits before numerics."""

import os
import pickle
import sys
import traceback
from pathlib import Path


def main():
    job, output, threads, memory = sys.argv[1:]
    import stream_limits

    spill = Path(job).parent / "spill"
    spill.mkdir()
    stream_limits.set_limits(
        threads=int(threads), memory_gb=float(memory), spill_dir=spill, override=True
    )
    # Import this same checkout/wheel, including when invoked from a notebook
    # or from a working directory outside the package repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        # Let the package apply scheduler defaults before importing Polars.
        from diff_stream.bootstrap import _refit_attempt  # noqa: I001

        import polars as pl

        with open(job, "rb") as f:
            task = pickle.load(f)  # Private, locally written task; never external input.
        result = _refit_attempt(task)
        payload = dict(
            result=result,
            resources=dict(pid=os.getpid(), polars_threads=pl.thread_pool_size()),
        )
    except BaseException as error:
        # Preserve resource/programming exceptions instead of treating them
        # as statistical failures or returning a partial bootstrap result.
        try:
            pickle.dumps(error)
        except Exception:
            error = RuntimeError(f"{type(error).__name__}: {error}")
        payload = dict(error=error, traceback=traceback.format_exc())
    with open(output, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
