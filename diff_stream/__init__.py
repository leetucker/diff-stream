"""Disk-backed staggered difference-in-differences and TWFE diagnostics."""

import stream_limits as _limits

_limits.apply_scheduler_defaults()

from hdfe_stream.core.profile import Profile  # noqa: E402
from hdfe_stream.core.reporting import etable  # noqa: E402
from stream_limits import limit_threads  # noqa: E402

from .api import bjs_stream, cs_stream, diff_stream, sunab_stream  # noqa: E402
from .bacon import BaconResult, bacon_decomp  # noqa: E402
from .bjs_placebo import BJSPlaceboResult  # noqa: E402
from .bootstrap import BootstrapFailureWarning, RefitBootstrapResult, refit_bootstrap  # noqa: E402
from .effective_sample import LowEffectiveSampleWarning  # noqa: E402
from .etwfe import etwfe_stream  # noqa: E402
from .gardner import GardnerResult, gardner_stream  # noqa: E402
from .reference import ReferenceWindowWarning  # noqa: E402
from .results import DIDResult  # noqa: E402
from .twfe_weights import TWFEWeightResult, twfe_weights  # noqa: E402

__version__ = "0.1.0.dev0"
__all__ = [
    "diff_stream",
    "bjs_stream",
    "sunab_stream",
    "cs_stream",
    "gardner_stream",
    "etwfe_stream",
    "GardnerResult",
    "DIDResult",
    "bacon_decomp",
    "twfe_weights",
    "refit_bootstrap",
    "RefitBootstrapResult",
    "BootstrapFailureWarning",
    "TWFEWeightResult",
    "BaconResult",
    "BJSPlaceboResult",
    "LowEffectiveSampleWarning",
    "ReferenceWindowWarning",
    "Profile",
    "etable",
    "limit_threads",
]
