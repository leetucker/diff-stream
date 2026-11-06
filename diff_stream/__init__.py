"""Disk-backed staggered difference-in-differences and TWFE diagnostics."""

import stream_limits as _limits

_limits.apply_scheduler_defaults()

from hdfe_stream.core.profile import Profile  # noqa: E402
from hdfe_stream.core.reporting import etable  # noqa: E402
from stream_limits import limit_threads  # noqa: E402

from .absorbed import UnimputedObservationsWarning  # noqa: E402
from .api import did_bjs, did_callaway_santanna, did_sun_abraham  # noqa: E402
from .bacon import BaconResult, bacon_decomp  # noqa: E402
from .bjs_placebo import BJSPlaceboResult  # noqa: E402
from .bootstrap import BootstrapFailureWarning, RefitBootstrapResult, refit_bootstrap  # noqa: E402
from .effective_sample import LowEffectiveSampleWarning  # noqa: E402
from .etwfe import did_extended_twfe  # noqa: E402
from .gardner import TwoStageResult, did_two_stage  # noqa: E402
from .reference import ReferenceWindowWarning  # noqa: E402
from .results import DIDResult  # noqa: E402
from .stacked import StackedResult, did_stacked  # noqa: E402
from .twfe_weights import TWFEWeightResult, twfe_weights  # noqa: E402

__version__ = "0.1.0.dev0"
__all__ = [
    "did_callaway_santanna",
    "did_sun_abraham",
    "did_bjs",
    "did_two_stage",
    "did_extended_twfe",
    "did_stacked",
    "bacon_decomp",
    "twfe_weights",
    "refit_bootstrap",
    "DIDResult",
    "TwoStageResult",
    "StackedResult",
    "BJSPlaceboResult",
    "BaconResult",
    "TWFEWeightResult",
    "RefitBootstrapResult",
    "BootstrapFailureWarning",
    "LowEffectiveSampleWarning",
    "ReferenceWindowWarning",
    "UnimputedObservationsWarning",
    "Profile",
    "etable",
    "limit_threads",
]
