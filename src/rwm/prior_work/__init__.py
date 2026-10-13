"""Methods published by others. One module or folder per method.

Nothing in here is our contribution. See README.md in this folder.
"""
from rwm.prior_work import naive  # noqa: F401
from rwm.prior_work import timesfm_zero_shot  # noqa: F401  (imports timesfm only when used)

try:  # needs the optional `trees` dependencies
    from rwm.prior_work import lightgbm_direct  # noqa: F401
    from rwm.prior_work import graph_direct  # noqa: F401  (also needs torch, imported only when used)
except ImportError:
    pass
from rwm.prior_work import chronos_zero_shot  # noqa: F401  (imports chronos only when used)
from rwm.prior_work import tft  # noqa: F401  (imports neuralforecast only when used)
