"""Methods published by others. One module or folder per method.

Nothing in here is our contribution. See README.md in this folder.
"""
from rwm.prior_work import naive  # noqa: F401

try:  # needs the optional `trees` dependencies
    from rwm.prior_work import lightgbm_direct  # noqa: F401
except ImportError:
    pass
