# SPDX-License-Identifier: Apache-2.0
"""Target distribution factories."""

from .m0_categorical import categorical_target
from .m1_bigram import bigram_target

__all__ = ["categorical_target", "bigram_target"]

