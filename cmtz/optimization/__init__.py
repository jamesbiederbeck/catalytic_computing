"""IR optimization passes for Cook-Mertz v2."""

from .rotate_fusion import fuse_rotations
from .cse import eliminate_common_subexpressions
from .constant_fold import fold_constants
