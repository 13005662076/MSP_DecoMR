import math
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Rel, Validator
from mindspore import nn
from mindspore.ops.primitive import constexpr, Primitive

@constexpr
def bilinear(shape, size, scale, align_corners):
    """Check input and calculate shape"""
    
    if not isinstance(align_corners, bool):
        raise TypeError("align_corners should be type boolean")
    if size is None and scale is None:
        raise ValueError("size and scale both none")
    if size is not None and scale is not None:
        raise ValueError("size and scale both not none")
    if size is not None:
        if not isinstance(size, (tuple, list)):
            raise ValueError("size must be tuple or list")
        Validator.check_int(len(size), 2, Rel.EQ, "size", "bilinear")
        Validator.check_int(size[0], 1, Rel.GE, "size[0]", "bilinear")
        Validator.check_int(size[1], 1, Rel.GE, "size[1]", "bilinear")
        return size
    
    Validator.check_int(scale, 1, Rel.GE, "scale factor", "bilinear")
    ret = (scale * shape[2], scale * shape[3])
    return ret


class ResizeBilinear(nn.Cell):
    def __init__(self,size=None, scale_factor=None, align_corners=False):
        super(ResizeBilinear, self).__init__()
        self.size=size
        self.scale_factor=scale_factor
        self.align_corners=align_corners

    def construct(self, x):
        shape = bilinear(x.shape, self.size, self.scale_factor, self.align_corners)
        resize_bilinear = P.ResizeBilinear(shape, self.align_corners)
        return resize_bilinear(x)
