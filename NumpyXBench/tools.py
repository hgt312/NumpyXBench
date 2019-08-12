from .blobs import *
from . import operators
from .utils.mxnet_util import *


def test_mxnet_coverage():
    op_list = dir(operators)
    print(len(op_list), len(set(op_list)))
