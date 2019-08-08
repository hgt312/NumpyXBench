import os

try:
    import mxnet
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    print("Set MXNet engine type to naive engine!")
except ImportError:
    print("Cannot use MXNet!")

try:
    import torch
except ImportError:
    print("Cannot use pytorch!")

__version__ = '0.0.1'


