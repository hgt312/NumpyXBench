import os

try:
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    import mxnet
    print("Set MXNet engine type to naive engine!")
except Exception:
    print("Cannot use MXNet!")

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
except Exception:
    print("Cannot use jax!")

try:
    import torch
except Exception:
    print("Cannot use pytorch!")

__version__ = '0.0.2'
