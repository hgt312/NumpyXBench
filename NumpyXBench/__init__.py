import os

try:
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    import mxnet
except Exception:
    print("Cannot use MXNet!")

try:
    # os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
except Exception:
    print("Cannot use jax!")

__version__ = '0.0.2'
