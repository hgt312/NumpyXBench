import os

try:
    os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
    import mxnet
    print("Set MXNet engine type to naive engine!")
except Exception:
    print("Cannot use MXNet!")

try:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    print("Enable x64 for JAX")
    # print(jax.numpy.ones((1,), dtype='float64').dtype)
except Exception:
    print("Cannot use jax!")

__version__ = '0.0.2'
