import os

os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

# os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "true"

__version__ = '0.0.5'
