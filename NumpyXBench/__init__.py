import os

__version__ = '0.0.1'

os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"
print("Set MXNet engine type to naive engine!")
