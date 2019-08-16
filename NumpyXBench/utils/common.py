from collections import defaultdict

backend_switcher = defaultdict(lambda: 'numpy')
backend_switcher.update({
    'numpy': 'numpy',  # numpy
    'np': 'numpy',
    'mxnet': 'mxnet.numpy',  # mxnet numpy
    'mx': 'mxnet.numpy',
    'mxnet.numpy': 'mxnet.numpy',
    'jax': 'jax.numpy',  # jax numpy
    'jax.numpy': 'jax.numpy',
    'pytorch': 'torch',
    'torch': 'torch',
})

AllTypes = ["float32", "float64", "float16", "uint8", "int8", "int32", "int64"]
RealTypes = ["float32", "float64", "float16"]
RealTypesNoHalf = ['float32', 'float64']
