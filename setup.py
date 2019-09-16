from setuptools import setup, find_packages
from NumpyXBench import __version__

requirements = [
    'numpy',
    'ConfigSpace',
    'Jinja2',
    'bokeh==1.3.4'
]

setup(
    name='NumpyXBench',
    version=__version__,
    install_requires=requirements,
    python_requires='>=3.5',
    author='hgt312',
    author_email='hgt312@foxmail.com',
    description="Benchmark for NumPy Compatible Operators",
    packages=find_packages(),
)
