Operator reports
================

The finished operator reports are listed below.

For each operator, there is a bar chart report, x-axis shows configs, y-axis shows speedup rate with `NumPy`, \
value of `NumPy` is 1, and there is no bar if its backend is not implemented/supported. \
Some operators use a series of determined config, while others use random generated configs, \
hover the mouse over bars, you can see the detail config.

Note that for NumPy has no GPU support, the results in GPU version report are computed by CPU. \
In addition, the y axis of the chart is not start from 0 so that the bar won't be too short for hovering.

Binary operators
****************
.. toctree::
    :maxdepth: 2

    reports/add
    reports/subtract
    reports/multiply
    reports/divide
    reports/mod

Unary operators
****************
.. toctree::
    :maxdepth: 2

    reports/abs
    reports/arccos
    reports/arccosh
    reports/arcsin
    reports/arcsinh
    reports/arctan
    reports/arctanh
    reports/cbrt
    reports/ceil
    reports/cos
    reports/cosh
    reports/degrees
    reports/exp
    reports/expm1
    reports/fix
    reports/floor
    reports/log
    reports/log10
    reports/log1p
    reports/log2
    reports/logical_not
    reports/radians
    reports/reciprocal
    reports/rint
    reports/sign
    reports/sin
    reports/sinh
    reports/sqrt
    reports/square
    reports/tan
    reports/tanh
    reports/trunc

Creation operators
******************
.. toctree::
    :maxdepth: 2

    reports/empty
    reports/ones
    reports/zeros
    reports/ones_like
    reports/zeros_like
    reports/arange
    reports/linspace

Reduction operators
*************************************
.. toctree::
    :maxdepth: 2

    reports/sum
    reports/prod
