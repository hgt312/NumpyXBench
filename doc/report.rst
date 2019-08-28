Operator reports
================

The finished operator reports are listed below, only forward mode and only `float32` dtype now.

For each operator, there is a bar chart report, x-axis shows configs, y-axis shows speed rate with `NumPy`, \
value of `NumPy` is 1, and there is no bar if its backend is not implemented/supported. \
Some operators use a series of determined config, while others use random generated configs, \
hover the mouse over bars, you can see the detail config.

Binary operators
****************
.. toctree::
    :maxdepth: 2

    reports/add
    reports/subtract
    reports/multiply
    reports/divide
    reports/mod

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
