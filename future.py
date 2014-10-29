#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, unicode_literals

# Standard library.
import argparse

# External libraries.
import matplotlib.pyplot as pl
import numpy as np

# This package.
import loader

def fold_data(data):
    r'''
    Folds the data around the middle element and averages.

    The expectation is to yield a :math:`\cosh` function. The transformation of
    the :math:`\{x_i\colon i = 1, \ldots, N\}` will generate new data points
    like this:

    .. math::

        y_i := \frac{x_i + x_{N-i+1}}2

    :param np.array data: Array with an even number of elements
    :returns: Folded array with :math:`N/2` elements
    :rtype: np.array
    '''
    n = len(data)
    first = data[:n//2]
    second = data[n//2:]
    return (first + second[::-1]) / 2

def main():
    options = _parse_args()

    for filename in options.filename:
        data = loader.correlator_loader(filename)
        print(data)

        real = np.real(data)

        pl.plot(real)
        pl.plot(fold_data(real))
        pl.show()


def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('filename', nargs='+')
    options = parser.parse_args()

    return options

if __name__ == '__main__':
    main()
