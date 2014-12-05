#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

'''
Helper functions to load the binary data that I was given from
``/hiskp2/correlators/``.
'''

from __future__ import division, absolute_import, print_function, unicode_literals

import re

import numpy as np


TWO_PATTERN = re.compile(r'C2_pi\+-_conf(\d{4}).dat')
FOUR_PATTERN = re.compile(r'C4_(\d)_conf(\d{4}).dat')


def correlator_loader(filename):
    '''
    Loads binary correlator files.
    
    Such a file would be
    ``/hiskp2/correlators/A100.24_L24_T48_beta190_mul0100_musig150_mudel190_kappa1632550/ev120/TB2_SO_LI6_new/C2_pi+-_conf0501.dat``.

    It is assumed that the data in the files are only *little endian 8-byte
    float* numbers, real and imaginary part right after each other.

    :param str filename: Path to the binary file
    :returns: NumPy array with complex numbers
    :rtype: np.array
    '''
    dtype = np.dtype(np.complex128)
    data = np.fromfile(filename, dtype)

    return data


def loader_iterator(filenames):
    '''
    Iterator that gives the data to the given filenames.

    :param list filenames: List of filenames (str)
    '''
    for filename in filenames:
        data = correlator_loader(filename)
        yield data


def list_loader(filenames):
    return list(loader_iterator(filenames))


def folded_list_loader(filenames):
    return [fold_data(data) for data in loader_iterator(filenames)]


def average_loader(filenames):
    '''
    Loads multiple files of the same size and averages over them, also creating
    an error estimate.

    The files are loaded with loader_iterator().

    :param list filenames: List of filenames
    :returns: Value and error as a arrays
    :rtype: tuple of np.array
    '''
    sets = np.array(list(loader_iterator(filenames)))
    total = np.column_stack(sets)

    val = np.real(np.mean(total, axis=1))
    err = np.real(np.std(total, axis=1)) / np.sqrt(len(filenames))

    return val, err


def fold_data(val):
    r'''
    Folds the data around the middle element and averages.

    The expectation is to yield a :math:`\cosh` function. The transformation of
    the :math:`\{x_i\colon i = 1, \ldots, N\}` will generate new data points
    like this:

    .. math::

        y_i := \frac{x_i + x_{N-i}}2

    :param np.array val: Array with an even number of elements, values
    :param np.array err: Array with an even number of elements, errors
    :returns: Folded array with :math:`N/2` elements
    :rtype: np.array
    '''
    n = len(val)
    second_rev_val = val[n//2+1:][::-1]
    first_val = val[:n//2+1]
    first_val[1:-1] += second_rev_val
    first_val[1:-1] /= 2.

    return first_val


def fold_data_with_error(val, err):
    r'''
    Folds the data around the middle element and averages.

    The expectation is to yield a :math:`\cosh` function. The transformation of
    the :math:`\{x_i\colon i = 1, \ldots, N\}` will generate new data points
    like this:

    .. math::

        y_i := \frac{x_i + x_{N-i}}2

    :param np.array val: Array with an even number of elements, values
    :param np.array err: Array with an even number of elements, errors
    :returns: Folded array with :math:`N/2` elements
    :rtype: tuple of np.array
    '''
    n = len(val)
    second_rev_val = val[n//2+1:][::-1]
    first_val = val[:n//2+1]
    first_val[1:-1] += second_rev_val
    first_val[1:-1] /= 2.

    second_rev_err = err[n//2+1:][::-1]
    first_err = err[:n//2+1]
    first_err[1:-1] = np.sqrt(first_err[1:-1]**2 + second_rev_err**2) / 2

    return first_val, first_err
