#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

'''
Helper functions to load the binary data that I was given from
``/hiskp2/correlators/``.
'''

from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np

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

def average_loader(filenames):
    sets = np.array(list(loader_iterator(filenames)))
    total = np.column_stack(sets)

    val = np.real(np.mean(total, axis=1))
    err = np.real(np.std(total, axis=1))

    return val, err
