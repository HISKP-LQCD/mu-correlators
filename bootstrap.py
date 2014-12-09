#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, unicode_literals

import random

import numpy as np


def average_arrays(arrays):
    '''
    Computes the element wise average of a list of arrays.
    '''
    total = np.column_stack(arrays)

    val = np.real(np.mean(total, axis=1))

    return val


def average_and_std_arrays(arrays):
    '''
    Computes the element wise average of a list of arrays.
    '''
    total = np.column_stack(arrays)

    val = np.real(np.mean(total, axis=1))
    err = np.real(np.std(total, axis=1)) #/ np.sqrt(len(arrays))

    return val, err


def bootstrap_pre_transform(transform, sets, reduction=average_arrays,
                            sample_count=250, seed=None):
    '''
    Bootstraps the sets, reduces them to a single set and transforms them.

    This is the recommended method!

    The return value of the function is assumed to be a one dimensional NumPy
    array. The return value of this function is one array with the values and
    another with the errors.
    '''
    random.seed(seed)

    results = []
    for sample_id in xrange(sample_count):
        sample = generate_sample(sets)
        argument = reduction(sample)
        results.append(transform(argument))

    val, err = average_and_std_arrays(results)

    return val, err


def bootstrap_post_transform(transform, sets, reduction=average_arrays,
                             sample_count=250, seed=None):
    '''
    Applies the transformation to each set and bootstraps the results.

    The return value of the function is assumed to be a one dimensional NumPy
    array. The return value of this function is one array with the values and
    another with the errors.
    '''
    random.seed(seed)

    transformed_sets = map(transform, sets)

    results = []
    for sample_id in xrange(sample_count):
        sample = generate_sample(transformed_sets)
        results.append(reduction(sample))

    val, err = average_and_std_arrays(results)

    return val, err


def generate_sample(elements):
    '''
    Generates a sample from the given list.

    The number of elements in the sample is taken to be the same as the number
    of elements given.
    '''
    result = []
    for i in xrange(len(elements)):
        result.append(random.choice(elements))

    return result

def average_combined_array(combined):
    '''
    Given a list of tuples or arrays of the kind “configuration → n-point →
    correlator“, it creates the average of the correlator over all the
    configurations for each “n-point”.

    It will then return the averaged out correlator for the two- and for the
    four-point correlation function.

    The input list is converted into the structure “n-point → configuration →
    correlator“ using zip(). Then the function average_arrays() is used on each
    element of the outer list.
    '''
    result = []
    npoints = zip(*combined)
    for npoint in npoints:
        result.append(average_and_std_arrays(npoints))

    return result
