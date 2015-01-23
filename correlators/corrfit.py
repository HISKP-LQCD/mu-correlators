#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2 (or later)

'''
Fitting correlated data with least squares.
'''

import numpy as np


def correlation_matrix(sets):
    r'''
    Computes the correlation matrix from a set of (multiple) time series.

    The input must have dimension 2. The first index shall label the
    measurement, the second one the time :math:`t`.

    The correlation matrix is given by:

    .. math::

        C_{ij} = \frac{1}{N[N-1]} \sum_{k=1}^N
        [x_{ik} - \bar x_{iN}] [x_{jk} - \bar x_{jN}]

    The indices are the other way around here. That is a simple matter of
    transposing the whole matrix.

    Using two samples of a time series with four elements, this function
    produces the following:

    >>> sets = [
    ...     [10, 8.4, 7.3, 5.1],
    ...     [10.5, 9.5, 6.3, 4.1],
    ... ]
    >>> correlation_matrix(sets)
    x
    [[ 10.    8.4   7.3   5.1]
     [ 10.5   9.5   6.3   4.1]]
    Average
    [ 10.25   8.95   6.8    4.6 ]
    x_i
    [[-0.25  0.25]
     [-0.55  0.55]
     [ 0.5  -0.5 ]
     [ 0.5  -0.5 ]]
    x_j
    [[-0.25 -0.55  0.5   0.5 ]
     [ 0.25  0.55 -0.5  -0.5 ]]
    Result
    [[ 0.0625  0.1375 -0.125  -0.125 ]
     [ 0.1375  0.3025 -0.275  -0.275 ]
     [-0.125  -0.275   0.25    0.25  ]
     [-0.125  -0.275   0.25    0.25  ]]

    :param np.array sets: All measurements of the time series.
    :returns: Correlation matrix
    :rtype: np.array
    '''
    N = len(sets)

    x = np.array(sets)
    print('x')
    print(x)

    average = np.mean(sets, axis=0)
    print('Average')
    print(average)

    xi = np.asmatrix(x - average).T
    xj = np.asmatrix(x - average)

    print('x_i')
    print(xi)
    print('x_j')
    print(xj)


    matrix = 1/(N*(N-1)) * xi * xj

    print('Result')
    print(matrix)

    return matrix


def main():
    sets = [
        [10, 8.4, 7.3, 5.1],
        [10.5, 9.5, 6.3, 4.1],
    ]

    cm = correlation_matrix(sets)


if __name__ == '__main__':
    main()
