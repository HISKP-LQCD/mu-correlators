#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014-2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

r'''
The value for the A100 data sets should be 0.22293(35)(38).

With the data from
``A100.24_L24_T48_beta190_mul0100_musig150_mudel190_kappa1632550/ev120/TB2_SO_LI6_new/C2_pi+-_conf????.dat``, I got :math:`0.22229 \pm 0.00003`
'''

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, \
    unicode_literals

import argparse
import logging

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

import correlators.traversal
import unitprint


def main():
    options = _parse_args()

    logging.basicConfig(level=logging.INFO)

    if options.plot_only:
        result = pd.read_csv('results.csv')
    else:
        result = correlators.traversal.handle_path(options.path).T
        print(result)
        result.to_csv('results.csv')

    plot_results(result)


def leading_order(x):
    # return - (x / (4 * np.pi))**2
    return - 0.0390625 * x**2


def plot_results(result):
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(result['m_pi/f_pi_val'], result['a0*m2_val'],
                xerr=result['m_pi/f_pi_err'], yerr=result['a0*m2_err'],
                linestyle='none', marker='+')

    lo_x = np.linspace(
        np.min(result['m_pi/f_pi_val']),
        np.max(result['m_pi/f_pi_val']),
        1000
    )
    lo_y = leading_order(lo_x)
    ax.plot(lo_x, lo_y)

    ax.margins(0.05, 0.05)
    ax.set_xlabel(r'$m_\pi / f_\pi$')
    ax.set_ylabel(r'$m_\pi a_0$')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('result.pdf')


def _parse_args():
    '''
    Parses the command line arguments.

    :return: Namespace with arguments.
    :rtype: Namespace
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path')
    parser.add_argument('--plot-only', action='store_true')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
