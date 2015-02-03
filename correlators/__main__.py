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
import itertools

import colorsys
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

import correlators.traversal
import unitprint

COLORMAPPER_9 = [
    '#e41a1c',
    '#377eb8',
    '#4daf4a',
    '#984ea3',
    '#ff7f00',
    '#ffff33',
    '#a65628',
    '#f781bf',
    '#999999',
]

def coloriter(n):
    return itertools.cycle([
        colorsys.hsv_to_rgb(x*1.0/n, .9, .9)
        for x in range(n)
    ])


def main():
    options = _parse_args()

    logging.basicConfig(level=logging.INFO)

    if options.plot_only:
        result = pd.read_csv('results.csv')
    else:
        result = correlators.traversal.handle_path(options)
        pd.set_option('display.max_columns', None)
        print(result)
        result.to_csv('results.csv')

    plot_results(result)


def leading_order(x):
    return - x**2 / (8 * np.pi)


def plot_results(result):
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)

    result = result.sort(['m_pi/f_pi_val'], ascending=[True])

    color_iter = coloriter(6)
    for ensemble, data in result.T.iteritems():
        color = next(color_iter)
        ax.errorbar(
            data['m_pi/f_pi_val'], data['a_0*m_2:1val'],
            xerr=data['m_pi/f_pi_err'], yerr=data['a_0*m_2:3err'],
            linestyle='none', marker='+', label=data[0], color=color,
        )
        ax.errorbar(
            data['m_pi/f_pi_val']+0.005, data['a0*m_pi_paper_val'],
            xerr=data['m_pi/f_pi_err'], yerr=data['a0*m_pi_paper_err'],
            linestyle='none', marker='d', color=color,
        )

    lo_x = np.linspace(
        np.min(result['m_pi/f_pi_val']),
        np.max(result['m_pi/f_pi_val']),
        1000
    )
    lo_y = leading_order(lo_x)
    ax.plot(lo_x, lo_y, color='black')

    ax.margins(0.05, 0.05)
    ax.set_xlabel(r'$m_\pi / f_\pi$')
    ax.set_ylabel(r'$m_\pi a_0$')
    ax.set_ylim([-0.35, -0.05])
    ax.grid(True)

    ax.legend(loc='best', prop={'size':9})

    #fig.show()
    #raw_input()

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
    parser.add_argument('--corr-fit', action='store_true', help='Perform a correlated fit')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
