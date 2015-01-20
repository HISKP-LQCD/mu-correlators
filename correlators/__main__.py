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

import unitprint
import matplotlib.pyplot as pl

import correlators.traversal


def main():
    options = _parse_args()

    logging.basicConfig(level=logging.INFO)

    result_dicts = []

    for path in options.path:
        result = correlators.traversal.handle_path(path)
        result_dicts.append(result)

    result_dict = reduce(merge_dicts, result_dicts)
    present_result_dict(result_dict)

    plot_results(result)


def merge_dicts(a, b):
    """
    From: http://stackoverflow.com/a/38990
    """
    return dict(a.items() + b.items())


def present_result_dict(result):
    print()
    print('Results')
    print('=======')

    for path, quantities in sorted(result.iteritems()):
        print()
        print(path)
        print()
        print('{:_^15s}  {:_^15s}  {:_^15s}  {:_^20s}'.format(
            'Name', 'Value', 'Error', 'Value+Error'
        ))
        for name, item in sorted(quantities.iteritems()):
            if isinstance(item, tuple):
                val, err = item
                print('{:15s}  {:15g}  {:15g}  {:^20s}'.format(
                    name, val, err, unitprint.siunitx(val, err)
                ))
            else:
                print('{:15s}  {:15}'.format(
                    name, item,
                ))


def plot_results(result):
    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []
    xerr = []
    yerr = []

    for q in result.itervalues():
        x.append(q['m_pi/f_pi'][0])
        y.append(q['a0*m2'][0])
        xerr.append(q['m_pi/f_pi'][1])
        yerr.append(q['a0*m2'][1])

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle='none', marker='+')
    ax.margins(0.05, 0.05)
    ax.set_xlabel(r'$m_\pi / f_\pi$')
    ax.set_ylabel(r'$m_\pi \cdot a_0$')
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
    parser.add_argument('path', nargs='+')
    options = parser.parse_args()

    return options


if __name__ == '__main__':
    main()
