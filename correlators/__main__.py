#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
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


def merge_dicts(a, b):
    """
    From: http://stackoverflow.com/a/38990
    """
    return dict(a.items() + b.items())


def present_result_dict(result):
    print()
    print('Results')
    print('=======')

    for path, quantities in result.iteritems():
        print()
        print(path)
        print()
        print('{:_^15s}  {:_^15s}  {:_^15s}  {:_^20s}'.format(
            'Name', 'Value', 'Error', 'Value+Error'
        ))
        for name, (val, err) in sorted(quantities.iteritems()):
            print('{:15s}  {:15g}  {:15g}  {:^20s}'.format(
                name, val, err, unitprint.siunitx(val, err)
            ))


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
