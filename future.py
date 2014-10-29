#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, unicode_literals

import argparse

import loader

def main():
    options = _parse_args()

    for filename in options.filename:
        data = loader.correlator_loader(filename)
        print(data)


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
