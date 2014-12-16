#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import os


def handle_path(path):
    '''
    Performs the analysis of every folder below the given path.
    '''
    for root, dirs, files in os.walk(path):
        if len(dirs) == 0:
            print('Found a leaf at', root)
