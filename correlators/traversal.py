#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import logging
import os

import correlators.analysis


LOGGER = logging.getLogger(__name__)


def handle_path(path):
    '''
    Performs the analysis of every folder below the given path.
    '''
    all_results = {}
    for root, dirs, files in os.walk(path):
        # Skip all folders which contain the string `liuming` since they have a
        # different data format.
        if 'liuming' in root:
            continue


        if len(dirs) == 0:
            LOGGER.info('Found a leaf at `%s`.', root)

            if len(files) == 0:
                LOGGER.warning('Empty directory as `%s`.', root)
                continue

            try:
                all_results[root] = correlators.analysis.handle_path(root)
            except RuntimeError as e:
                LOGGER.error(str(e))
            except ValueError as e:
                LOGGER.error(str(e))

        else:
            dirs.sort()

    return all_results
