#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014-2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import logging
import os

import pandas as pd

import correlators.analysis


LOGGER = logging.getLogger(__name__)


def handle_path(path):
    '''
    Performs the analysis of every folder below the given path.
    '''
    all_series = []
    for root, dirs, files in os.walk(path):
        # Skip all folders which contain the string `liuming` since they have a
        # different data format.
        if 'liuming' in root:
            del dirs[:]
            continue

        if 'Kpi' in root:
            del dirs[:]
            continue

        if len(dirs) == 0:
            LOGGER.info('Found a leaf at `%s`.', root)

            if len(files) == 0:
                LOGGER.warning('Empty directory as `%s`.', root)
                continue

            try:
                abspath = os.path.abspath(root)
                results = correlators.analysis.handle_path(root)
                all_series.append(results)
            except RuntimeError as e:
                LOGGER.error('RuntimeError: %s', str(e))
            #except ValueError as e:
                #LOGGER.error('ValueError: %s', str(e))

        else:
            dirs.sort()

    all_results = pd.DataFrame(all_series)
    return all_results
