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
    all_results = pd.DataFrame()
    for root, dirs, files in os.walk(path):
        # Skip all folders which contain certains strings since they have a
        # different data format.
        if any([flee in root for flee in ['liuming', 'Kpi']]):
            del dirs[:]
            continue

        if len(dirs) == 0:
            LOGGER.info('Found a leaf at `%s`.', root)

            if len(files) == 0:
                LOGGER.warning('Empty directory as `%s`.', root)
                continue

            abspath = os.path.abspath(root)
            ensemble, results = correlators.analysis.handle_path(root)
            all_results[ensemble] = results

        else:
            dirs.sort()

    return all_results
