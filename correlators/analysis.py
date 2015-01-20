#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014-2015 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

"""
Fragments for the analysis.
"""

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import logging

import numpy as np

import correlators.bootstrap
import correlators.fit
import correlators.loader
import correlators.plot
import correlators.scatlen
import correlators.transform


LOGGER = logging.getLogger(__name__)

ENSENBLE_DATA = {
    'A30.32': {
        'm_pi/f_pi_val' : 1.86,
        'm_pi/f_pi_err' : 0.0,
    },
    'A40.32': {
        'm_pi/f_pi_val' : 2.06,
        'm_pi/f_pi_err' : 0.01,
    },
    'A40.24': {
        'm_pi/f_pi_val' : 2.03,
        'm_pi/f_pi_err' : 0.03,
    },
    'A40.20': {
        'm_pi/f_pi_val' : 2.11,
        'm_pi/f_pi_err' : 0.05,
    },
    'D45.32': {
        'm_pi/f_pi_val' : 2.49,
        'm_pi/f_pi_err' : 0.0
    },
    'B55.32': {
        'm_pi/f_pi_val' : 2.34,
        'm_pi/f_pi_err' : 0.0,
    },
    'A60.24': {
        'm_pi/f_pi_val' : 2.32,
        'm_pi/f_pi_err' : 0.0,
    },
    'A80.24': {
        'm_pi/f_pi_val' : 2.55,
        'm_pi/f_pi_err' : 0.0,
    },
    'A100.24': {
        'm_pi/f_pi_val' : 2.77,
        'm_pi/f_pi_err' : 0.0,
    },
}
'List of ensembles used in arXiv:1412.0408v1'


def handle_path(path):
    '''
    Performs the analysis of all the files in the given folder.
    '''
    LOGGER.info('Working on path `%s`.', path)
    two_points, four_points, parameters = correlators.loader.folder_loader(path)

    name = parameters['path'].replace('/', '__')

    m_pi_f_pi_val = ENSENBLE_DATA[parameters['ensemble']]['m_pi/f_pi_val']
    m_pi_f_pi_err = ENSENBLE_DATA[parameters['ensemble']]['m_pi/f_pi_err']

    shift = int(parameters['T'])

    correlators.plot.plot_correlator(two_points, name+'_c2', shift)
    correlators.plot.plot_correlator(four_points, name+'_c4', shift, offset=True)
    correlators.plot.plot_effective_mass(two_points, name+'_c2')
    correlators.plot.plot_effective_mass(four_points, name+'_c4')

    # Combine the two lists of data into one list of lists. That way the
    # configurations are grouped together.
    combined = zip(two_points, four_points)

    val, err = correlators.bootstrap.bootstrap_pre_transform(
        mass_difference_decorator(shift),
        combined,
        correlators.bootstrap.average_combined_array,
    )

    results = {
        r'ensemble': parameters['ensemble'],
        r'm_2': (val[0], err[0]),
        r'm_4': (val[1], err[1]),
        r'\Delta E': (val[2], err[2]),
        r'a_0': (val[3], err[3]),
        r'amp_2': (val[4], err[4]),
        r'amp_4': (val[5], err[5]),
        r'offset_4': (val[6], err[6]),
        r'a0*m2': (val[7], err[7]),
        r'm2**2': (val[8], err[8]),
        r'm_pi/f_pi': (m_pi_f_pi_val, m_pi_f_pi_err),
    }

    return results


def mass_difference_decorator(shift):
    def mass_difference(params):
        # Unpack all the arguments from the list.
        (c2_val, c2_err), (c4_val, c4_err) = params

        # Generate a single time, they are all the same.
        time = np.array(range(len(c2_val)))

        # Perform the fits.
        fit2 = correlators.fit.cosh_fit_decorator(shift)
        p2 = correlators.fit.fit(fit2, time, c2_val, c2_err,
                                 omit_pre=13, p0=[0.222, c2_val[0]])
        fit4 = correlators.fit.cosh_fit_offset_decorator(shift)
        p4 = correlators.fit.fit(fit4, time, c4_val,
                                 c4_err, omit_pre=13, p0=[0.45, c2_val[0], 0])

        m2 = p2[0]
        m4 = p4[0]

        amp2 = p2[1]
        amp4 = p4[1]

        offset = p4[2]

        delta_m = m4 - 2 * m2

        a0 = correlators.scatlen.compute_a0(m2, m4, 24)

        return m2, m4, delta_m, a0, amp2, amp4, offset, a0*m2, m2**2

    return mass_difference
