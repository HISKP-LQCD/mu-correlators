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

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

import correlators.bootstrap
import correlators.corrfit
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

    T = int(parameters['T'])
    L = int(parameters['L'])

    # Combine the two lists of data into one list of lists. That way the
    # configurations are grouped together and will stay together in
    # bootstrapping.
    combined_correlators = zip(two_points, four_points)

    analysis_function = analyze_factory(T, L)

    val, err = correlators.bootstrap.bootstrap_pre_transform(
        analysis_function,
        combined_correlators,
    )

    p0_2 = [val[0], val[4]]
    p0_4 = [val[1], val[5], val[6]]

    corr_fit_param, corr_fit_err = correlators.bootstrap.bootstrap_pre_transform(
        mass_difference_correlated_decorator(T, L, p0_2, p0_4),
        combined
    )

    print(corr_fit_param, corr_fit_err)

    series = pd.Series({
        'm_2_val': val[0],
        'm_2_err': err[0],
        'm_4_val': val[1],
        'm_4_err': err[1],
        'Delta E_val': val[2],
        'Delta E_err': err[2],
        'a_0_val': val[3],
        'a_0_err': err[3],
        'amp_2_val': val[4],
        'amp_2_err': err[4],
        'amp_4_val': val[5],
        'amp_4_err': err[5],
        'offset_4_val': val[6],
        'offset_4_err': err[6],
        'a0*m2_val': val[7],
        'a0*m2_err': err[7],
        'm2**2_val': val[8],
        'm2**2_err': err[8],
        'corr__m_2_val': corr_fit_param[0],
        'corr__m_2_err': corr_fit_err[0],
        'corr__m_4_val': corr_fit_param[1],
        'corr__m_4_err': corr_fit_err[1],
        'corr__Delta E_val': corr_fit_param[2],
        'corr__Delta E_err': corr_fit_err[2],
        'corr__a_0_val': corr_fit_param[3],
        'corr__a_0_err': corr_fit_err[3],
        'corr__amp_2_val': corr_fit_param[4],
        'corr__amp_2_err': corr_fit_err[4],
        'corr__amp_4_val': corr_fit_param[5],
        'corr__amp_4_err': corr_fit_err[5],
        'corr__offset_4_val': corr_fit_param[6],
        'corr__offset_4_err': corr_fit_err[6],
        'corr__a0*m2_val': corr_fit_param[7],
        'corr__a0*m2_err': corr_fit_err[7],
        'corr__m2**2_val': corr_fit_param[8],
        'corr__m2**2_err': corr_fit_err[8],
        'corr__chi_sq_2_val': corr_fit_param[9],
        'corr__chi_sq_2_err': corr_fit_err[9],
        'corr__chi_sq_4_val': corr_fit_param[10],
        'corr__chi_sq_4_err': corr_fit_err[10],
        'corr__p_value_2_val': corr_fit_param[11],
        'corr__p_value_2_err': corr_fit_err[11],
        'corr__p_value_4_val': corr_fit_param[12],
        'corr__p_value_4_err': corr_fit_err[12],
        'm_pi/f_pi_val': m_pi_f_pi_val,
        'm_pi/f_pi_err': m_pi_f_pi_err,
        'L': parameters['L'],
        'T': parameters['T'],
    })

    correlators.plot.plot_correlator(two_points, name+'_c2', T)
    correlators.plot.plot_correlator(four_points, name+'_c4', T, offset=True)
    correlators.plot.plot_effective_mass(two_points, name+'_c2')
    correlators.plot.plot_effective_mass(four_points, name+'_c4')

    return parameters['ensemble'], series

def analyze(sets, T, L):
    params = correlators.bootstrap.average_combined_array(sets)
    # Unpack all the arguments from the list.
    (c2_val, c2_err), (c4_val, c4_err) = params

    # Generate a single time, they are all the same.
    time = np.array(range(len(c2_val)))

    # Perform the fits.
    fit2 = correlators.fit.cosh_fit_decorator(T)
    p2 = correlators.fit.fit(fit2, time, c2_val, c2_err,
                             omit_pre=13, p0=[0.222, c2_val[0]])
    fit4 = correlators.fit.cosh_fit_offset_decorator(T)
    p4 = correlators.fit.fit(fit4, time, c4_val,
                             c4_err, omit_pre=13, p0=[0.45, c2_val[0], 0])

    m2 = p2[0]
    m4 = p4[0]

    amp2 = p2[1]
    amp4 = p4[1]

    offset = p4[2]

    delta_m = m4 - 2 * m2

    a0 = correlators.scatlen.compute_a0(m2, m4, L, fig)

    return m2, m4, delta_m, a0, amp2, amp4, offset, a0*m2, m2**2

    sets2, sets4 = zip(*sets)

    sets2 = np.array(sets2)
    sets4 = np.array(sets4)

    # Generate a single time, they are all the same.
    time = np.array(range(len(sets2[0])))

    # Perform the fits.
    fit2 = correlators.fit.cosh_fit_decorator(T)
    p2, chi_sq_2, p_value_2 = correlators.corrfit.fit(
        fit2, time, sets2, omit_pre=13, p0=p0_2)
    fit4 = correlators.fit.cosh_fit_offset_decorator(T)
    p4, chi_sq_4, p_value_4 = correlators.corrfit.fit(
        fit4, time, sets4, omit_pre=13, p0=p0_4)


    m2 = p2[0]
    m4 = p4[0]

    amp2 = p2[1]
    amp4 = p4[1]

    offset = p4[2]

    delta_m = m4 - 2 * m2

    a0 = correlators.scatlen.compute_a0(m2, m4, L, fig)

    return m2, m4, delta_m, a0, amp2, amp4, offset, a0*m2, m2**2, \
            chi_sq_2, chi_sq_4, p_value_2, p_value_4


def analyze_factory(T, L):
    '''
    Generates an :py:`analyse` function that already has the values for T and L
    fixed.
    '''
    def analyze_curried(sets):
        analyse(sets, T, L)

    return analyze_curried
