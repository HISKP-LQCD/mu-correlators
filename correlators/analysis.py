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
import sys

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.optimize as op
import scipy.stats

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
    orig_correlators = zip(two_points, four_points)
    sample_count = 3 * len(orig_correlators)
    boot_correlators = correlators.bootstrap.generate_reduced_samples(
        orig_correlators, sample_count,
    )

    # Extract the corrleator values to compute correlation matrices.
    correlators_2_val = unwrap_correlator_values(boot_correlators, 0)
    correlators_4_val = unwrap_correlator_values(boot_correlators, 1)
    correlators_2_err = unwrap_correlator_errors(boot_correlators, 0)
    correlators_4_err = unwrap_correlator_errors(boot_correlators, 1)

    # Compute the correlation matrices from the bootstrap samples only.
    corr_matrix_2 = correlators.corrfit.correlation_matrix(correlators_2_val)
    corr_matrix_4 = correlators.corrfit.correlation_matrix(correlators_4_val)

    omit_pre = 13

    inv_corr_mat_2 = corr_matrix_2[omit_pre:, omit_pre:].getI()
    inv_corr_mat_4 = corr_matrix_4[omit_pre:, omit_pre:].getI()

    # Generate a single time, they are all the same.
    time = np.array(range(len(correlators_2_val[0])))

    for sample_id in range(sample_count):
        m_2, p_value_2 = perform_fits(
            time, correlators_2_val[sample_id], correlators_2_err[sample_id],
            inv_corr_mat_2, correlators.fit.cosh_fit_decorator,
            [0.222, correlators_2_val[sample_id][0]], T, omit_pre
        )

        m_4, p_value_4 = perform_fits(
            time, correlators_4_val[sample_id], correlators_4_err[sample_id],
            inv_corr_mat_4, correlators.fit.cosh_fit_offset_decorator,
            [0.222, correlators_2_val[sample_id][0], 0], T, omit_pre
        )

        delta_m = m_4 - 2 * m_2

        a_0 = correlators.scatlen.compute_a0(m_2, m_4, L)

        print(m_2, m_4)

    series = pd.Series({
        'm_pi/f_pi_val': m_pi_f_pi_val,
        'm_pi/f_pi_err': m_pi_f_pi_err,
        'L': parameters['L'],
        'T': parameters['T'],
    })

    return parameters['ensemble'], series

def analyze(sets, T, L):
    # Unpack all the arguments from the list.
    (c2_val, c2_err), (c4_val, c4_err) = params

    # Generate a single time, they are all the same.
    time = np.array(range(len(c2_val)))

    # Perform the fits.
    fit2 = correlators.fit.cosh_fit_decorator(T)
    fit4 = correlators.fit.cosh_fit_offset_decorator(T)

    p2 = correlators.fit.fit(fit2, time, c2_val, c2_err,
                             omit_pre=13, p0=[0.222, c2_val[0]])
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


def unwrap_correlator_values(boot_correlators, index):
    return [
        bootstrap_set[index][0]
        for bootstrap_set in boot_correlators
    ]


def unwrap_correlator_errors(boot_correlators, index):
    return [
        bootstrap_set[index][1]
        for bootstrap_set in boot_correlators
    ]


def perform_fits(time, corr_val, corr_err, inv_corr_mat, fit_factory, p0, T, omit_pre):
    # Generate a fit function from the factory.
    fit_function = fit_factory(T)

    # Select the data for the fit.
    used_x, used_y, used_yerr = correlators.fit._cut(time, corr_val.T,
                                                     corr_err.T, omit_pre, 0)
    used_y = used_y.T
    used_yerr = used_yerr.T

    # Perform a regular fit with the given initial parameters.
    fit_param, pconv = op.curve_fit(fit_function, used_x, used_y, p0=p0,
                                    sigma=used_yerr)

    # Then perform a correlated fit using the previous result as the input.
    # This way it should be more stable.
    fit_param_corr, chi_sq = correlators.corrfit.curve_fit_correlated(
        fit_function, used_x, used_y, inv_corr_mat, p0=p0
    )

    dof = len(used_x) - 1 - len(fit_param_corr)
    p_value = 1 - scipy.stats.chi2.cdf(chi_sq, dof)

    return fit_param_corr[0], p_value
