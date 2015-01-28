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
import progressbar
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
    (mean_2_val, mean_2_err), (mean_4_val, mean_4_err) = \
            correlators.bootstrap.average_combined_array(orig_correlators)

    sample_count = 3 * len(orig_correlators)
    #sample_count = 100

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

    mean_result = analyze_2_4(
        time, mean_2_val, mean_2_err, inv_corr_mat_2,
        correlators.fit.cosh_fit_decorator, [0.222, mean_2_val[0]],
        mean_4_val, mean_4_err, inv_corr_mat_4,
        correlators.fit.cosh_fit_offset_decorator, [0.222, mean_4_val[0], 0],
        T, L, omit_pre,
    )

    boot_results = pd.DataFrame()

    p_bar = progressbar.ProgressBar()
    for sample_id in p_bar(range(sample_count)):
        boot_result = analyze_2_4(
            time, correlators_2_val[sample_id], correlators_2_err[sample_id],
            inv_corr_mat_2, correlators.fit.cosh_fit_decorator,
            [0.222, correlators_2_val[sample_id][0]],
            correlators_4_val[sample_id], correlators_4_err[sample_id],
            inv_corr_mat_4, correlators.fit.cosh_fit_offset_decorator,
            [0.222, correlators_2_val[sample_id][0], 0], T, L, omit_pre
        )

        boot_series = pd.Series(boot_result)
        boot_results[sample_id] = boot_series

    print(boot_results)


    boot_result = {
        key: np.std(np.array(dist))
        for key, dist in boot_results.T.iteritems()
    }

    series = pd.Series({
        'a_0_val': mean_result['a_0'],
        'a_0_err': boot_result['a_0'],
        'm_2_val': mean_result['m_2'],
        'm_2_err': boot_result['m_2'],
        'm_4_val': mean_result['m_4'],
        'm_4_err': boot_result['m_4'],
        'a_0*m_2_val': mean_result['a_0*m_2'],
        'a_0*m_2_err': boot_result['a_0*m_2'],
        'p_value_2_val': mean_result['p_value_2'],
        'p_value_2_err': boot_result['p_value_2'],
        'p_value_4_val': mean_result['p_value_4'],
        'p_value_4_err': boot_result['p_value_4'],
        'm_pi/f_pi_val': m_pi_f_pi_val,
        'm_pi/f_pi_err': m_pi_f_pi_err,
        'L': parameters['L'],
        'T': parameters['T'],
    }, name=parameters['ensemble'])

    return series


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


def analyze_2_4(time, correlator_2_val, correlator_2_err, inv_corr_mat_2,
                fit_factory_2, p0_2, correlator_4_val, correlator_4_err,
                inv_corr_mat_4, fit_factory_4, p0_4, T, L, omit_pre):
    m_2, p_value_2 = perform_fits(
        time, correlator_2_val, correlator_2_err, inv_corr_mat_2,
        fit_factory_2, p0_2, T, omit_pre
    )

    m_4, p_value_4 = perform_fits(
        time, correlator_4_val, correlator_4_err, inv_corr_mat_4,
        fit_factory_4, p0_4, T, omit_pre
    )

    delta_m = m_4 - 2 * m_2

    a_0 = correlators.scatlen.compute_a0(m_2, m_4, L)

    return {
        'a_0': a_0,
        'm_2': m_2,
        'm_4': m_4,
        'a_0*m_2': a_0 * m_2,
        'm_2^2': m_2**2,
        'p_value_2': p_value_2,
        'p_value_4': p_value_4,
    }
