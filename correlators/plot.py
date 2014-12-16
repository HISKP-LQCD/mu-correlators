#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

"""
Plots of data sets.
"""

from __future__ import division, absolute_import, print_function, \
    unicode_literals

import matplotlib.pyplot as pl
import numpy as np

import correlators.bootstrap
import correlators.fit
import correlators.transform


def plot_correlator(sets, name, offset=False):
    folded_val, folded_err = correlators.bootstrap.bootstrap_pre_transform(lambda x: x, sets)

    time_folded = np.array(range(len(folded_val)))

    fig_f = pl.figure()
    ax2 = fig_f.add_subplot(1, 1, 1)

    ax2.errorbar(time_folded, folded_val, yerr=folded_err, linestyle='none',
                 marker='+', label='folded')

    fit_param = {'color': 'gray'}
    used_param = {'color': 'blue'}
    data_param = {'color': 'black'}

    p0 = [0.222, 700, 30]
    if offset:
        fit_func = correlators.fit.cosh_fit_offset
        p0.append(0)
    else:
        fit_func = correlators.fit.cosh_fit

    p = correlators.fit.fit_and_plot(ax2, fit_func, time_folded, folded_val,
                                     folded_err, omit_pre=13, p0=p0,
                                     fit_param=fit_param, used_param=used_param,
                                     data_param=data_param)
    print('Fit parameters folded (mass, amplitude, shift, offset:', p)

    ax2.set_yscale('log')
    ax2.margins(0.05, tight=False)
    ax2.set_title('Folded Correlator')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$\frac{1}{2} [C(t) + C(T-t)]$')
    ax2.grid(True)

    fig_f.tight_layout()
    fig_f.savefig('{}_folded.pdf'.format(name))


def plot_effective_mass(sets, name):
    m_eff_val1, m_eff_err1 = correlators.bootstrap.bootstrap_pre_transform(
        correlators.transform.effective_mass_cosh, sets
    )
    time = np.arange(len(m_eff_val1)+2)
    time_cut = time[1:-1]

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.errorbar(time_cut, m_eff_val1, yerr=m_eff_err1, linestyle='none',
                marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax.set_title(r'Effective Mass $\operatorname{arcosh} ([C(t-1)+C(t+1)]/[2C(t)])$')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$m_\mathrm{eff}(t)$')
    ax.grid(True)
    ax.margins(0.05, 0.05)

    ax2.errorbar(time_cut[8:], m_eff_val1[8:], yerr=m_eff_err1[8:],
                 linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax2.errorbar([max(time_cut[8:])], [0.22293], [0.00035], marker='+')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$m_\mathrm{eff}(t)$')
    ax2.grid(True)
    ax2.legend(loc='best')
    ax2.margins(0.05, 0.05)

    fig.tight_layout()
    fig.savefig('{}_m_eff.pdf'.format(name))