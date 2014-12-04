#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, unicode_literals

import argparse

import matplotlib.pyplot as pl
import numpy as np

import bootstrap
import fit
import loader


def effective_mass(data, delta_t=1):
    r'''
    Computes the effective mass of the data.

    The effective mass is defined as:

    .. math::

        m_\text{eff} := - \frac{1}{\Delta t}
        \ln\left(\frac{C(t + \Delta t)}{C(t)}\right)

    :param np.array data: Time series of correlation functions, :math:`C(t)`
    :param int delta_t: Number of elements to use as :math:`\Delta t`
    :returns: Effective mass
    :rtype: float
    '''
    m = - np.log(data[delta_t:] / data[:-delta_t]) / delta_t
    return m


def effective_mass_cosh(val, dt=1):
    r'''
    .. math::

        \operatorname{arcosh} \left(\frac{C(t-1)+C(t+1)}{2C(t)}\right)
    '''
    frac = (val[:-2*dt] + val[2*dt:]) / val[dt:-dt] / 2
    m_eff = np.arccosh(frac)
    return m_eff


def plot_correlator(sets):
    real_val = np.real(val)
    real_err = np.real(err)
    folded_val, folded_err = fold_data(val, err)

    time = np.array(range(len(real_val)))
    time_folded = np.array(range(len(folded_val)))

    fig = pl.figure()
    fig_f = pl.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = fig_f.add_subplot(1, 1, 1)

    ax.errorbar(time, real_val, yerr=real_err, linestyle='none', marker='+',
                label='complete')
    ax2.errorbar(time_folded, folded_val, yerr=folded_err, linestyle='none',
                 marker='+', label='folded')

    fit_param = {'color': 'gray'}
    used_param = {'color': 'blue'}
    data_param = {'color': 'black'}

    p = fit.fit_and_plot(fit.cosh_fit, time, real_val, real_err, ax,
                         omit_pre=13, omit_post=12, p0=[0.2, 400, 30, 0],
                         fit_param=fit_param, used_param=used_param,
                         data_param=data_param)
    print('Fit parameters cosh:', p[0])

    p = fit.fit_and_plot(fit.cosh_fit, time_folded, folded_val, folded_err,
                         ax2, omit_pre=13, p0=[0.222, 700, 30, 0],
                         fit_param=fit_param, used_param=used_param,
                         data_param=data_param)
    print('Fit parameters folded:', p[0])

    ax.set_yscale('log')
    ax.margins(0.05, tight=False)
    ax.set_title('Correlator')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$C(t)$')
    #ax.legend(loc='best')
    ax.grid(True)

    ax2.set_yscale('log')
    ax2.margins(0.05, tight=False)
    ax2.set_title('Folded Correlator')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$\frac{1}{2} [C(t) + C(T-t)]$')
    #ax2.legend(loc='best')
    ax2.grid(True)
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position('right')

    fig.tight_layout()
    fig.savefig('correlator.pdf')

    fig_f.tight_layout()
    fig_f.savefig('folded.pdf')


def plot_effective_mass(sets):
    m_eff_val1, m_eff_err1 = bootstrap.bootstrap_pre_transform(effective_mass_cosh, sets)
    time = np.arange(len(m_eff_val1)+2)
    time_cut = time[1:-1]

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.errorbar(time_cut, m_eff_val1, yerr=m_eff_err1, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax.errorbar(time_cut+0.1, m_eff_val2, yerr=m_eff_err2, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ post')
    ax.set_title(r'Effective Mass $\operatorname{arcosh} ([C(t-1)+C(t+1)]/[2C(t)])$')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$m_\mathrm{eff}(t)$')
    #ax.legend(loc='best')
    ax.grid(True)
    ax.margins(0.05, 0.05)

    ax2.errorbar(time_cut[8:], m_eff_val1[8:], yerr=m_eff_err1[8:], linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax2.errorbar(time_cut[8:]+0.1, m_eff_val2[8:], yerr=m_eff_err2[8:], linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ post')
    ax2.errorbar([max(time_cut[8:])], [0.22293], [0.00035], marker='+')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$m_\mathrm{eff}(t)$')
    ax2.grid(True)
    ax2.legend(loc='best')
    ax2.margins(0.05, 0.05)

    fig.tight_layout()
    fig.savefig('m_eff.pdf')


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


def main():
    options = _parse_args()

    sets = loader.folded_list_loader(options.filename)

    print('Correlators:')
    plot_correlator(sets)

    print('Effective Mass:')
    plot_effective_mass(sets)


if __name__ == '__main__':
    main()
