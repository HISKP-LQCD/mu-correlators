#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, unicode_literals

# Standard library.
import argparse

# External libraries.
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize as op
import scipy.stats

# This package.
import loader

def fit_and_plot(func, x, y, yerr, axes, omit_pre=0, omit_post=0, p0=None,
                 fit_param={}, data_param={}, used_param={}, axes_res=None):

    if omit_post == 0:
        used_x = x[omit_pre:]
        used_y = y[omit_pre:]
        used_yerr = yerr[omit_pre:]
    else:
        end = - omit_post - 1
        used_x = x[omit_pre:end]
        used_y = y[omit_pre:end]
        used_yerr = yerr[omit_pre:end]

    popt, pconv = op.curve_fit(func, used_x, used_y, p0=p0, sigma=used_yerr)
    try:
        error = np.sqrt(pconv.diagonal())
    except AttributeError:
        error = np.nan * np.ones(popt.shape)

    fx = np.linspace(np.min(x), np.max(x), 1000)
    fy = func(fx, *popt)

    param = {}
    param = dict(param.items() + fit_param.items())
    axes.plot(fx, fy, **param)

    param = {'marker': '+', 'linestyle': 'none'}
    param = dict(param.items() + data_param.items())
    axes.errorbar(x, y, yerr=yerr, **param)

    param = {'marker': '+', 'linestyle': 'none'}
    param = dict(param.items() + used_param.items())
    axes.errorbar(used_x, used_y, yerr=used_yerr, **param)

    axes_res = axes.twinx()
    axes_res.errorbar(used_x, used_y - func(used_x, *popt), yerr=used_yerr,
                      marker='+', linestyle='none', color='red', alpha=0.3)
    axes_res.set_ylabel('Residual')

    dof = len(used_y) - len(popt) - 1
    chisq, p = scipy.stats.chisquare(used_y, func(used_y, *popt), ddof=len(popt))

    print('χ2:', chisq)
    print('χ2/DOF:', chisq/dof)
    print('p:', p)

    return list(zip(popt, error))

def fold_data(val, err):
    r'''
    Folds the data around the middle element and averages.

    The expectation is to yield a :math:`\cosh` function. The transformation of
    the :math:`\{x_i\colon i = 1, \ldots, N\}` will generate new data points
    like this:

    .. math::

        y_i := \frac{x_i + x_{N-i}}2

    :param np.array val: Array with an even number of elements, values
    :param np.array err: Array with an even number of elements, errors
    :returns: Folded array with :math:`N/2` elements
    :rtype: np.array
    '''
    n = len(val)
    second_rev_val = val[n//2+1:][::-1]
    first_val = val[:n//2+1]
    first_val[1:-1] += second_rev_val
    first_val[1:-1] /= 2.

    second_rev_err = err[n//2+1:][::-1]
    first_err = err[:n//2+1]
    first_err[1:-1] = np.sqrt(first_err[1:-1]**2 + second_rev_err**2) / 2

    return first_val, first_err

def effective_mass(data, delta_t=1):
    r'''
    Computes the effective mass of the data.

    The effective mass is defined as:

    .. math::

        m_\text{eff} := - \frac{1}{\Delta t} \ln\left(\frac{C(t + \Delta t)}{C(t)}\right)

    :param np.array data: Time series of correlation functions, :math:`C(t)`
    :param int delta_t: Number of elements to use as :math:`\Delta t`
    :returns: Effective mass
    :rtype: float
    '''
    m = - np.log(data[delta_t:] / data[:-delta_t]) / delta_t
    return m

def effective_mass_cosh(val, err, dt=1):
    r'''
    .. math::

        \operatorname{arcosh} \left(\frac{C(t-1)+C(t+1)}{2C(t)}\right)
    '''
    frac_val = (val[:-2*dt] + val[2*dt:]) / val[dt:-dt] / 2
    frac_err = np.sqrt(
        (val[2*dt:] / val[dt:-dt] / 2 * err[:-2*dt])**2
        + (val[:-2*dt] / val[dt:-dt] / 2 * err[2*dt:])**2
        + ((val[:-2*dt] + val[2*dt:]) / val[dt:-dt]**2 / 2 * err[dt:-dt])**2
    )

    m_eff_val = np.arccosh(frac_val)
    m_eff_err = 1 / np.sqrt(frac_val**2 - 1) * frac_err

    return m_eff_val, m_eff_err

def cosh_fit(x, m, a, shift, offset):
    '''

    .. math::

        \operatorname{fit}(x; m_1, m_2, a_1, a_2, \mathrm{offset})
        = a_1 \exp(- m_1 x) + a_2 \exp(- m_2 [n - x]) + \mathrm{offset}

    :param np.array x: Input values
    :param float m1: Effective mass for falling exponential
    :param float m2: Effective mass for rising exponential
    :param float a1: Amplitude for falling exponential
    :param float a2: Amplitude for rising exponential
    :param float offset: Constant offset
    '''
    y = shift - x
    first = a * np.exp(-x*m)
    second = a * np.exp(-y*m)
    return first + second + offset

def exp_fit(x, m1, a1, offset):
    '''
    :param np.array x: Input values
    :param float m1: Effective mass for falling exponential
    :param float a1: Amplitude for falling exponential
    :param float offset: Constant offset
    '''
    return a1 * np.exp(-x*m1) + offset

def plot_correlator(val, err):
    real_val = np.real(val)
    real_err = np.real(err)
    folded_val, folded_err = fold_data(val, err)


    time = np.array(range(len(real_val)))
    time_folded = np.array(range(len(folded_val)))

    fig = pl.figure()
    fig_f = pl.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax2 = fig_f.add_subplot(1, 1, 1)

    ax.errorbar(time, real_val, yerr=real_err, linestyle='none', marker='+', label='complete')
    ax2.errorbar(time_folded, folded_val, yerr=folded_err, linestyle='none', marker='+', label='folded')

    fit_param = {'color': 'gray'}
    used_param = {'color': 'blue'}
    data_param = {'color': 'black'}

    p = fit_and_plot(cosh_fit, time, real_val, real_err, ax, omit_pre=13, omit_post=12,
                     p0=[0.2, 400, 30, 0], fit_param=fit_param,
                     used_param=used_param, data_param=data_param)
    print('Fit parameters cosh:', p[0])

    p = fit_and_plot(cosh_fit, time_folded, folded_val, folded_err, ax2, omit_pre=13,
                     p0=[0.222, 700, 30, 0], fit_param=fit_param,
                     used_param=used_param, data_param=data_param)
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

def plot_effective_mass(val, err):
    time = np.arange(len(val))
    m_eff_val, m_eff_err = effective_mass_cosh(val, err)
    time_cut = time[1:-1]

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.errorbar(time_cut, m_eff_val, yerr=m_eff_err, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ complete')
    ax.set_title(r'Effective Mass $\operatorname{arcosh} ([C(t-1)+C(t+1)]/[2C(t)])$')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$m_\mathrm{eff}(t)$')
    #ax.legend(loc='best')
    ax.grid(True)
    ax.margins(0.05, 0.05)

    ax2.errorbar(time_cut[8:-8], m_eff_val[8:-8], yerr=m_eff_err[8:-8], linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ complete')
    ax2.errorbar([22.5], [0.22293], [0.00035], marker='+')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$m_\mathrm{eff}(t)$')
    ax2.grid(True)
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

    val, err = loader.average_loader(options.filename)

    print('Correlators:')
    plot_correlator(val, err)
    print()
    print('Effective Mass:')
    plot_effective_mass(val, err)

if __name__ == '__main__':
    main()
