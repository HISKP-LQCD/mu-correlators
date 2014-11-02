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

# This package.
import loader

def fit_and_plot(func, x, y, axes, omit_pre=0, omit_post=0, p0=None,
                 fit_param={}, data_param={}, used_param={}):

    used_x = x[omit_pre:-omit_post-1]
    used_y = y[omit_pre:-omit_post-1]
    popt, pconv = op.curve_fit(func, used_x, used_y, p0=p0)
    error = np.sqrt(pconv.diagonal())

    fx = np.linspace(np.min(x), np.max(x), 1000)
    fy = func(fx, *popt)

    param = {}
    param = dict(param.items() + fit_param.items())
    axes.plot(fx, fy, **param)

    param = {'marker': '+', 'linestyle': 'none'}
    param = dict(param.items() + data_param.items())
    axes.plot(x, y, **param)

    param = {'marker': '+', 'linestyle': 'none'}
    param = dict(param.items() + used_param.items())
    axes.plot(used_x, used_y, **param)

    return list(zip(popt, error))

def fold_data(data):
    r'''
    Folds the data around the middle element and averages.

    The expectation is to yield a :math:`\cosh` function. The transformation of
    the :math:`\{x_i\colon i = 1, \ldots, N\}` will generate new data points
    like this:

    .. math::

        y_i := \frac{x_i + x_{N-i+1}}2

    :param np.array data: Array with an even number of elements
    :returns: Folded array with :math:`N/2` elements
    :rtype: np.array
    '''
    n = len(data)
    first = data[:n//2]
    second = data[n//2:]
    return (first + second[::-1]) / 2

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

def effective_mass_cosh(data, delta_t=1):
    r'''
    .. math::

        \operatorname{arcosh} \left(\frac{C(t-1)+C(t+1)}{2C(t)}\right)
    '''
    frac = (data[:-2*delta_t] + data[2*delta_t:]) / data[delta_t:-delta_t] / 2

    return np.arccosh(frac)


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

def main():
    options = _parse_args()

    data = loader.average_loader(options.filename)

    plot_correlator(data)
    plot_effective_mass(data)

def plot_correlator(data):
    real = np.real(data)
    folded = fold_data(real)

    time = np.array(range(len(data)))
    time_folded = np.array(range(len(folded)))

    fig = pl.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax.plot(real, linestyle='none', marker='+', label='complete')
    ax2.plot(folded, linestyle='none', marker='+', label='folded')

    fit_param = {'color': 'gray'}
    used_param = {'color': 'blue'}
    data_param = {'color': 'black'}

    p = fit_and_plot(cosh_fit, time, real, ax, omit_pre=9, omit_post=9,
                     p0=[0.2, 400, 30, 0], fit_param=fit_param,
                     used_param=used_param, data_param=data_param)
    print('Fit parameters cosh:', p[0])

    p = fit_and_plot(exp_fit, time_folded, folded, ax2, omit_pre=5,
                     omit_post=3, fit_param=fit_param, used_param=used_param,
                     data_param=data_param)
    print('Fit parameters exp:', p[0])

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
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    fig.tight_layout()
    fig.savefig('folded.pdf')

def plot_effective_mass(data):
    real = np.real(data)
    time = np.array(range(len(real)))
    m_eff = effective_mass_cosh(real)
    time_cut = time[1:-1]

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.plot(time_cut, m_eff, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ complete')
    ax.set_title(r'Effective Mass $\operatorname{arcosh} ([C(t-1)+C(t+1)]/[2C(t)])$')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$m_\mathrm{eff}(t)$')
    #ax.legend(loc='best')
    ax.grid(True)
    ax.margins(0.05, 0.05)

    ax2.plot(time_cut[8:-8], m_eff[8:-8], linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ complete')
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

if __name__ == '__main__':
    main()
