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


def cosh_fit_decorator(n):
    r'''
    Generates a cosh fit function for given data length

    The generated function will be the following with :math:`n` baked in:

    .. math::

        \operatorname{fit}(x; m_1, m_2, a_1, a_2, \mathrm{offset})
        = a_1 \exp(- m_1 x) + a_2 \exp(- m_2 [n - x]) + \mathrm{offset}

    :rtype: function
    '''
    def cosh_fit(x, m1, m2, a1, a2, offset):
        '''
        :param np.array x: Input values
        :param float m1: Effective mass for falling exponential
        :param float m2: Effective mass for rising exponential
        :param float a1: Amplitude for falling exponential
        :param float a2: Amplitude for rising exponential
        :param float offset: Constant offset
        '''
        y = n - x
        first = a1 * np.exp(-x*m1)
        second = a2 * np.exp(-y*m2)
        return first + second + offset

    return cosh_fit

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
    print(data)

    plot_correlator(data)
    plot_effective_mass(data)

def plot_correlator(data):
    real = np.real(data)
    folded = fold_data(real)

    time = np.array(range(len(data)))
    time_folded = np.array(range(len(folded)))

    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)

    fit_func = cosh_fit_decorator(len(data))
    popt, pconv = op.curve_fit(fit_func, time, real, p0=[0.1, 0.1, 400, 300/np.exp(5), 10])
    print(popt)
    print(pconv)
    print(np.sqrt(pconv.diagonal()))
    x = np.linspace(np.min(time), np.max(time), 1000)
    y = fit_func(x, *popt)
    ax.plot(x, y, label='cosh fit')

    fit_func = exp_fit
    popt, pconv = op.curve_fit(fit_func, time_folded, folded, p0=[0.2, 450, 0])
    print(popt)
    print(pconv)
    print(np.sqrt(pconv.diagonal()))
    x = np.linspace(np.min(time_folded), np.max(time_folded), 1000)
    y = fit_func(x, *popt)
    ax.plot(x, y, label='exp fit')

    ax.plot(real, linestyle='none', marker='+', label='complete')
    ax.plot(folded, linestyle='none', marker='+', label='folded')

    ax.set_yscale('log')
    ax.set_title('Correlator')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$C(t)$')
    ax.legend(loc='best')
    ax.grid(True)
    fig.savefig('folded.pdf')

def plot_effective_mass(data):
    real = np.real(data)
    folded = fold_data(real)
    m_eff = effective_mass_cosh(folded)

    time = np.array(range(len(data)))
    time_folded = np.array(range(len(folded)))


    pl.clf()
    pl.plot(m_eff, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ folded')
    #pl.plot(effective_mass_cosh(real), linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ complete')
    pl.title('Effective Mass')
    pl.xlabel(r'$i$')
    pl.ylabel(r'$m_\mathrm{eff}(t_i)$')
    pl.legend(loc='best')
    pl.grid(True)
    pl.savefig('m_eff.pdf')
    #pl.show()
    pl.clf()


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
