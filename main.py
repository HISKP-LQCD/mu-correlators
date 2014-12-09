#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>
# Licensed under The GNU Public License Version 2

r'''
The value for the A100 data sets should be 0.22293(35)(38).

With the data from
``A100.24_L24_T48_beta190_mul0100_musig150_mudel190_kappa1632550/ev120/TB2_SO_LI6_new/C2_pi+-_conf????.dat``, I got :math:`0.22229 \pm 0.00003`
'''

# I am used to Python 3, this enables some future features here in Python 2.
from __future__ import division, absolute_import, print_function, unicode_literals

import argparse

import matplotlib.pyplot as pl
import numpy as np

import bootstrap
import fit
import loader
import unitprint


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


def fit_correlator(sets):
    popt, perr = bootstrap.bootstrap_pre_transform(
        correlator_single_fit, sets,
        reduction=bootstrap.average_and_std_arrays,
    )
    print(popt)
    print(perr)
    print(unitprint.siunitx(popt, perr))


def correlator_single_fit(params):
    val, err = params
    time = np.array(range(len(val)))
    p = fit.fit(fit.cosh_fit, time, val, err, omit_pre=13,
                p0=[0.222, 700, 30, 0])
    return p


def mass_difference(params):
    # Unpack all the arguments from the list.
    (c2_val, c2_err), (c4_val, c4_err) = params

    # Generate a single time, they are all the same.
    time = np.array(range(len(c2_val)))

    # Perform the fits.
    p2 = fit.fit(fit.cosh_fit, time, c2_val, c2_err, omit_pre=13,
                p0=[0.222, 700, 30, 0])
    p4 = fit.fit(fit.cosh_fit, time, c4_val, c4_err, omit_pre=13,
                p0=[0.222, 700, 30, 0])

    m2 = p2[0]
    m4 = p4[0]

    delta_m = m4 - m2

    return m2, m4, delta_m

def plot_correlator(sets, name):
    folded_val, folded_err = bootstrap.bootstrap_pre_transform(lambda x: x, sets)

    time_folded = np.array(range(len(folded_val)))

    fig_f = pl.figure()
    ax2 = fig_f.add_subplot(1, 1, 1)

    ax2.errorbar(time_folded, folded_val, yerr=folded_err, linestyle='none',
                 marker='+', label='folded')

    fit_param = {'color': 'gray'}
    used_param = {'color': 'blue'}
    data_param = {'color': 'black'}

    p = fit.fit_and_plot(ax2, fit.cosh_fit, time_folded, folded_val,
                         folded_err, omit_pre=13, p0=[0.222, 700, 30, 0],
                         fit_param=fit_param, used_param=used_param,
                         data_param=data_param)
    print('Fit parameters folded (mass, amplitude, shift, offset:', p)

    ax2.set_yscale('log')
    ax2.margins(0.05, tight=False)
    ax2.set_title('Folded Correlator')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$\frac{1}{2} [C(t) + C(T-t)]$')
    #ax2.legend(loc='best')
    ax2.grid(True)
    #ax2.yaxis.tick_right()
    #ax2.yaxis.set_label_position('right')

    fig_f.tight_layout()
    fig_f.savefig('{}_folded.pdf'.format(name))


def plot_effective_mass(sets, name):
    m_eff_val1, m_eff_err1 = bootstrap.bootstrap_pre_transform(effective_mass_cosh, sets)
    time = np.arange(len(m_eff_val1)+2)
    time_cut = time[1:-1]

    fig = pl.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax.errorbar(time_cut, m_eff_val1, yerr=m_eff_err1, linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax.set_title(r'Effective Mass $\operatorname{arcosh} ([C(t-1)+C(t+1)]/[2C(t)])$')
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$m_\mathrm{eff}(t)$')
    #ax.legend(loc='best')
    ax.grid(True)
    ax.margins(0.05, 0.05)

    ax2.errorbar(time_cut[8:], m_eff_val1[8:], yerr=m_eff_err1[8:], linestyle='none', marker='+', label=r'$m_{\mathrm{eff}}$ pre')
    ax2.errorbar([max(time_cut[8:])], [0.22293], [0.00035], marker='+')
    ax2.set_xlabel(r'$t/a$')
    ax2.set_ylabel(r'$m_\mathrm{eff}(t)$')
    ax2.grid(True)
    ax2.legend(loc='best')
    ax2.margins(0.05, 0.05)

    fig.tight_layout()
    fig.savefig('{}_m_eff.pdf'.format(name))


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

    if len(options.filename) == 1:
        two_points, four_points = loader.folder_loader(options.filename[0])

        # Combine the two lists of data into one list of lists. That way the
        # configurations are grouped together.
        combined = zip(two_points, four_points)

        val, err = bootstrap.bootstrap_pre_transform(
            mass_difference,
            combined,
            bootstrap.average_combined_array,
        )

        for name, string in zip(
            ['m₂', 'm₄', 'ΔE'],
            unitprint.siunitx(val, err),
        ):
            print(name, string)

        plot_correlator(two_points, 'c2')
        plot_correlator(four_points, 'c4')
        plot_effective_mass(two_points, 'c2')
        plot_effective_mass(four_points, 'c4')
    else:
        sets = loader.folded_list_loader(options.filename)
        print(len(sets), 'set loaded.')

        print('Plot correlators:')
        plot_correlator(sets, 'c2-single')

        print('Fit correlators:')
        fit_correlator(sets)

        print('Effective mass:')
        plot_effective_mass(sets, 'c2-single')


if __name__ == '__main__':
    main()
