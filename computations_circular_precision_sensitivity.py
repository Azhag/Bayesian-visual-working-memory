#!/usr/bin/env python
# encoding: utf-8
"""
computations_circular_precision_sensitivity.py

Created by Loic Matthey on 2013-10-05
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as spst

import dataio as DataIO
import utils

import em_circularmixture

# from statisticsmeasurer import *
# from randomfactorialnetwork import *
# from datagenerator import *
# from slicesampler import *

import progress

def check_precision_sensitivity_determ():
    ''' Let's construct a situation where we have one Von Mises component and one random component. See how the random component affects the basic precision estimator we use elsewhere.
    '''

    N = 1000
    kappa_space = np.array([3., 10., 20.])
    # kappa_space = np.array([3.])
    nb_repeats = 20
    ratio_to_kappa = False
    savefigs = True
    precision_nb_samples = 101

    N_rnd_space             = np.linspace(0, N/2, precision_nb_samples).astype(int)
    precision_all           = np.zeros((N_rnd_space.size, nb_repeats))
    kappa_estimated_all     = np.zeros((N_rnd_space.size, nb_repeats))
    precision_squared_all   = np.zeros((N_rnd_space.size, nb_repeats))
    kappa_mixtmodel_all     = np.zeros((N_rnd_space.size, nb_repeats))
    mixtmodel_all           = np.zeros((N_rnd_space.size, nb_repeats, 2))

    dataio = DataIO.DataIO()

    target_samples = np.zeros(N)

    for kappa in kappa_space:

        true_kappa = kappa*np.ones(N_rnd_space.size)

        # First sample all as von mises
        samples_all = spst.vonmises.rvs(kappa, size=(N_rnd_space.size, nb_repeats, N))

        for repeat in progress.ProgressDisplay(xrange(nb_repeats)):
            for i, N_rnd in enumerate(N_rnd_space):
                samples = samples_all[i, repeat]

                # Then set K of them to random [-np.pi, np.pi] values.
                samples[np.random.randint(N, size=N_rnd)] = utils.sample_angle(N_rnd)

                # Estimate precision from those samples.
                precision_all[i, repeat] = utils.compute_precision_samples(samples, square_precision=False, remove_chance_level=False)
                precision_squared_all[i, repeat] = utils.compute_precision_samples(samples, square_precision=True)

                # convert circular std dev back to kappa
                kappa_estimated_all[i, repeat] = utils.stddev_to_kappa(1./precision_all[i, repeat])

                # Fit mixture model
                params_fit = em_circularmixture.fit(samples, target_samples)
                kappa_mixtmodel_all[i, repeat] = params_fit['kappa']
                mixtmodel_all[i, repeat] = params_fit['mixt_target'], params_fit['mixt_random']

                print "%d/%d N_rnd: %d, Kappa: %.3f, precision: %.3f, kappa_tilde: %.3f, precision^2: %.3f, kappa_mixtmod: %.3f" % (repeat, nb_repeats, N_rnd, kappa, precision_all[i, repeat], kappa_estimated_all[i, repeat], precision_squared_all[i, repeat], kappa_mixtmodel_all[i, repeat])


        if ratio_to_kappa:
            precision_all /= kappa
            precision_squared_all /= kappa
            kappa_estimated_all /= kappa
            true_kappa /= kappa

        f, ax = plt.subplots()
        ax.plot(N_rnd_space/float(N), true_kappa, 'k-', linewidth=3, label='Kappa_true')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(precision_all, axis=-1), np.std(precision_all, axis=-1), ax_handle=ax, label='precision')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(precision_squared_all, axis=-1), np.std(precision_squared_all, axis=-1), ax_handle=ax, label='precision^2')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(kappa_estimated_all, axis=-1), np.std(kappa_estimated_all, axis=-1), ax_handle=ax, label='kappa_tilde')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(kappa_mixtmodel_all, axis=-1), np.std(kappa_mixtmodel_all, axis=-1), ax_handle=ax, label='kappa mixt model')

        ax.legend()
        ax.set_title('Effect of random samples on precision. kappa: %.2f. ratiokappa %s' % (kappa, ratio_to_kappa))
        ax.set_xlabel('Proportion random samples. N tot %d' % N)
        ax.set_ylabel('Kappa/precision (not same units)')
        f.canvas.draw()

        if savefigs:
            dataio.save_current_figure("precision_sensitivity_kappa%dN%d_{unique_id}.pdf" % (kappa, N))

        # Do another plot, with kappa and mixt_target/mixt_random. Use left/right axis separately
        f, ax = plt.subplots()
        ax2 = ax.twinx()

        # left axis, kappa
        ax.plot(N_rnd_space/float(N), true_kappa, 'k-', linewidth=3, label='kappa true')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(kappa_mixtmodel_all, axis=-1), np.std(kappa_mixtmodel_all, axis=-1), ax_handle=ax, label='kappa')

        # Right axis, mixture probabilities
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(mixtmodel_all[..., 0], axis=-1), np.std(mixtmodel_all[..., 0], axis=-1), ax_handle=ax2, label='mixt target', color='r')
        utils.plot_mean_std_area(N_rnd_space/float(N), np.mean(mixtmodel_all[..., 1], axis=-1), np.std(mixtmodel_all[..., 1], axis=-1), ax_handle=ax2, label='mixt random', color='g')
        ax.set_title('Mixture model parameters evolution. kappa: %.2f, ratiokappa %s' % (kappa, ratio_to_kappa))
        ax.set_xlabel('Proportion random samples. N tot %d' % N)
        ax.set_ylabel('Kappa')
        ax2.set_ylabel('Mixture proportions')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)

        if savefigs:
            dataio.save_current_figure("precision_sensitivity_mixtmodel_kappa%dN%d_{unique_id}.pdf" % (kappa, N))



    return locals()

if __name__ == '__main__':

    all_vars = check_precision_sensitivity_determ()

    for key, val in all_vars.items():
        locals()[key] = val

    plt.show()



