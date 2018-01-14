#!/usr/bin/env python
# encoding: utf-8
"""
computations_marginalfisherinfo_marginalposterior_2d.py

Created by Loic Matthey on 2013-02-23.
Copyright (c) 2012 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scsp

from utils import *
# from statisticsmeasurer import *
# from randomfactorialnetwork import *
# from datagenerator import *
# from slicesampler import *
# from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from dataio import *
import progress


def main():
    ####
    #   2D two stimuli
    ####

    N = int(13 * 13)
    kappa = 3.0
    sigma = 0.5
    amplitude = 1.0
    min_distance = 0.001

    dataio = DataIO(label='compute_fimarg', calling_function='')
    additional_comment = ''

    N_sqrt = int(np.sqrt(N))
    coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt, endpoint=False)
    pref_angles = np.array(cross(2 * [coverage_1D.tolist()]))

    def population_code_response_2D(theta1,
                                    theta2,
                                    pref_angles,
                                    N=10,
                                    kappa=0.1,
                                    amplitude=1.0):
        return amplitude * np.exp(kappa * np.cos(theta1 - pref_angles[:, 0]) +
                                  kappa * np.cos(theta2 - pref_angles[:, 1])
                                  ) / (4. * np.pi**2. * scsp.i0(kappa)**2.)

    ## Estimate likelihood
    num_points = 100
    # num_points_space = np.arange(50, 1000, 200)
    # effects_num_points = []

    # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
    all_angles_1D = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

    def enforce_distance(theta1, theta2, min_distance=0.1):
        return np.abs(wrap_angles(theta1 - theta2)) > min_distance

    def show_population_output(data):
        N_sqrt = int(np.sqrt(data.size))
        plt.figure()
        plt.imshow(
            data.reshape((N_sqrt, N_sqrt)).T,
            origin='left',
            interpolation='nearest')
        plt.show()

    #### Compute Theo Inverse Fisher Info

    if True:
        ### Loop over min_distance and kappa
        # min_distance_space = np.linspace(0.0, 1.5, 10)
        # min_distance_space = np.array([min_distance])
        # min_distance_space = np.array([0.001])
        # min_distance_space = np.array([0.8])
        # kappa_space = np.linspace(0.05, 30., 40.)
        # kappa_space = np.array([kappa])

        # sigma_space = np.array([sigma])
        # sigma_space = np.array([0.1, 0.25, 0.5])
        # sigma_space = np.linspace(0.1, 1.0, 11)

        # dim1_size = min_distance_space.size
        # dim2_size = sigma_space.size

        item1_theta1_space = np.array([0.])
        item1_theta2_space = np.array([0.])

        item2_theta1_space = all_angles_1D
        item2_theta2_space = all_angles_1D

        deriv_mu = np.zeros((4, N))

        FI_2d_1obj_search = np.zeros((item1_theta1_space.size,
                                      item1_theta2_space.size, 2, 2))
        inv_FI_2d_1obj_search = np.zeros((item1_theta1_space.size,
                                          item1_theta2_space.size, 2, 2))
        FI_2d_search = np.zeros(
            (item1_theta1_space.size, item1_theta2_space.size,
             item2_theta1_space.size, item2_theta2_space.size, 4, 4))
        inv_FI_2d_search = np.zeros(
            (item1_theta1_space.size, item1_theta2_space.size,
             item2_theta1_space.size, item2_theta2_space.size, 4, 4))

        search_progress = progress.Progress(
            item1_theta1_space.size * item1_theta2_space.size *
            item2_theta1_space.size * item2_theta2_space.size)

        print "Doing from marginal FI"

        for i, item1_theta1 in enumerate(item1_theta1_space):
            for j, item1_theta2 in enumerate(item1_theta2_space):

                deriv_mu[0] = -kappa * np.sin(pref_angles[:, 0] - item1_theta1
                                              ) * population_code_response_2D(
                                                  item1_theta1,
                                                  item1_theta2,
                                                  pref_angles,
                                                  N=N,
                                                  kappa=kappa,
                                                  amplitude=amplitude)
                deriv_mu[1] = -kappa * np.sin(pref_angles[:, 1] - item1_theta2
                                              ) * population_code_response_2D(
                                                  item1_theta1,
                                                  item1_theta2,
                                                  pref_angles,
                                                  N=N,
                                                  kappa=kappa,
                                                  amplitude=amplitude)

                for k, item2_theta1 in enumerate(item2_theta1_space):
                    if search_progress.percentage() % 5.0 < 0.0001:
                        print "%.2f%%, %s left - %s" % (
                            search_progress.percentage(),
                            search_progress.time_remaining_str(),
                            search_progress.eta_str())

                    for l, item2_theta2 in enumerate(item2_theta2_space):

                        if (enforce_distance(
                                item1_theta1,
                                item2_theta1,
                                min_distance=min_distance)
                                and enforce_distance(
                                    item1_theta2,
                                    item2_theta2,
                                    min_distance=min_distance)):
                            # Only compute if items are sufficiently different

                            deriv_mu[2] = -kappa * np.sin(
                                pref_angles[:, 0] - item2_theta1
                            ) * population_code_response_2D(
                                item2_theta1,
                                item2_theta2,
                                pref_angles,
                                N=N,
                                kappa=kappa,
                                amplitude=amplitude)
                            deriv_mu[3] = -kappa * np.sin(
                                pref_angles[:, 1] - item2_theta2
                            ) * population_code_response_2D(
                                item2_theta1,
                                item2_theta2,
                                pref_angles,
                                N=N,
                                kappa=kappa,
                                amplitude=amplitude)

                            # FI, 2 items, 2 features per item
                            for ii in xrange(4):
                                for jj in xrange(4):
                                    FI_2d_search[i, j, k, l, ii, jj] = np.sum(
                                        deriv_mu[ii] * deriv_mu[jj]) / (
                                            sigma**2.)

                            # FI_2d_search[i, j, k, l, 0, 0] = np.sum(der_0*der_0)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 0, 1] = np.sum(der_0*der_1)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 0, 2] = np.sum(der_0*der_2)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 0, 3] = np.sum(der_0*der_3)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 1, 1] = np.sum(der_1*der_1)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 1, 2] = np.sum(der_1*der_2)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 1, 3] = np.sum(der_1*der_3)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 2, 2] = np.sum(der_2*der_2)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 2, 3] = np.sum(der_2*der_3)/(2.*sigma**2.)
                            # FI_2d_search[i, j, k, l, 3, 3] = np.sum(der_3*der_3)/(2.*sigma**2.)

                            # Complete matrix
                            # triu_2_tril(FI_2d_search[i, j, k, l])

                            inv_FI_2d_search[i, j, k, l] = np.linalg.inv(
                                FI_2d_search[i, j, k, l])

                        search_progress.increment()

                # FI for 1 object
                FI_2d_1obj_search[i, j, 0, 0] = np.sum(deriv_mu[0]**
                                                       2.) / sigma**2.
                FI_2d_1obj_search[i, j, 0, 1] = np.sum(
                    deriv_mu[0] * deriv_mu[1]) / sigma**2.
                FI_2d_1obj_search[i, j, 1, 1] = np.sum(deriv_mu[1]**
                                                       2.) / sigma**2.
                FI_2d_1obj_search[i, j, 1, 0] = FI_2d_1obj_search[i, j, 0, 1]

                inv_FI_2d_1obj_search[i, j] = np.linalg.inv(
                    FI_2d_1obj_search[i, j])

        # Compute marginal Fisher Info
        FI_2d_2obj_marginal = np.mean(
            np.mean(np.ma.masked_equal(np.squeeze(FI_2d_search), 0.0), axis=0),
            axis=0)
        inv_FI_2d_2obj_marginal_bis = np.mean(
            np.mean(
                np.ma.masked_equal(np.squeeze(inv_FI_2d_search), 0.0), axis=0),
            axis=0)
        inv_FI_2d_2obj_marginal = np.linalg.inv(FI_2d_2obj_marginal)

        # print 1./inv_FI_search
        # print FI_search
        # print 1./inv_FI_1_search

        # Some plots
        pcolor_2d_data(inv_FI_2d_search[0, 0, :, :, 0, 0])

        plt.figure()
        plt.bar(
            [0, 1, 2], [
                inv_FI_2d_1obj_search[0, 0, 0, 0],
                inv_FI_2d_2obj_marginal[0, 0],
                inv_FI_2d_2obj_marginal_bis[0, 0]
            ],
            width=0.5)

        # plt.figure()
        # plt.semilogy(min_distance_space, inv_FI_search- inv_FI_1_search)

        # plt.figure()
        # plt.semilogy(min_distance_space, inv_FI_search)
        # plt.semilogy(min_distance_space, inv_FI_1_search)

        # plt.figure()
        # plt.plot(min_distance_space, inv_FI_search)

        plt.rcParams['font.size'] = 18

        # plt.figure()
        # plt.semilogy(min_distance_space, (inv_FI_search- inv_FI_1_search)[:, 1:])
        # plt.xlabel('Minimum distance')
        # plt.ylabel('$\hat{I_F}^{-1} - {I_F^{(1)}}^{-1}$')

    plt.show()

    # say_finished(additional_comment=additional_comment)
    return locals()


if __name__ == '__main__':
    main()