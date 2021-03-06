#!/usr/bin/env python
# encoding: utf-8
"""
fisherinformation.py

Created by Loic Matthey on 2012-06-03.
Copyright (c) 2012 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scsp

from utils import *
from statisticsmeasurer import *
from randomfactorialnetwork import *
from datagenerator import *
import slicesampler
# from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
import progress

def profile_me():
    print "-------- Profiling ----------"

    import cProfile
    import pstats

    cProfile.runctx('run_fisher_info_2d()', globals(), locals(), filename='profile_fi.stats')

    stat = pstats.Stats('profile_fi.stats')
    stat.strip_dirs().sort_stats('cumulative').print_stats()

    return {}


if __name__ == '__main__':
    '''
        Compute and plot the Fisher Information for a RandomFactorialNetwork
    '''

    ###

    if False:
        ## Small tries firsts

        N_sqrt = 30.

        sigma_x_2 = (0.5)**2.
        kappa1 = 3.0
        kappa2 = 5.0

        # Get the mu and gamma (means of receptive fields)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        means = np.array(cross(2*[coverage_1D.tolist()]))

        mu = means[:, 0]
        gamma = means[:, 1]

        precision = 1.
        stim_space = np.linspace(-np.pi, np.pi, precision, endpoint=False)
        stim_space = np.array(cross(stim_space, stim_space))

        ## Check that \sum_i f_i() is ~constant
        print np.mean(np.sum(np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))
        print np.std(np.sum(np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))
        IF_perneuron_multstim = np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))
        sum_IF_perneuron_multstim = np.mean(IF_perneuron_multstim, axis=1)

        plt.figure()
        plt.plot(sum_IF_perneuron_multstim)
        plt.ylim((0, 1.1*sum_IF_perneuron_multstim.max()))
        plt.title('Fisher Information constant for all neurons, when averaging over stimuli')

        ## Show how the FI is different per neuron, for a given stimulus
        stim = np.array([0., 0.])
        IF_perneuron = np.sin(stim[0] - mu[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim[1] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim[0] - gamma[:, np.newaxis]))


        plt.figure()
        plt.imshow(np.reshape(IF_perneuron, (N_sqrt, N_sqrt)), interpolation='nearest')
        plt.colorbar()
        plt.title('Local effects of Fisher Information')


    if False:
        ### Now check the dependence on N_sqrt
        sigma_x_2 = (0.5)**2.
        kappa1 = 3.0
        kappa2 = 5.0

        all_N_sqrt = np.arange(1, 21)
        all_FI_11 = np.zeros(all_N_sqrt.size)
        all_FI_12 = np.zeros(all_N_sqrt.size)
        all_FI_22 = np.zeros(all_N_sqrt.size)
        all_FI_11_limN = np.zeros(all_N_sqrt.size)
        all_FI_22_limN = np.zeros(all_N_sqrt.size)

        for i, N_sqrt in enumerate(all_N_sqrt):
            # Get the mu and gamma (means of receptive fields)
            coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
            means = np.array(cross(2*[coverage_1D.tolist()]))

            mu = means[:, 0]
            gamma = means[:, 1]

            precision = 1.
            stim_space = np.linspace(0, 2.*np.pi, precision, endpoint=False)
            stim_space = np.array(cross(stim_space, stim_space))

            # Full FI
            all_FI_11[i] = 1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

            all_FI_12[i] = 1./sigma_x_2 * kappa1*kappa2*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.sin(stim_space[:, 1] - gamma[:, np.newaxis])*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

            all_FI_22[i] = 1./sigma_x_2 * kappa2**2.*np.mean(np.sum(np.sin(stim_space[:, 1] - gamma[:, np.newaxis])**2.*np.exp(2.*kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + 2.*kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

            # Large N limit
            density = N_sqrt**2./(4.*np.pi**2.)
            all_FI_11_limN[i] = 1./sigma_x_2*density*2.*np.pi**2.*kappa1**2.0*scsp.i0(2*kappa2)*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))
            all_FI_22_limN[i] = 1./sigma_x_2*density*2.*np.pi**2.*kappa2**2.0*scsp.i0(2*kappa1)*(scsp.i0(2*kappa2) - scsp.iv(2, 2*kappa2))


        plt.plot(all_N_sqrt**2., all_FI_11, all_N_sqrt**2., all_FI_11_limN, all_N_sqrt**2., all_FI_12, all_N_sqrt**2., all_FI_22, all_N_sqrt**2., all_FI_22_limN)
        plt.xlabel('Number of neurons')
        plt.ylabel('Fisher Information')




    if False:
        ####
        # Getting the 'correct' Fisher information: check the FI for an average object
        N_sqrt = 20.

        # sigma_x_2 = (0.5)**2.
        kappa1 = 5.0
        kappa2 = kappa1
        alpha = 0.9999
        beta = 1.0

        # Get the mu and gamma (means of receptive fields)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        means = np.array(cross(2*[coverage_1D.tolist()]))

        mu = means[:, 0]
        gamma = means[:, 1]

        R = 2
        sigma_x = 2.0
        sigma_y = 0.001

        precision = 1.
        stim_space = np.linspace(0, 2.*np.pi, precision, endpoint=False)
        stim_space = np.array(cross(stim_space, stim_space))

        density = N_sqrt**2./(4.*np.pi**2.)

        T_all = np.arange(1, 7)
        FI_Tt = np.zeros((T_all.size, T_all.size))
        all_FI_11_T = np.zeros(T_all.size)
        covariance_all = np.zeros((T_all.size, int(N_sqrt**2.), int(N_sqrt**2.)))

        kappa1 = kappa1**0.5
        kappa2 = kappa1

        for T_i, T in enumerate(T_all):
            time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta=beta, specific_weighting=0.1, weight_prior='uniform')
            # TODO warning, kappa1 is being transformed in RFN...
            rn = RandomFactorialNetwork.create_full_conjunctive(int(N_sqrt**2.), R=R, scale_moments=(kappa1**2.0, 0.0001), ratio_moments=(1.0, 0.001))
            data_gen_noise = DataGeneratorRFN(4000, T, rn, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            covariance = stat_meas.model_parameters['covariances'][2][-1]

            # FI equal for different t, as alpha=1
            # for j, t in enumerate(xrange(T)):

            #     covariance_all[i, j] = covariance

            #     # Full FI
            #     all_FI_11_Tt[i, j] = alpha**(2.*T-2.*t)*1./sigma_x**2. * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.linalg.solve(covariance, np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))), axis=0))
                # all_FI_11_Tt[i, j] = alpha**(2.*T-2.*t + 2.)*1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

                # FI_Tt[i, j] = alpha**(2.*T-2.*t)*1./sigma_x**2.*density*2.*np.pi**2.*kappa1**2.0*scsp.i0(2*kappa2)*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))

            covariance_all[T_i] = covariance

            all_FI_11_T[T_i] = beta**2.0*kappa1**2.*np.mean(np.sum((np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])))*np.linalg.solve(covariance, np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))), axis=0))

        all_FI_11_T[all_FI_11_T == 0.0] = np.nan

        plt.figure()
        plt.plot(T_all, all_FI_11_T)

    if False:
        # Do the optimisation
        target_experimental_precisions  = np.array([5.0391, 3.4834, 2.9056, 2.2412, 1.7729])/2.

        N_sqrt = int(200.**0.5)

        # sigma_x_2 = (0.5)**2.
        kappa1 = 0.8
        kappa2 = kappa1

        # Get the mu and gamma (means of receptive fields)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        means = np.array(cross(2*[coverage_1D.tolist()]))
        mu = means[:, 0]
        gamma = means[:, 1]

        sigma_x = 2.0
        sigma_y = 0.001

        precision = 1.0
        stim_space = np.linspace(-np.pi, np.pi, precision, endpoint=False)
        stim_space = np.array(cross(stim_space, stim_space))

        T_all = np.arange(1, 6)

        # covariances_all = np.zeros((T_all.size, int(N_sqrt**2.), int(N_sqrt**2.)))

        # for T_i, T in enumerate(T_all):
        #     time_weights_parameters = dict(weighting_alpha=1.0, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
        #     rn = RandomFactorialNetwork.create_full_conjunctive(int(N_sqrt**2.), R=2, scale_moments=(1.0, 0.1), ratio_moments=(1.0, 0.2))
        #     data_gen_noise = DataGeneratorRFN(3000, T, rn, sigma_y=sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1)
        #     stat_meas = StatisticsMeasurer(data_gen_noise)
        #     covariances_all[T_i] = stat_meas.model_parameters['covariances'][2][-1]


        def compute_FI(sigma_x, kappa1, kappa2, T_all, stim_space, mu, gamma, covariances_all):
            FI_T = np.zeros(T_all.size)

            # Full FI
            for T_i, T in enumerate(T_all):
                FI_T[T_i] = (kappa1**2./(sigma_x**2.*16.*np.pi**4.0*scsp.i0(kappa1)**2.0*scsp.i0(kappa2)**2.0)*np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.linalg.solve(covariances_all[T_i], np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])))).mean(axis=-1).sum(axis=-1)
                # all_FI_11_Tt[i, j] = alpha**(2.*T-2.*t + 2.)*1./sigma_x_2 * kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis])), axis=0))

                # FI_Tt[i, j] = alpha**(2.*T-2.*t)*1./sigma_x**2.*density*2.*np.pi**2.*kappa1**2.0*scsp.i0(2*kappa2)*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))

            return FI_T


        def MSE_from_experimental(computed_FI, target_experimental_precisions):

            return np.sum((target_experimental_precisions - computed_FI)**2.)

        def MSE_powerlaw_experimental(computed_FI, target_experimental_precisions):

            return np.sum((fit_powerlaw(np.arange(1, 6), target_experimental_precisions) - fit_powerlaw(np.arange(1, 6), computed_FI))**2.)


        if False:
            #### See effect of sigma_x
            sigma_x_space = np.linspace(0.05, 10.0, 100.)

            sigma_x_MSE_experimental = np.zeros_like(sigma_x_space)
            sigma_x_powerlaw_params = np.zeros((sigma_x_space.size, 2))

            FI_all_sigma_x = np.zeros((sigma_x_space.size, T_all.size))
            for i, sigma_x in enumerate(sigma_x_space):
                FI_all_sigma_x[i] = compute_FI(sigma_x, kappa1, kappa2, T_all, stim_space, mu, gamma, covariances_all)

                # sigma_x_MSE_experimental[i] = MSE_from_experimental(FI_all_sigma_x[i], target_experimental_precisions)

                sigma_x_MSE_experimental[i] = MSE_powerlaw_experimental(FI_all_sigma_x[i], target_experimental_precisions)
                sigma_x_powerlaw_params[i] = fit_powerlaw(np.arange(1, 6), FI_all_sigma_x[i])

            plt.figure()
            plt.semilogy(sigma_x_space, sigma_x_MSE_experimental)

            # Show the best sigma
            best_sigma_x_ind = np.argmin(sigma_x_MSE_experimental)

            print "Best sigma_x : %.2f. MSE: %.2f" % (sigma_x_space[best_sigma_x_ind], sigma_x_MSE_experimental[best_sigma_x_ind])

            # Precision curve
            plt.figure()
            plt.plot(T_all, FI_all_sigma_x[best_sigma_x_ind])
            plt.plot(T_all, target_experimental_precisions)
            plt.legend(['Model', 'Experimental'])

            # 2D effect of sigma, not good
            # plt.figure()
            # plt.imshow(FI_all_sigma_x.T, norm=plt.matplotlib.colors.LogNorm())

            # 3D effect of sigma, nice enough
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(sigma_x_space, T_all)
            ax.plot_surface(X, Y, np.log(FI_all_sigma_x.T))

            # Effect of sigma on FI[0], not very interesting
            plt.figure()
            plt.semilogy(sigma_x_space, FI_all_sigma_x[:, 0])

            # Effect of sigma on the power law exponent: null.
            plt.figure()
            plt.plot(sigma_x_space, sigma_x_powerlaw_params[:, 0])

            # Effect of sigma on the power law y0 value: same as FI[0]
            plt.figure()
            plt.semilogy(sigma_x_space, sigma_x_powerlaw_params[:, 1])

        if True:
            sigma_x = 0.01
            beta = 1.0
            sigma_y = 0.01

            N_sqrt = 10.0

            N = int(N_sqrt**2.)

            coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
            means = np.array(cross(2*[coverage_1D.tolist()]))
            mu = means[:, 0]
            gamma = means[:, 1]

            #### See effect of kappa
            kappa_space = np.linspace(0.01, 15.0, 10.)
            # kappa_space = np.array([1.0])

            T_all = np.arange(1, 6)

            kappa_MSE_experimental = np.zeros_like(kappa_space)
            FI_all_kappa = np.zeros((kappa_space.size, T_all.size))
            kappa_powerlaw_params = np.zeros((kappa_space.size, 2))

            covariances_all = np.zeros((T_all.size, N, N))

            for i, kappa in enumerate(kappa_space):
                print "%.f%%, Doing kappa: %.2f" % (i*100./kappa_space.size, kappa)

                for T_i, T in enumerate(T_all):
                    time_weights_parameters = dict(weighting_alpha=1.0, weighting_beta=beta, specific_weighting=0.1, weight_prior='uniform')
                    rn = RandomFactorialNetwork.create_full_conjunctive(N, R=2, scale_moments=(kappa**2.0, 0.0001), ratio_moments=(1.0, 0.0001))
                    # Measure covariances
                    # data_gen_noise = DataGeneratorRFN(3000, T, rn, sigma_x=sigma_x, sigma_y=sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1)
                    # stat_meas = StatisticsMeasurer(data_gen_noise)
                    # covariances_all[T_i] = stat_meas.model_parameters['covariances'][2][-1]

                    # Computed covariances
                    covariances_all[T_i] = rn.compute_covariance_KL(sigma_2=(beta**2.0*sigma_x**2. + sigma_y**2.), T=T, beta=beta, precision=100)
                    # covariances_all[T_i] = sigma_x**2.*np.eye(N)

                FI_all_kappa[i] = compute_FI(sigma_x, kappa, kappa, T_all, stim_space, mu, gamma, covariances_all)

                # sigma_x_MSE_experimental[i] = MSE_from_experimental(FI_all_kappa[i], target_experimental_precisions)

                kappa_MSE_experimental[i] = MSE_powerlaw_experimental(FI_all_kappa[i], target_experimental_precisions)
                kappa_powerlaw_params[i] = fit_powerlaw(np.arange(1, 6), FI_all_kappa[i])

            # plt.figure()
            # plt.semilogy(kappa_space, kappa_MSE_experimental)

            # # Show the best sigma
            # best_kappa_ind = np.argmin(kappa_MSE_experimental)

            # print "Best kappa: %.2f. MSE: %.2f" % (kappa_space[best_kappa_ind], kappa_MSE_experimental[best_kappa_ind])

            # # Precision curve
            # plt.figure()
            # plt.plot(T_all, FI_all_kappa[best_kappa_ind])
            # plt.plot(T_all, target_experimental_precisions)
            # plt.legend(['Model', 'Experimental'])

            # # 2D effect of sigma, not good
            # # plt.figure()
            # # plt.imshow(FI_all_kappa.T, norm=plt.matplotlib.colors.LogNorm())

            # # 3D effect of kappa, nice enough. Smoother than sigmax
            # f = plt.figure()
            # ax = f.add_subplot(111, projection='3d')
            # X, Y = np.meshgrid(kappa_space, T_all)
            # ax.plot_surface(X, Y, np.log(FI_all_kappa.T))

            # # Effect of kappa on the power law exponent: null.
            # plt.figure()
            # plt.plot(kappa_space, kappa_powerlaw_params[:, 0])

            # # Effect of kappa on the power law y0 value: same as FI[0]
            # plt.figure()
            # plt.semilogy(kappa_space, kappa_powerlaw_params[:, 1])

            # Fisher information for single object, as function of kappa
            plt.figure()
            plt.plot(kappa_space, FI_all_kappa[:, 0])
            plt.xlabel('kappa')
            # plt.xlim([0, 10])
            # plt.ylim([0, 10000])
            plt.ylabel('FI for single item')


        if False:
            # Double optimisation!!
            #### See effect of kappa

            recompute_big_array = True

            if recompute_big_array:
                sigma_x_space = np.linspace(0.05, 20.0, 31.)
                kappa_space = np.linspace(0.05, 3.0, 30.)

                kappasigma_MSE_experimental = np.zeros((kappa_space.size, sigma_x_space.size))
                FI_all_kappasigma = np.zeros((kappa_space.size, sigma_x_space.size, T_all.size))
                kappasigma_powerlaw_params = np.zeros((kappa_space.size, sigma_x_space.size, 2))

                for i, kappa in enumerate(kappa_space):
                    print i*100./kappa_space.size
                    for j, sigma_x in enumerate(sigma_x_space):
                        FI_all_kappasigma[i, j] = compute_FI(sigma_x, kappa, kappa, T_all, stim_space, mu, gamma, covariances_all)

                        kappasigma_MSE_experimental[i, j] = MSE_from_experimental(FI_all_kappasigma[i, j], target_experimental_precisions)

                        # kappasigma_MSE_experimental[i, j] = MSE_powerlaw_experimental(FI_all_kappasigma[i, j], target_experimental_precisions)
                        # kappasigma_powerlaw_params[i, j] = fit_powerlaw(np.arange(1, 6), FI_all_kappasigma[i, j])

            # Show results, log z-axis
            f = plt.figure()
            ax = f.add_subplot(111)
            # ax.pcolor(kappa_space, sigma_x_space, kappasigma_MSE_experimental.T, norm=plt.matplotlib.colors.LogNorm())
            ax.pcolor(kappa_space, sigma_x_space, np.log(kappasigma_MSE_experimental.T))
            ax.set_xlim([kappa_space.min(), kappa_space.max()])

            # Callback function when moving mouse around figure.
            def report_pixel(x, y):
                # Extract loglik at that position
                x_i = (np.abs(kappa_space-x)).argmin()
                y_i = (np.abs(sigma_x_space-y)).argmin()
                v = np.log(kappasigma_MSE_experimental[x_i, y_i])
                return "x=%f y=%f value=%f" % (x, y, v)

            ax.format_coord = report_pixel

            # Show the best sigma
            best_kappasigma_ind = argmin_indices(kappasigma_MSE_experimental)

            print "Best kappa: %.2f, best sigma_x: %.2f, MSE: %.2f" % (kappa_space[best_kappasigma_ind[0]], sigma_x_space[best_kappasigma_ind[1]], kappasigma_MSE_experimental[best_kappasigma_ind])

            # Show the memory curves
            plt.figure()
            plt.plot(np.arange(1, 6), target_experimental_precisions)
            plt.plot(np.arange(1, 6), FI_all_kappasigma[best_kappasigma_ind])
            plt.legend(['Experimental', 'FI'])

    if False:
        # Fisher info, new try, 13.08.12
        kappa = 1.0
        kappa2 = 1.0
        sigma = 1.0

        N_sqrt = 1.
        N = int(N_sqrt**2.)
        coverage_1D = np.linspace( -np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)
        means = np.array(cross(2*[coverage_1D.tolist()]))

        precision_space = 200.
        phi_space = np.linspace(-np.pi, np.pi, precision_space)
        psi_space = np.linspace(-np.pi, np.pi, precision_space)

        # Slow way
        # fisher_info = np.zeros((phi_space.size, psi_space.size))
        # for i, phi in enumerate(phi_space):
        #     for j, psi in enumerate(psi_space):
        #         fisher_info[i, j] = (kappa1**2.)/(sigma**2.*16.*np.pi**2.*scsp.i0(kappa1)**2.0*scsp.i0(kappa1)**2.0)*np.sum(np.sin(phi - means[:, 0])**2.*np.exp(2.*kappa1*np.cos(phi - means[:, 0]) + 2.*kappa1*np.cos(psi - means[:, 1])))

        # Array way
        stimulus_space = np.array(cross(phi_space, psi_space))
        fisher_info_arr = (kappa**2.)/(sigma**2.*16.*np.pi**4.*scsp.i0(kappa)**2.0*scsp.i0(kappa)**2.0)*np.sin(stimulus_space[:, 0] - means[:, 0][:, np.newaxis])**2.*np.exp(2.*kappa*np.cos(stimulus_space[:, 0] - means[:, 0][:, np.newaxis]) + 2.*kappa*np.cos(stimulus_space[:, 1] - means[:, 1][:, np.newaxis]))
        mean_FI_total = fisher_info_arr.mean(axis=-1).sum()
        # fisher_info_arr.shape = (phi_space.size, psi_space.size)

        # Multiple kappas
        kappa_space = np.linspace(0.05, 15.0, 100.0)
        fisher_info_kappa = np.zeros((kappa_space.size, precision_space*precision_space))
        for i, kappa in enumerate(kappa_space):
            # IF_{psi psi}
            fisher_info_kappa[i] = ((kappa**2.)/(sigma**2.*16.*np.pi**4.*scsp.i0(kappa)**2.0*scsp.i0(kappa)**2.0)*np.sin(stimulus_space[:, 0] - means[:, 0][:, np.newaxis])**2.*np.exp(2.*kappa*np.cos(stimulus_space[:, 0] - means[:, 0][:, np.newaxis]) + 2.*kappa*np.cos(stimulus_space[:, 1] - means[:, 1][:, np.newaxis]))).mean(axis=-1).sum(axis=-1)

            # IF_{psi phi} = 0
            # fisher_info_kappa[i] = ((kappa**2.)/(sigma**2.*16.*np.pi**4.*scsp.i0(kappa)**2.0*scsp.i0(kappa)**2.0)*np.sin(stimulus_space[:, 0] - means[:, 0][:, np.newaxis])*np.sin(stimulus_space[:, 1] - means[:, 1][:, np.newaxis])*np.exp(2.*kappa*np.cos(stimulus_space[:, 0] - means[:, 0][:, np.newaxis]) + 2.*kappa*np.cos(stimulus_space[:, 1] - means[:, 1][:, np.newaxis]))).mean(axis=-1).sum(axis=-1)

        # Put the squareroot kappa in, as it is done in RandomFactorialNetwork
        fisher_info_kappasqrt = np.zeros((kappa_space.size, N, precision_space*precision_space))
        for i, kappa in enumerate(kappa_space):
            kappa = kappa**0.5
            fisher_info_kappasqrt[i] = (kappa**2.)/(sigma**2.*16.*np.pi**2.*scsp.i0(kappa)**2.0*scsp.i0(kappa)**2.0)*np.sin(stimulus_space[:, 0] - means[:, 0][:, np.newaxis])**2.*np.exp(2.*kappa*np.cos(stimulus_space[:, 0] - means[:, 0][:, np.newaxis]) + 2.*kappa*np.cos(stimulus_space[:, 1] - means[:, 1][:, np.newaxis]))

        # Plot
        plt.figure()
        plt.plot(kappa_space, fisher_info_kappa.mean(axis=-1))

        ### For N>1, just multiply FI_1 by N, as everything is translational invariant.
        N_other = 17.**2.
        plt.figure()
        plt.plot(kappa_space, fisher_info_kappa.mean(axis=-1)*N_other)

        plt.figure()
        plt.plot(kappa_space, fisher_info_kappasqrt.mean(axis=-1))


        plt.show()

    if False:
        # Compare Fisher Information and curvature of posterior

        # Start with gaussian: x | \theta ~ N(\theta, \sigma I)
        # I_F = sigma^-1

        theta = 0.1
        sigma = 0.1

        xx = np.linspace(-10., 10., 10000)
        dx = np.diff(xx)[0]

        yy = spst.norm.pdf(xx, theta, sigma)

        # Theoretical
        IF_theo = 1./sigma**2.

        # Estimated using IF = E [ (\partial_theta log P(x, \theta))^ 2]
        IF_estim_1 = np.trapz((np.diff(np.log(yy)))**2.*yy[1:]/dx**2., xx[1:])

        # Estimated using IF = E [ - \partial_theta \partial_theta log P(x, \theta)]
        IF_estim_2 = np.trapz(-np.diff(np.diff(np.log(yy)))*yy[1:-1]/dx**2., xx[1:-1])

        print "Theo: %.3f, Estim_1: %.3f, Estim_curv: %.3f" % (IF_theo, IF_estim_1, IF_estim_2)

    if False:
        # Compare scale of theory/posterior for different parameters.
        weighting_alpha=1.0
        R=2
        M=100
        rc_scale=2.0
        T=1
        sigma_y = 0.000001
        sigma_x = 0.2
        N=100

        # param_space = np.arange(50, 500, 50)  # M
        # param_space = np.linspace(0.000001, 2.0, 50)  # sigma_y
        param_space = np.linspace(0.1, 2.0, 20)  # sigma_x
        # param_space = np.linspace(0.01, 10.0, 20)  # rc_scale

        FI_M_effect_mean = np.zeros(param_space.size)
        FI_M_effect_std = np.zeros(param_space.size)

        for i, param in enumerate(param_space):
            # M = param
            # sigma_y = param
            sigma_x = param
            # rc_scale = param

            # Build the random network
            time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')
            cued_feature_time = T-1

            random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))

            # Construct the real dataset
            # print "Building the database"
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters = time_weights_parameters, cued_feature_time=cued_feature_time)

            # Measure the noise structure
            # print "Measuring noise structure"
            data_gen_noise = DataGeneratorRFN(2000, T, random_network, sigma_y = sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)

            sampler = Sampler(data_gen, n_parameters = stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, param %.3f" % param
            (FI_M_effect_mean[i], FI_M_effect_std[i], _) = sampler.estimate_fisher_info_from_posterior_avg(num_points=100, return_std=True)

        # Dependence: FI = M * (sigma_y + sigma_x)**-1 * rcscale**2

        print (FI_M_effect_mean, FI_M_effect_std)


        plot_mean_std_area(param_space, FI_M_effect_mean, FI_M_effect_std)





################################################################################
########################################################################################################################################################
################################################################################



    if True:
        ## Redoing everything from scratch.
        # First try the Fisher Info in 1D, and see if the relation on kappa is correct.

        def fisher_info_Ninf(kappa=1.0, rho=0.1, N=None, sigma=1.0):
            if N:
                rho = 1./(2*np.pi/(N))

            return kappa**2.*rho*(scsp.i0(2*kappa) - scsp.iv(2, 2*kappa))/(sigma**2.*4*np.pi*scsp.i0(kappa)**2.)

        def fisher_info_Ninf_bis(kappa=1.0, rho=0.1, N=None, sigma=1.0):
            if N:
                rho = 1./(2*np.pi/(N))

            return kappa*rho*scsp.i1(2*kappa)/(sigma**2.*4*np.pi*scsp.i0(kappa)**2.)

        def fisher_info_N(N=100, kappa=1.0, sigma=1.0):
            pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

            theta = np.linspace(0., 2*np.pi, 200.)
            # theta = 0.0

            return np.mean(np.sum(kappa**2.*np.sin(theta[:, np.newaxis] - pref_angles)**2.*np.exp(2*kappa*np.cos(theta[:, np.newaxis] - pref_angles))/(sigma**2.*4.*np.pi**2.*scsp.i0(kappa)**2.), axis=-1))
            # return np.sum(kappa**2.*np.sin(theta - pref_angles)**2.*np.exp(2*kappa*np.cos(theta - pref_angles))/(sigma**2.*4.*np.pi**2.*scsp.i0(kappa)**2.))


        if False:

            kappa_space = np.linspace(0., 100., 1000.)

            plt.figure()
            plt.plot(kappa_space, fisher_info_Ninf(kappa=kappa_space), kappa_space, fisher_info_Ninf_bis(kappa=kappa_space))
            plt.title('Two different defitinion of the large N limit equation')

            out = fisher_info_Ninf(kappa=kappa_space)
            slope = np.log(out[200]/out[150])/np.log(kappa_space[200]/kappa_space[150])
            print "Slope of Fisher Info: %.3f" % slope

            # Comparison between large N limit and exact.
            N_space = np.arange(1, 1000)
            fi_largeN = np.array([fisher_info_Ninf_bis(rho=1./(2*np.pi/n)) for n in N_space])
            fi_exact = np.array([fisher_info_N(N=n) for n in N_space])

            plt.figure()
            plt.plot(N_space, fi_exact, N_space, fi_largeN)
            plt.title('Comparison between large N limit equation and exact FI definition')

            plt.figure()
            plt.semilogy(N_space, (fi_exact - fi_largeN)**2., nonposy='clip')
            plt.title('Comparison between large N limit equation and exact FI definition')

            # Plot I_1(2k)/I_0(k)**2
            plt.figure()
            out = scsp.iv(1, 2*kappa_space)/scsp.i0(kappa_space)**2.
            plt.plot(kappa_space, out)
            # ~ kappa**0.5
            slope = np.log(out[200]/out[150])/np.log(kappa_space[200]/kappa_space[150])
            print "Slope of I_1(2k)/I_0(k)**2: %.3f" % slope

            # Plot \sum_i sin**2() exp(k cos()) / I_0(k)**2

            # Plot exp(2 k)/4pi**2 I_0(k)**2
            # not smaller than 1
            plt.figure()
            plt.plot(np.exp(2*kappa_space)/(4*np.pi**2.*scsp.i0(kappa_space)**2.))


        ## Redo everything here.
        if True:

            ## Population
            N     = 100
            kappa = 3.0
            sigma = 0.2
            amplitude = 1.0

            put_noise_dataset = True
            use_slice_sampler = True

            kappa_space = np.linspace(0.01, 5., 10)
            # kappa_space = np.linspace(5.0, 5.0, 1.)
            # kappa_space = np.array([3.0])

            # N_space = np.array([100, 200, 300, 500])
            N_space = np.array([100])

            # Dataset size.
            #  Big number required for clean estimate of recall precision...
            M = 100


            def population_code_response(theta, pref_angles=None, N=100, kappa=0.1, amplitude=1.0):

                if pref_angles is None:
                    pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

                return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))


            search_progress = progress.Progress(kappa_space.size*M*N_space.size)

            for N_i, N in enumerate(N_space):

                print "N %d" % N

                effects_kappa = []
                effects_kappa_std = []

                all_gauss_fits = []
                samples = np.zeros(500)

                # pref_angles = np.linspace(0.0, 2.*np.pi, N, endpoint=False)
                pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

                for k, kappa in enumerate(kappa_space):
                    # print '%d DOING KAPPA: %.3f' % (k, kappa)

                    ## Generate dataset

                    # stimuli_used = np.random.rand(M)*np.pi*2.
                    # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
                    # stimuli_used = (np.random.rand(M) - 0.5)*np.pi/3.
                    stimuli_used = np.ones(M)*0.0
                    # stimuli_used = np.ones(M)*np.pi

                    dataset = np.zeros((M, N))
                    for i, stim in enumerate(stimuli_used):
                        if put_noise_dataset:
                            dataset[i] = population_code_response(stim, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude) + sigma*np.random.randn(N)
                        else:
                            dataset[i] = population_code_response(stim, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                    ## Estimate likelihood
                    num_points = 1000
                    # num_points_space = np.arange(50, 1000, 200)
                    # effects_num_points = []

                    # for k, num_points in enumerate(num_points_space):

                    # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
                    all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

                    def likelihood(data, all_angles, N=100, kappa=0.1, sigma=1.0, should_exponentiate=False, remove_mean=False):

                        lik = np.zeros(all_angles.size)
                        for i, angle in enumerate(all_angles):
                            # lik[i] = -np.log((2*np.pi)**0.5*sigma) -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)
                            lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))**2.)

                        if remove_mean:
                            lik -= np.mean(lik)

                        if should_exponentiate:
                            lik = np.exp(lik)

                        return lik


                    def lik_sampler(angle, params):
                        sigma = params['sigma']
                        data = params['data']
                        N = params['N']
                        kappa = params['kappa']
                        amplitude = 1.0
                        pref_angles = params['pref_angles']
                        return -1./(2.*sigma**2.0)*np.sum((data - population_code_response(angle, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))**2.)

                    ## Estimate fisher info
                    # print "Estimate fisher info"
                    fisher_info_curve = np.zeros(M)
                    fisher_info_prec = np.zeros(M)
                    gauss_fits = np.zeros((M, 2))
                    true_fits = np.zeros((M, 2))
                    dx = np.diff(all_angles)[0]

                    samples_all_precisions = []
                    recall_samples = np.zeros(M)
                    recall_samples_gauss = np.zeros(M)

                    for m, data in enumerate(dataset):
                        if search_progress.percentage() % 5.0 < 0.001:
                            print "KAPPA %.3f %d. %.2f%%, %s left - %s" % (kappa, k, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                        posterior = likelihood(data, all_angles, N=N, kappa=kappa, sigma=sigma, should_exponentiate=True)
                        # log_posterior = likelihood(data, all_angles, N=N, kappa=kappa, sigma=sigma, should_exponentiate=False)
                        log_posterior = np.log(posterior)

                        log_posterior[np.isinf(log_posterior)] = 0.0
                        log_posterior[np.isnan(log_posterior)] = 0.0

                        # posterior = np.exp(log_posterior)
                        posterior /= np.sum(posterior*dx)

                        # Fails when angles are close to 0/2pi.
                        # Could roll the posterior around to center it, wouldn't be that bad.
                        # fisher_info_curve[m] = np.trapz(-np.diff(np.diff(log_posterior))*posterior[1:-1]/dx**2., all_angles[1:-1])

                        # Actually wrong, see Issue #23
                        # fisher_info_curve[m] = np.trapz(-np.gradient(np.gradient(log_posterior))*posterior/dx**2., all_angles)

                        # Take curvature at ML value
                        ml_index = np.argmax(posterior)
                        curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.
                        # fisher_info_curve[m] = curv_logp[ml_index]
                        fisher_info_curve[m] = 1./curv_logp[ml_index]

                        # Fit a gaussian to it
                        gauss_fits[m] = fit_gaussian(all_angles, posterior, return_fitted_data=False, should_plot = False)[:2]
                        # Sample from this gaussian instead
                        samples_gauss = gauss_fits[m, 0] + gauss_fits[m, 1]*np.random.randn(500)
                        fisher_info_prec[m] = np.var(samples_gauss)

                        #fisher_info_prec[m] = 1./fit_gaussian(all_angles, posterior, should_plot=False, return_fitted=False)[1]**2.
                        # fisher_info_prec[m] = 1./(-2.*np.log(np.abs(np.trapz(posterior*np.exp(1j*all_angles), all_angles))))
                        # fisher_info_prec[m] = (-2.*np.log(np.abs(np.trapz(posterior*np.exp(1j*all_angles), all_angles))))

                        # Using samples, estimate the precision for each data
                        if use_slice_sampler:
                            params = dict(sigma=sigma, data=data, N=N, kappa=kappa, pref_angles=pref_angles)
                            samples, _ = slicesampler.sample_1D_circular(500, np.random.rand()*2.*np.pi-np.pi, lik_sampler, burn=100, thinning=1, widths=np.pi/8., loglike_fct_params=params, debug=False, step_out=True)

                        # samples_circ_std_dev = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*samples), axis=0))))
                        # samples_circ_std_dev = np.std(samples)
                        samples_circ_std_dev = np.var(samples)
                        # samples_all_precisions.append(1./samples_circ_std_dev**2.)
                        # samples_all_precisions.append(samples_circ_std_dev**2.)
                        samples_all_precisions.append(samples_circ_std_dev)

                        # Now sample one angle, and compute the fisher information from the distribution of the recall samples
                        # choose last one
                        # recall_samples[m] = np.median(samples[-100:])
                        recall_samples[m] = samples[-1]
                        recall_samples_gauss[m] = samples_gauss[-1]

                        # Estimate the true mean and variance of the current posterio
                        true_fits[m, 0] = np.trapz(posterior*all_angles, all_angles)
                        true_fits[m, 1] = np.trapz(posterior*(all_angles - true_fits[m, 0])**2., all_angles)**0.5

                        search_progress.increment()


                    fisher_info_curve_mean = np.mean(fisher_info_curve)
                    fisher_info_curve_std = np.std(fisher_info_curve)
                    fisher_info_prec_mean = np.mean(fisher_info_prec)
                    fisher_info_prec_std = np.std(fisher_info_prec)

                    samples_precisions_mean = np.mean(samples_all_precisions)
                    samples_precisions_std = np.std(samples_all_precisions)

                    all_gauss_fits.append(gauss_fits)

                    # recall_samples_precision = 1./np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2
                    # recall_samples_precision = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2
                    recall_samples_precision = np.var(recall_samples - stimuli_used)
                    recall_samples_gauss_precision = np.var(recall_samples_gauss - stimuli_used)

                    recall_precision_corrected = recall_samples_gauss_precision - np.mean((gauss_fits[:, 0] - stimuli_used)**2.)

                    bias_ratio = np.mean((gauss_fits[:, 0] - stimuli_used)**2.)/fisher_info_curve_mean

                    # Save it
                    effects_kappa.append([fisher_info_curve_mean, fisher_info_prec_mean, 1./fisher_info_N(N=N, kappa=kappa, sigma=sigma), 1./fisher_info_Ninf(kappa=kappa, N=N, sigma=sigma), samples_precisions_mean, recall_samples_precision, recall_samples_gauss_precision])
                    effects_kappa_std.append([fisher_info_curve_std, fisher_info_prec_std, 0, 0, samples_precisions_std, 0, 0])

                    # effects_num_points.append((fisher_info_curve_mean, fisher_info_curve_std))

                    print "FI curve: %.3f, FI precision: %.3f, Samples: %.3f, Recall precision: %.3f, Recall gauss precision: %.3f, recall precision corrected: %.3f, Theo: %.3f, Theo large N: %.3f, ratio prec/curv: %.3f, ratio prec_g/curv: %.3f, ratio prec_corr/curv: %.3f" % (fisher_info_curve_mean, fisher_info_prec_mean, samples_precisions_mean, recall_samples_precision, recall_samples_gauss_precision, recall_precision_corrected, 1./fisher_info_N(N=N, kappa=kappa, sigma=sigma), 1./fisher_info_Ninf(kappa=kappa, N=N, sigma=sigma), recall_samples_precision/fisher_info_curve_mean, recall_samples_gauss_precision/fisher_info_curve_mean, recall_precision_corrected/fisher_info_curve_mean)

                    print "Difference between true mean/std and gaussian fits: MSE(mu - mu_g)=%.3g , MSE(std - std_g)=%.3g" % tuple(np.mean((true_fits - gauss_fits)**2., axis=0))



                # plot_mean_std_area(num_points_space, np.array(effects_num_points)[:, 0], np.array(effects_num_points)[:, 1])
                effects_kappa = np.array(effects_kappa)
                effects_kappa_std = np.array(effects_kappa_std)

                all_gauss_fits = np.array(all_gauss_fits)

                # No small N / big kappa effect on Fisher information.
                plot_multiple_mean_std_area(kappa_space, effects_kappa.T, effects_kappa_std.T)

                plt.legend(['Curvature', 'Posterior precision', 'Theo', 'Theo large N', 'Samples', 'Recall precision'])

                plt.figure()
                plt.boxplot([fisher_info_curve, samples_all_precisions, recall_samples_precision, recall_samples_gauss_precision])
                plt.title('Comparison Curvature vs samples estimate vs recall precision. 1D pop code')
                plt.xticks([1, 2, 3, 4], ['Curvature', 'Samples', 'Precision', 'Precision gauss'], rotation=45)


                plt.figure()
                plt.plot(kappa_space, effects_kappa[:, -1]/effects_kappa[:, 2])
                plt.title('Ratio FI precision/curve, N %d' % N)

                print "N: %d" % N

                print "Ratio: prec/theo %f, prec/curve %f" % (np.mean(effects_kappa[:, -1]/effects_kappa[:, 2]), np.mean((effects_kappa[:, -1]/effects_kappa[:, 0])))
                print (effects_kappa[:, -1]/effects_kappa[:, 2]), (effects_kappa[:, -1]/effects_kappa[:, 0])

                e_sigma_minus_fi = (np.mean(all_gauss_fits[..., 1], axis=1)**2. - effects_kappa[:, 2])
                print "E[sigma_i^2 - FI], for each fitted gaussian: %f +- %f" % (np.mean(e_sigma_minus_fi), np.std(e_sigma_minus_fi))
                print e_sigma_minus_fi

                e_inv_sigma_minus_sigma = (1./np.mean(1./all_gauss_fits[..., 1]**2., axis=1) - np.mean(all_gauss_fits[..., 1], axis=1)**2.)
                print "Diff between E[sigma] and 1/E[sigma^-1]: %f +- %f" % (np.mean(e_inv_sigma_minus_sigma), np.std(e_inv_sigma_minus_sigma))
                print e_inv_sigma_minus_sigma



    if True:
        # Now do everything for 2D population code.

        N     = (15.)**2
        N_sqrt = int(np.sqrt(N))
        kappa1 = 1.0
        kappa2 = 1.0
        sigma = 0.2
        amplitude = 1.0

        ## Fisher info
        def fisher_info_2D_N(stimulus_space=None, pref_angles=None, N=100, kappa1=1.0, kappa2=1.0, sigma=1.0):
            if pref_angles is None:
                N_sqrt = int(np.sqrt(N))
                coverage_1D = np.linspace(0., 2*np.pi, N_sqrt, endpoint=False)
                pref_angles = np.array(cross(2*[coverage_1D.tolist()]))

            if stimulus_space is None:
                phi_space = np.linspace(0., 2*np.pi, 20., endpoint=False)
                psi_space = np.linspace(0., 2*np.pi, 20., endpoint=False)
                stimulus_space = np.array(cross(phi_space, psi_space))
            # theta = 0.0

            return np.mean(np.sum(kappa1**2.*np.sin(stimulus_space[:, 0, np.newaxis] - pref_angles[:, 0])**2.*np.exp(2*kappa1*np.cos(stimulus_space[:, 0, np.newaxis] - pref_angles[:, 0]) + 2*kappa2*np.cos(stimulus_space[:, 1, np.newaxis] - pref_angles[:, 1]))/(sigma**2.*16.*np.pi**4.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.), axis=-1))
            # return np.sum(kappa**2.*np.sin(theta - pref_angles)**2.*np.exp(2*kappa*np.cos(theta - pref_angles))/(sigma**2.*4.*np.pi**2.*scsp.i0(kappa)**2.))

        def fisher_info_2D_Ninf(N=100, kappa1=1.0, kappa2=1.0, sigma=1.0):
            if N:
                rho = 1./(2*np.pi/(N))

            return kappa1**2.*rho*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))*scsp.i0(2*kappa2)/(sigma**2.*16*np.pi**3.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)


        ## Population
        N_sqrt = int(np.sqrt(N))
        coverage_1D = np.linspace(0., 2*np.pi, N_sqrt, endpoint=False)
        pref_angles = np.array(cross(2*[coverage_1D.tolist()]))


        def population_code_response_2D(theta, pref_angles=None, N=100, kappa1=0.1, kappa2=0.1, amplitude=1.0):
            if pref_angles is None:
                N_sqrt = int(np.sqrt(N))
                coverage_1D = np.linspace(0., 2*np.pi, N_sqrt, endpoint=False)
                pref_angles = np.array(cross(2*[coverage_1D.tolist()]))

            return amplitude*np.exp(kappa1*np.cos(theta[0] - pref_angles[:, 0]) + kappa2*np.cos(theta[1] - pref_angles[:, 1]))/(4.*np.pi**2.*scsp.i0(kappa1)*scsp.i0(kappa2))

        def show_population_output(data):
            N_sqrt = int(np.sqrt(data.size))
            plt.figure()
            plt.imshow(data.reshape((N_sqrt, N_sqrt)).T, origin='left', interpolation='nearest')
            plt.show()


        def likelihood_2D(data, all_angles, pref_angles=None, N=100, kappa1=0.1, kappa2=0.1, sigma=1.0, should_exponentiate=True, remove_mean=False):
                if pref_angles is None:
                    N_sqrt = int(np.sqrt(N))
                    coverage_1D = np.linspace(0., 2*np.pi, N_sqrt, endpoint=False)
                    pref_angles = np.array(cross(2*[coverage_1D.tolist()]))

                lik = np.zeros(all_angles.shape[0])
                for i, angle in enumerate(all_angles):
                    lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response_2D(angle, pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, amplitude=amplitude))**2.)

                if remove_mean:
                    lik -= np.mean(lik)

                if should_exponentiate:
                    lik = np.exp(lik)

                return lik

        def likelihood_2D_clamped(data, theta2, angles_1D=None, pref_angles=None, N=100, kappa1=0.1, kappa2=0.1, sigma=1.0, should_exponentiate=True, remove_mean=True):
            if pref_angles is None:
                N_sqrt = int(np.sqrt(N))
                coverage_1D = np.linspace(0., 2*np.pi, N_sqrt, endpoint=False)
                pref_angles = np.array(cross(2*[coverage_1D.tolist()]))

            if angles_1D is None:
                angles_1D = np.linspace(0., 2.*np.pi, 1000, endpoint=False)

            lik = np.zeros(angles_1D.shape[0])
            for i, theta1 in enumerate(angles_1D):
                lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response_2D([theta1, theta2], pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, amplitude=amplitude))**2.)

            if remove_mean:
                lik -= np.mean(lik)

            if should_exponentiate:
                lik = np.exp(lik)

            return lik

        num_points = 500

        angles_1D = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
        all_angles = np.array(cross(angles_1D, angles_1D))

        angles_clamped_fi = np.linspace(0., 2.*np.pi, 1000, endpoint=False)

        # kappa_space = np.linspace(0.001, 20., 5)
        kappa_space = np.linspace(2., 2., 1)

        effects_kappa_mean = []
        effects_kappa_std = []
        effects_kappa_quantiles = []

        fisher_info_curve_clamped_all = []


        for k, kappa1 in enumerate(kappa_space):
            print 'DOING KAPPA: %.3f' % kappa1

            kappa2 = kappa1

            ## Generate dataset
            M = 100
            # stimuli_used = np.random.rand(M, 2)*np.pi*2.
            stimuli_used = np.random.rand(M, 2)*np.pi/2. + np.pi*3/4.
            stimuli_used[0] = np.array([np.pi-0.4, np.pi+0.4])

            dataset = np.zeros((M, N))
            for i, stim in enumerate(stimuli_used):
                dataset[i] = population_code_response_2D(stim, pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, amplitude=amplitude) + sigma*np.random.randn(N)

            ## Estimate fisher info
            print "Estimate fisher info"
            fisher_info_curve = np.zeros(M)
            fisher_info_curve_clamped = np.zeros(M)
            fisher_info_prec = np.zeros(M)
            dx = np.diff(all_angles[:, 0]).max()
            dx_clamped = np.diff(angles_clamped_fi)[0]

            for m, data in progress.ProgressDisplay(enumerate(dataset), display=progress.SINGLE_LINE):

                # posterior = likelihood_2D(data, all_angles, pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, sigma=sigma)
                # # posterior += 1e-310

                # log_posterior = np.log(posterior)

                # log_posterior[np.isinf(log_posterior)] = 0.0
                # log_posterior[np.isnan(log_posterior)] = 0.0

                # posterior /= np.sum(posterior*dx**2.)

                # # Full fisher info
                # fisher_info_curve[m] = np.trapz(np.trapz((-np.gradient(np.gradient(log_posterior.reshape((num_points, num_points)))[0])[0]*posterior.reshape((num_points, num_points))/dx**2.), angles_1D), angles_1D)

                # Clamped fisher info
                posterior_clamped = likelihood_2D_clamped(data, stimuli_used[m, 1], angles_1D=angles_clamped_fi, pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, sigma=sigma)
                log_posterior_clamped = np.log(posterior_clamped)
                log_posterior_clamped[np.isinf(log_posterior_clamped)] = 0.0
                log_posterior_clamped[np.isnan(log_posterior_clamped)] = 0.0
                posterior_clamped /= np.sum(posterior_clamped*dx_clamped)

                # Incorrect here, see Issue #23
                # fisher_info_curve_clamped[m] = np.trapz(-np.gradient(np.gradient(log_posterior_clamped))*posterior_clamped/dx_clamped**2., angles_clamped_fi)
                ml_index = np.argmax(posterior_clamped)
                curv_logp = -np.gradient(np.gradient(log_posterior_clamped))/dx_clamped**2.
                fisher_info_curve_clamped[m] = curv_logp[ml_index]

                #fisher_info_prec[m] = 1./fit_gaussian(all_angles, posterior, should_plot=False, return_fitted=False)[1]**2.
                fisher_info_prec[m] = 1./(-2.*np.log(np.abs(np.trapz(posterior_clamped*np.exp(1j*angles_clamped_fi), angles_clamped_fi))))

            fisher_info_curve_mean = np.mean(fisher_info_curve)
            fisher_info_curve_std = np.std(fisher_info_curve)

            fisher_info_prec_mean = np.mean(fisher_info_prec)
            fisher_info_prec_std = np.std(fisher_info_prec)
            fisher_info_prec_quantiles = spst.mstats.mquantiles(fisher_info_prec)

            fisher_info_curve_clamped_mean = np.mean(fisher_info_curve_clamped)
            fisher_info_curve_clamped_std = np.std(fisher_info_curve_clamped)
            fisher_info_curve_clamped_quantiles = spst.mstats.mquantiles(fisher_info_curve_clamped)

            fi_theo_N = fisher_info_2D_N(pref_angles=pref_angles, N=N, kappa1=kappa1, kappa2=kappa2, sigma=sigma)
            fi_theo_Ninf = fisher_info_2D_Ninf(N=N, kappa1=kappa1, kappa2=kappa2, sigma=sigma)

            print "FI precision: %.3f, FI clamped: %.3f, Theo: %.3f, Theo large N: %.3f" % (fisher_info_prec_mean, fisher_info_curve_clamped_mean, fi_theo_N, fi_theo_Ninf)

            effects_kappa_mean.append((fisher_info_prec_mean, fisher_info_curve_clamped_mean, fi_theo_N, fi_theo_Ninf))
            effects_kappa_std.append((fisher_info_prec_std, fisher_info_curve_clamped_std, 0.0, 0.0))
            effects_kappa_quantiles.append((fisher_info_prec_quantiles, fisher_info_curve_clamped_quantiles, np.array([fi_theo_N, ]*3), np.array([fi_theo_Ninf, ]*3)))
            fisher_info_curve_clamped_all.append(fisher_info_curve_clamped)


        effects_kappa_mean = np.array(effects_kappa_mean)
        effects_kappa_std = np.array(effects_kappa_std)
        fisher_info_curve_clamped_all = np.array(fisher_info_curve_clamped_all)
        effects_kappa_quantiles = np.array(effects_kappa_quantiles)

        plot_multiple_mean_std_area(kappa_space, effects_kappa_mean.T, effects_kappa_std.T)

        plt.legend(['Precision from 1D Clamped', '1D Clamped curvature', '2D Theo', '2D Theo large N'])

        plot_multiple_median_quantile_area(kappa_space, quantiles=effects_kappa_quantiles.transpose(1, 0, 2))
        plt.legend(['Precision from 1D Clamped', '1D Clamped curvature', '2D Theo', '2D Theo large N'])

    if False:
        # Try with true gaussian to see the effect of precision estimation
        N     = 1
        # kappa = 1.0
        sigma = 2.0
        M = 1000
        num_points = 1000

        # stimuli_used = np.random.rand(M) - 0.5
        # stimuli_used = np.random.randn(M)
        # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
        stimuli_used = np.ones(M)*0.0

        # dataset = np.zeros((M, N))
        # for i, stim in enumerate(stimuli_used):
            # dataset[i] = stimuli_used[i] + sigma*np.random.randn(N)
        dataset = stimuli_used[:, np.newaxis] + sigma*np.random.randn(M, N)
        # dataset = stimuli_used[:, np.newaxis]


        def loglik_sampler(angle, params):
            sigma = params['sigma']
            data = params['data']
            return -1./(2.*sigma**2.)*np.sum((data - angle)**2.)

        def loglik_fullspace(all_angles, params):
            lik = np.zeros(all_angles.size)
            for i, angle in enumerate(all_angles):
                lik[i] = loglik_sampler(angle, params)

            return lik

        all_angles = np.linspace(-5., 5., num_points, endpoint=False)
        dx = np.diff(all_angles)[0]

        fisher_info_curve = np.zeros(M)
        samples_all_precisions = np.zeros(M)
        recall_samples = np.zeros(M)

        for m in progress.ProgressDisplay(np.arange(M), display=progress.SINGLE_LINE):
            # print m

            params = dict(sigma=sigma, data=dataset[m])
            log_posterior = loglik_fullspace(all_angles, params)

            log_posterior[np.isinf(log_posterior)] = 0.0
            log_posterior[np.isnan(log_posterior)] = 0.0

            # Take curvature at ML value
            ml_index = np.argmax(log_posterior)
            curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.
            # fisher_info_curve[m] = curv_logp[ml_index]
            fisher_info_curve[m] = 1./curv_logp[ml_index]

            # Precision from samples
            # samples, _ = slicesampler.sample_1D(500, 0.0, loglik_sampler, thinning=2, burn=100, widths=np.pi/3., loglike_fct_params=params, debug=False, step_out=True)
            samples = dataset[m] + sigma*np.random.randn(500)

            samples_circ_std_dev = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*samples), axis=0))))
            # samples_circ_std_dev = np.std(samples)
            samples_all_precisions[m] = samples_circ_std_dev**2.

            # Now sample one angle, and compute the fisher information from the distribution of the recall samples
            # choose last one
            # recall_samples[m] = np.median(samples[-1:])
            recall_samples[m] = samples[-1]
            # recall_samples[m] = samples[np.random.randint(500)]

        # recall_samples_precision = 1./np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2
        recall_samples_precision = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2
        # recall_samples_precision = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples)), axis=0))))**2
        # recall_samples_precision = np.var(recall_samples - stimuli_used)

        plt.figure()
        plt.boxplot([fisher_info_curve, samples_all_precisions, recall_samples_precision])
        plt.title('Comparison Curvature vs samples estimate vs recall precision. Simple gaussian.')
        plt.xticks([1, 2, 3], ['Curvature', 'Samples', 'Precision'], rotation=45)

        print np.mean(fisher_info_curve), np.mean(samples_all_precisions), recall_samples_precision
        print np.mean(fisher_info_curve/recall_samples_precision)

    if False:
        ### Check precision/curvature ratio as fct of sigma

        N     = 1
        # kappa = 1.0
        sigma = 0.2
        M = 500
        num_points = 1000

        sigma_space = np.linspace(0.05, 0.8, 5.)
        ratio_precisioncurv = np.zeros(sigma_space.size)

        def loglik_sampler(angle, params):
            sigma = params['sigma']
            data = params['data']
            return -1./(2.*sigma**2.0)*np.sum((data - angle)**2.)

        def loglik_fullspace(all_angles, params):
            lik = np.zeros(all_angles.size)
            for i, angle in enumerate(all_angles):
                lik[i] = loglik_sampler(angle, params)

            return lik

        for sigma_i, sigma in enumerate(sigma_space):

            print "===> Sigma: %.3f" % sigma

            # stimuli_used = np.random.rand(M)*np.pi*2.
            # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
            stimuli_used = np.ones(M)*1.

            dataset = np.zeros((M, N))
            for i, stim in enumerate(stimuli_used):
                dataset[i] = stimuli_used[i] + sigma*np.random.randn(N)

            all_angles = np.linspace(-3.*np.pi, 3.*np.pi, num_points, endpoint=False)
            dx = np.diff(all_angles)[0]


            fisher_info_curve = np.zeros(M)
            samples_all_precisions = np.zeros(M)
            recall_samples = np.zeros(M)

            for m, data in enumerate(dataset):
                # print m

                params = dict(sigma=sigma, data=data)
                log_posterior = loglik_fullspace(all_angles, params)

                log_posterior[np.isinf(log_posterior)] = 0.0
                log_posterior[np.isnan(log_posterior)] = 0.0

                # Take curvature at ML value
                ml_index = np.argmax(log_posterior)
                curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.
                fisher_info_curve[m] = curv_logp[ml_index]

                # Precision from samples
                samples, _ = slicesampler.sample_1D_circular(300, np.random.rand()*2.*np.pi-np.pi, loglik_sampler, burn=100, widths=np.pi/3., loglike_fct_params=params, debug=False, step_out=True)

                samples_circ_std_dev = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*samples), axis=0))))
                samples_all_precisions[m] = 1./samples_circ_std_dev**2.

                # Now sample one angle, and compute the fisher information from the distribution of the recall samples
                # choose last one
                recall_samples[m] = np.median(samples[-1:])

            recall_samples_precision = 1./np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2

            ratio_precisioncurv[sigma_i] = np.mean(fisher_info_curve/recall_samples_precision)

            print ratio_precisioncurv


        plt.figure()
        plt.plot(sigma_space, ratio_precisioncurv)




    if False:
        ########
        ###
        ###
        ### SEE COMPUTATIONS_MARGINALFISHERINFO_MARGINALPOSTERIOR_1D.PY
        ###
        ###
        ########

        ####
        #   1D two stimuli
        ####

        N     = 100
        kappa = 3.0
        sigma = 0.2
        amplitude = 1.0
        min_distance = 0.001

        put_noise_dataset = True
        use_slice_sampler = False

        # Dataset size.
        #  Big number required for clean estimate of recall precision...


        def population_code_response(theta, pref_angles=None, N=100, kappa=0.1, amplitude=1.0):
            if pref_angles is None:
                pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

            return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))


        samples = np.zeros(500)

        pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

        ## Generate dataset

        # stimuli_used = np.random.rand(M)*np.pi*2.
        # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
        # stimuli_used = (np.random.rand(M) - 0.5)*np.pi/3.
        # stimuli_used_1 = np.ones(M)*0.0
        # stimuli_used = np.ones(M)*np.pi


        ## Estimate likelihood
        num_points = 500
        # num_points_space = np.arange(50, 1000, 200)
        # effects_num_points = []

        # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

        theta1_space = np.array([0.])
        theta2_space = all_angles

        def likelihood(data, all_angles, N=100, kappa=0.1, sigma=1.0, should_exponentiate=False, remove_mean=False):

            lik = np.zeros(all_angles.size)
            for i, angle in enumerate(all_angles):
                # lik[i] = -np.log((2*np.pi)**0.5*sigma) -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)
                lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))**2.)

            if remove_mean:
                lik -= np.mean(lik)

            if should_exponentiate:
                lik = np.exp(lik)

            return lik


        def likelihood2(data, all_angles, stim2=0.0, N=100, kappa=0.1, sigma=1.0, should_exponentiate=False, remove_mean=False):

            lik = np.zeros(all_angles.size)
            for i, angle in enumerate(all_angles):
                # lik[i] = -np.log((2*np.pi)**0.5*sigma) -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)
                lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude) - population_code_response(stim2, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))**2.)

            if remove_mean:
                lik -= np.mean(lik)

            if should_exponentiate:
                lik = np.exp(lik)

            return lik


        def lik_sampler(angle, params):
            sigma = params['sigma']
            data = params['data']
            N = params['N']
            kappa = params['kappa']
            amplitude = 1.0
            pref_angles = params['pref_angles']
            return -1./(2.*sigma**2.0)*np.sum((data - population_code_response(angle, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))**2.)

        def enforce_distance(theta1, theta2, min_distance=0.1):
            return np.abs(wrap_angles(theta1 - theta2)) > min_distance

        #### Compute Theo Inverse Fisher Info

        if False:
            ### Loop over min_distance and kappa
            min_distance_space = np.linspace(0.0, 1.5, 10)
            # min_distance_space = np.array([min_distance])
            # min_distance_space = np.array([0.001])
            # kappa_space = np.linspace(0.05, 30., 40.)
            kappa_space = np.array([kappa])

            inv_FI_search = np.zeros((min_distance_space.size, kappa_space.size))
            FI_search = np.zeros((min_distance_space.size, kappa_space.size))
            inv_FI_1_search = np.zeros((min_distance_space.size, kappa_space.size))

            search_progress = progress.Progress(min_distance_space.size*kappa_space.size)

            print "Doing from marginal FI"

            for m, min_distance in enumerate(min_distance_space):
                for k, kappa in enumerate(kappa_space):

                    if search_progress.percentage() % 5.0 < 0.0001:
                        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                    inv_FI_all = np.zeros((theta1_space.size, theta2_space.size))
                    FI_all = np.zeros((theta1_space.size, theta2_space.size))
                    inv_FI_1 = np.zeros(theta1_space.size)

                    # Check inverse FI for given min_distance and kappa
                    for i, theta1 in enumerate(theta1_space):
                        der_1 = kappa*np.sin(pref_angles - theta1)*population_code_response(theta1, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                        for j, theta2 in enumerate(theta2_space):

                            if enforce_distance(theta1, theta2, min_distance=min_distance):
                                # Only compute if theta1 different enough of theta2

                                der_2 = kappa*np.sin(pref_angles - theta2)*population_code_response(theta2, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                                # FI for 2 objects
                                FI_all[i, j] = np.sum(der_1**2.)/sigma**2.

                                # Inv FI for 2 objects
                                inv_FI_all[i, j] = sigma**2./(np.sum(der_1**2.) - np.sum(der_1*der_2)**2./np.sum(der_2**2.))

                        # FI for 1 object
                        inv_FI_1[i] = sigma**2./np.sum(der_1**2.)

                    # inv_FI_search[m, k] = np.mean(inv_FI_all)
                    inv_FI_search[m, k] = np.mean(np.ma.masked_equal(inv_FI_all, 0))
                    FI_search[m, k] = np.mean(FI_all)

                    inv_FI_1_search[m, k] = np.mean(inv_FI_1)

                    search_progress.increment()

            pcolor_2d_data(inv_FI_search, x=min_distance_space, y=kappa_space, log_scale=True)

            plt.figure()
            plt.semilogy(min_distance_space, inv_FI_search- inv_FI_1_search)

            plt.figure()
            plt.semilogy(min_distance_space, inv_FI_search)
            plt.semilogy(min_distance_space, inv_FI_1_search)

            plt.figure()
            plt.plot(min_distance_space, inv_FI_search)

            plt.rcParams['font.size'] = 18

            # plt.figure()
            # plt.semilogy(min_distance_space, (inv_FI_search- inv_FI_1_search)[:, 1:])
            # plt.xlabel('Minimum distance')
            # plt.ylabel('$\hat{I_F}^{-1} - {I_F^{(1)}}^{-1}$')


        if False:
            ## Compute p(r | theta_1) = \int p(r | theta_1, theta_2) p(theta_2 | theta_1)
            # by sampling loads of p(r | theta_1, theta_2) and integrating out theta_2 manually
            # p(theta_2 | theta_1) is uniform if abs(theta_2) > abs(theta_1) + delta

            # Sample multiple {r | theta_1, theta_2}
            num_samples = 1000
            num_points = 20
            all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
            theta1_space = all_angles
            theta2_space = all_angles

            dataset1 = np.zeros((num_points, N, num_samples))
            dataset2 = np.zeros((num_points, num_points, N, num_samples))

            for i in xrange(theta1_space.size):

                dataset1[i] = population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)[:, np.newaxis]

                if put_noise_dataset:
                    dataset1[i] += sigma*np.random.randn(N, num_samples)

                for j in xrange(theta2_space.size):
                    dataset2[i, j] = (population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude) + population_code_response(theta2_space[j], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))[:, np.newaxis]

                    if put_noise_dataset:
                        dataset2[i, j] += sigma*np.random.randn(N, num_samples)

            # Compute p(r | theta_1, theta_2)
            nb_bins_prob_est = 25
            bins_prob_est = np.linspace(np.min(dataset2), np.max(dataset2), nb_bins_prob_est+1)
            prob_r_theta1_theta2 = np.zeros((num_points, num_points, N, nb_bins_prob_est))
            for i, theta_1 in enumerate(theta1_space):
                for j, theta_2 in enumerate(theta2_space):
                    for n in xrange(N):
                        # Get histogram estimate of p(r_i | theta_1, theta_2)
                        prob_r_theta1_theta2[i, j, n] = np.histogram(dataset2[i, j, n], bins=bins_prob_est, density=True)[0]

            # Average out theta_2, forget about min_distance
            prob_r_theta1 = np.mean(prob_r_theta1_theta2, axis=1)

            # Check at effect when theta_1 and theta_2 are too close
            min_distance_space = np.linspace(0.0, 2.0, 5.)
            # min_distance_space = np.array([0.5])
            std_mindist = np.zeros(min_distance_space.size)

            thetas_space = np.array(cross(theta1_space, theta2_space))

            search_progress = progress.Progress(min_distance_space.size)

            for m, min_distance in enumerate(min_distance_space):
                if search_progress.percentage() % 5.0 < 0.0001:
                        print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                # Restore probabilities
                prob_r_theta1_theta2_delta = prob_r_theta1_theta2.copy()

                # Check if theta_1 and theta_2 are too close
                thetas_too_close = ~enforce_distance(thetas_space[:, 0], thetas_space[:, 1], min_distance=min_distance)
                thetas_space_valid = np.ma.masked_where(np.c_[thetas_too_close, thetas_too_close], thetas_space)

                # Mask it when too close
                prob_r_theta1_theta2_delta[np.reshape(thetas_too_close, (num_points, num_points))] = np.nan
                prob_r_theta1_theta2_delta = np.ma.masked_invalid(prob_r_theta1_theta2_delta)

                # Average out theta_2, with minimum space between theta_1 and theta_2 enforced
                prob_r_theta1_delta = np.mean(prob_r_theta1_theta2_delta, axis=1)

                # Check out the effect on the standard deviations (assuming the obtained densities are gaussians)
                std_theta1_n_delta = np.zeros((num_points, N))
                mean_theta1_n_delta = np.zeros((num_points, N))
                # std_theta1_n = np.zeros((num_points, N))
                for i, theta_1 in enumerate(theta1_space):
                    for n in xrange(N):
                        stats = fit_gaussian((bins_prob_est+np.diff(bins_prob_est)[0]/2.)[:-1], prob_r_theta1_delta[i, n], should_plot=False, return_fitted_data=False)
                        mean_theta1_n_delta[i, n] = stats[0]
                        std_theta1_n_delta[i, n] = stats[1]
                        # std_theta1_n[i, n] = fit_gaussian(bins_prob_est[:-1], prob_r_theta1[i, n], should_plot=False, return_fitted_data=False)[1]


                std_mindist[m] = np.mean(std_theta1_n_delta)

                search_progress.increment()


            plt.figure()
            plt.plot(min_distance_space, std_mindist**2.)

            plt.figure()
            plt.plot(mean_theta1_n_delta[10])
            plt.plot(population_code_response(theta1_space[10], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))

        if True:
            ## Redo sampling, by putting distance constraint into prior
            # Compute p(r | theta_1) = \int p(r | theta_1, theta_2) p(theta_2 | theta_1)

            # min_distance_space = np.array([0.0001])
            min_distance_space = np.array([1.2])
            # min_distance_space = np.linspace(0.0, 1.5, 10)

            # Number of samples
            num_samples = 10000
            num_samples_test = 500

            num_points = 101

            mean_fisher_info_curve_1obj_mindist = np.zeros(min_distance_space.size)
            mean_fisher_info_curve_1obj_old_mindist = np.zeros(min_distance_space.size)
            mean_fisher_info_curve_2obj_mindist = np.zeros(min_distance_space.size)
            mean_fisher_info_curve_2obj_old_mindist = np.zeros(min_distance_space.size)

            print "Estimating from marginal probabilities"

            for mm, min_distance in enumerate(min_distance_space):
                print "- min_dist %f" % min_distance

                all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
                theta1_space = all_angles

                dataset1 = np.zeros((theta1_space.size, N, num_samples))
                dataset1_test = np.zeros((theta1_space.size, N, num_samples_test))
                dataset2 = np.zeros((theta1_space.size, N, num_samples))
                dataset2_test = np.zeros((theta1_space.size, N, num_samples_test))
                theta2_used = np.zeros((theta1_space.size, num_samples))

                for i in progress.ProgressDisplay(np.arange(theta1_space.size), display=progress.SINGLE_LINE):
                    ## One object
                    dataset1[i] = population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)[:, np.newaxis]

                    dataset1[i] += sigma*np.random.randn(N, num_samples)

                    ## Test dataset
                    dataset1_test[i] = population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)[:, np.newaxis] + sigma*np.random.randn(N, num_samples_test)

                    ## Two objects
                    for sample_i in xrange(num_samples):
                        # Sample new theta2
                        theta2_rand = 2*np.random.rand()*np.pi - np.pi

                        while ~enforce_distance(theta1_space[i], theta2_rand, min_distance=min_distance):
                            # enforce minimal distance
                            theta2_rand = 2*np.random.rand()*np.pi - np.pi

                        theta2_used[i, sample_i] = theta2_rand
                        dataset2[i, :, sample_i] = (population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude) + population_code_response(theta2_rand, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))


                    dataset2_test[i] = dataset2[i, :, :num_samples_test]

                    dataset2[i] += sigma*np.random.randn(N, num_samples)
                    dataset2_test[i] += sigma*np.random.randn(N, num_samples_test)

                theta2_test_used = theta2_used[:, :num_samples_test]

                # Compute p(r | theta_1), averaging over sampled theta_2 (already enforcing min_distance)
                nb_bins_prob_est = 53
                bins_prob_est = np.linspace(1.05*np.min(dataset1), 1.05*np.max(dataset1), nb_bins_prob_est+1)
                binsmid_prob_est = (bins_prob_est+np.diff(bins_prob_est)[0]/2.)[:-1]

                prob_r_theta1_2obj = np.zeros((theta1_space.size, N, nb_bins_prob_est))
                mean_theta1_n = np.zeros((theta1_space.size, N))
                std_theta1_n = np.zeros((theta1_space.size, N))

                prob_r_theta1_1obj = np.zeros((theta1_space.size, N, nb_bins_prob_est))
                mean_theta1_n_1obj = np.zeros((theta1_space.size, N))
                std_theta1_n_1obj = np.zeros((theta1_space.size, N))

                for i in progress.ProgressDisplay(np.arange(theta1_space.size), display=progress.SINGLE_LINE):
                    for n in xrange(N):
                        # Get histogram estimate of p(r_i | theta_1, theta_2)
                        prob_r_theta1_2obj[i, n] = np.histogram(dataset2[i, n], bins=bins_prob_est, density=True)[0]

                        # # Compute mean and std
                        # stats = fit_gaussian(binsmid_prob_est, prob_r_theta1_2obj[i, n], should_plot=False, return_fitted_data=False)
                        # mean_theta1_n[i, n] = stats[0]
                        # std_theta1_n[i, n] = stats[1]

                        # Do same for 1obj
                        prob_r_theta1_1obj[i, n] = np.histogram(dataset1[i, n], bins=bins_prob_est, density=True)[0]
                        # stats = fit_gaussian(binsmid_prob_est, prob_r_theta1_1obj[i, n], should_plot=False, return_fitted_data=False)
                        # mean_theta1_n_1obj[i, n] = stats[0]
                        # std_theta1_n_1obj[i, n] = stats[1]

                # Compute data likelihood
                loglikelihood_theta1_samples_1obj = np.zeros((theta1_space.size, num_samples_test, theta1_space.size))
                loglikelihood_theta1_samples_2obj = np.zeros((theta1_space.size, num_samples_test, theta1_space.size))

                for i in progress.ProgressDisplay(np.arange(theta1_space.size), display=progress.SINGLE_LINE):
                    index_probs = np.argmin((bins_prob_est[:-1, np.newaxis, np.newaxis] - dataset1_test[i, :, :])**2, axis=0)
                    for s in xrange(num_samples_test):
                        lik = np.log(prob_r_theta1_1obj[:, np.arange(N), index_probs[:, s]])
                        lik[np.isinf(lik)] = 0.
                        # lik = np.ma.masked_invalid(lik)

                        # Now combine likelihood of all neurons
                        loglikelihood_theta1_samples_1obj[i, s] = np.sum(lik, axis=-1)

                    ## 2 objects
                    index_probs_2 = np.argmin((bins_prob_est[:-1, np.newaxis, np.newaxis] - dataset2_test[i, :, :])**2, axis=0)
                    for s in xrange(num_samples_test):
                        lik = np.log(prob_r_theta1_2obj[:, np.arange(N), index_probs_2[:, s]])
                        lik[np.isinf(lik)] = 0.
                        # lik = np.ma.masked_invalid(lik)

                        # Now combine likelihood of all neurons
                        loglikelihood_theta1_samples_2obj[i, s] = np.sum(lik, axis=-1)

                # Now the fisher information, taken from the curvature of the likelihood
                dx = np.diff(theta1_space)[0]
                fisher_info_curve_1obj = np.zeros((theta1_space.size, num_samples_test))
                fisher_info_curve_1obj_old = np.zeros(theta1_space.size)
                fisher_info_curve_2obj = np.zeros((theta1_space.size, num_samples_test))
                fisher_info_curve_2obj_old = np.zeros(theta1_space.size)
                ml_indices_1obj = np.zeros((theta1_space.size, num_samples_test))
                ml_indices_1obj_old = np.zeros(theta1_space.size)
                ml_indices_2obj = np.zeros((theta1_space.size, num_samples_test))
                ml_indices_2obj_old = np.zeros(theta1_space.size)

                for i in xrange(theta1_space.size):
                    ml_indices_1obj[i] = np.argmax(loglikelihood_theta1_samples_1obj[i], axis=1)
                    curv_logp2 = -np.gradient(np.gradient(loglikelihood_theta1_samples_1obj[i])[1])[1]/dx**2.
                    fisher_info_curve_1obj[i] = curv_logp2[np.arange(num_samples_test), ml_indices_1obj[i].astype(int)]

                    # 1obj save
                    likelihood_chosen_theta1_samples_1obj = np.mean(loglikelihood_theta1_samples_1obj[i], axis=0)
                    ml_indices_1obj_old[i] = np.argmax(likelihood_chosen_theta1_samples_1obj)
                    curv_logp = -np.gradient(np.gradient(likelihood_chosen_theta1_samples_1obj))/dx**2.
                    fisher_info_curve_1obj_old[i] = curv_logp[ml_indices_1obj_old[i]]

                    # Same for 2 objects
                    ml_indices_2obj[i] = np.argmax(loglikelihood_theta1_samples_2obj[i], axis=1)
                    curv_logp2 = -np.gradient(np.gradient(loglikelihood_theta1_samples_2obj[i])[1])[1]/dx**2.
                    fisher_info_curve_2obj[i] = curv_logp2[np.arange(num_samples_test), ml_indices_2obj[i].astype(int)]

                    likelihood_chosen_theta1_samples_2obj = np.mean(loglikelihood_theta1_samples_2obj[i], axis=0)
                    ml_indices_2obj_old[i] = np.argmax(likelihood_chosen_theta1_samples_2obj)
                    curv_logp = -np.gradient(np.gradient(likelihood_chosen_theta1_samples_2obj))/dx**2.
                    fisher_info_curve_2obj_old[i] = curv_logp[ml_indices_2obj_old[i]]


                mean_fisher_info_curve_1obj_old_mindist[mm] = np.mean(fisher_info_curve_1obj_old[3:-3])
                mean_fisher_info_curve_2obj_old_mindist[mm] = np.mean(fisher_info_curve_2obj_old[3:-3])


            theta1_to_plot = int(theta1_space.size/2)
            theta2_test_used_sorted = np.argsort(theta2_test_used[theta1_to_plot])
            diff_thetas = (theta2_test_used[theta1_to_plot, theta2_test_used_sorted] - theta1_space[theta1_to_plot])

            ## First show how the loglikelihoods for all samples vary as a function of the position of theta2
            loglikelihood_theta1_samples_2obj_sortedfiltered = loglikelihood_theta1_samples_2obj[theta1_to_plot, theta2_test_used_sorted]
            pcolor_2d_data(loglikelihood_theta1_samples_2obj_sortedfiltered - np.mean(loglikelihood_theta1_samples_2obj_sortedfiltered, axis=1)[:, np.newaxis], y=theta1_space, x=diff_thetas, xlabel='$\\theta_2-\\theta_1$', ylabel='$p(\\theta_1 | r)$', ticks_interpolate=10)
            plt.plot(np.argmax(loglikelihood_theta1_samples_2obj_sortedfiltered, axis=1), 'bo', markersize=5)

            # pcolor_2d_data(curv_logp[theta2_test_used_sorted])
            # plot(ml_indices_2obj[theta1_to_plot, theta2_test_used_sorted], 'ro', markersize=5)
            plt.figure()
            plt.plot(diff_thetas, fisher_info_curve_2obj[theta1_to_plot, theta2_test_used_sorted])






    if False:
        #####
        #   Marginal fisher information, 2 items, 2 features
        #####

        N     = 10
        kappa = 3.0
        sigma = 0.3
        amplitude = 1.0
        min_distance = 2.
        num_points = 100

        def population_code_response_2D(theta1, theta2, pref_angles=None, N=10, kappa=0.1, amplitude=1.0):
            if pref_angles is None:
                pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

            return amplitude*np.exp(kappa*np.cos(theta1 - pref_angles) + kappa*np.cos(theta2 - pref_angles))/(4.*np.pi**2.*scsp.i0(kappa)**2.)

        # Preferred stimuli
        pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

        # Space discretised
        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

        item1_theta1_space = np.array([0.])
        item1_theta2_space = np.array([0.])

        item2_theta1_space = all_angles
        item2_theta2_space = all_angles

        for i, item1_theta1 in enumerate(item1_theta1_space):
            for j, item1_theta2 in enumerate(item1_theta2_space):

                der_1 = kappa*np.sin(pref_angles - item1_theta1)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                for k, item2_theta1 in enumerate(item2_theta1_space):
                    for l, item2_theta2 in enumerate(item2_theta2_space):

                        if enforce_distance(item1_theta1, item2_theta1, min_distance=min_distance) and enforce_distance(item1_theta2, item2_theta2, min_distance=min_distance):
                            # Only compute if items are sufficiently different

                            der_2 = kappa**2.*np.sin(pref_angles - theta2)*population_code_response(theta2, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                            # FI for 2 objects
                            # FIX MATHS AND DIMENSIONALITY
                            # FI_all[i, j, k, l] = np.sum(der_1**2.)/sigma**2.

                            # Inv FI for 2 objects
                            # TODO FIX MATHS AND DIMENSIONALITY
                            # inv_FI_all[i, j, k, l] = sigma**2./(np.sum(der_1**2.) - np.sum(der_1*der_2)**2./np.sum(der_2**2.))

                # FI for 1 object
                # CHECK DIMENSIONALITY
                # inv_FI_1[i, j] = sigma**2./np.sum(der_1**2.)





    plt.show()
    import sh
    sh.say('Work complete')


