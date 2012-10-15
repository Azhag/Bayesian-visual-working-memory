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
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *

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
            kappa_space = np.linspace(0.01, 30.0, 100.)
            # kappa_space = np.array([1.0])

            T_all = np.arange(1, 2)

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
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters = stat_meas.model_parameters, tc=cued_feature_time)
                
            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, param %.3f" % param
            (FI_M_effect_mean[i], FI_M_effect_std[i], _) = sampler.estimate_fisher_info_from_posterior_avg(num_points=100, return_std=True)

        # Dependence: FI = M * (sigma_y + sigma_x)**-1 * rcscale**2
        
        print (FI_M_effect_mean, FI_M_effect_std)


        plot_mean_std_area(param_space, FI_M_effect_mean, FI_M_effect_std)





################################################################################
########################################################################################################################################################
################################################################################


    
    if False:
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
            N     = 200
            kappa = 1.0
            sigma = 0.2
            amplitude = 1.0
            
            def population_code_response(theta, N=100, kappa=0.1, amplitude=1.0):

                pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

                return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))

            kappa_space = np.linspace(0.001, 20., 10)
            # kappa_space = np.linspace(2., 2., 1)

            effects_kappa = []
            effects_kappa_std = []

            slicesampler = SliceSampler()

            for k, kappa in enumerate(kappa_space):
                print 'DOING KAPPA: %.3f' % kappa

                ## Generate dataset
                M = 200
                # stimuli_used = np.random.rand(M)*np.pi*2.
                # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
                stimuli_used = np.ones(M)*1.2

                dataset = np.zeros((M, N))
                for i, stim in enumerate(stimuli_used):
                    dataset[i] = population_code_response(stim, N=N, kappa=kappa, amplitude=amplitude) + sigma*np.random.randn(N)

                ## Estimate likelihood
                num_points = 1000
                # num_points_space = np.arange(50, 1000, 200)
                # effects_num_points = []

                # for k, num_points in enumerate(num_points_space):

                all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)

                def likelihood(data, all_angles, N=100, kappa=0.1, sigma=1.0, should_exponentiate=True, remove_mean=False):

                    lik = np.zeros(all_angles.size)
                    for i, angle in enumerate(all_angles):
                        # lik[i] = -np.log((2*np.pi)**0.5*sigma) -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)
                        lik[i] = -1./(2*sigma**2.)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)

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
                    return -1./(2.*sigma**2.0)*np.sum((data - population_code_response(angle, N=N, kappa=kappa, amplitude=amplitude))**2.)

                ## Estimate fisher info
                print "Estimate fisher info"
                fisher_info_curve = np.zeros(M)
                fisher_info_prec = np.zeros(M)
                dx = np.diff(all_angles)[0]

                samples_all_precisions = []
                recall_samples = np.zeros(M)

                for m, data in enumerate(dataset):
                    print m
                    
                    posterior = likelihood(data, all_angles, N=N, kappa=kappa, sigma=sigma)
                    log_posterior = np.log(posterior)
                    
                    log_posterior[np.isinf(log_posterior)] = 0.0
                    log_posterior[np.isnan(log_posterior)] = 0.0

                    posterior /= np.sum(posterior*dx)

                    # Fails when angles are close to 0/2pi.
                    # Could roll the posterior around to center it, wouldn't be that bad.
                    # fisher_info_curve[m] = np.trapz(-np.diff(np.diff(log_posterior))*posterior[1:-1]/dx**2., all_angles[1:-1])
                    
                    # Actually wrong, see Issue #23
                    # fisher_info_curve[m] = np.trapz(-np.gradient(np.gradient(log_posterior))*posterior/dx**2., all_angles)

                    # Take curvature at ML value
                    ml_index = np.argmax(posterior)
                    curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.
                    fisher_info_curve[m] = curv_logp[ml_index]
                    
                    #fisher_info_prec[m] = 1./fit_gaussian(all_angles, posterior, should_plot=False, return_fitted=False)[1]**2.
                    fisher_info_prec[m] = 1./(-2.*np.log(np.abs(np.trapz(posterior*np.exp(1j*all_angles), all_angles))))

                    # Using samples, estimate the precision for each data
                    params = dict(sigma=sigma, data=data, N=N, kappa=kappa)
                    samples, _ = slicesampler.sample_1D_circular(300, np.random.rand()*2.*np.pi-np.pi, lik_sampler, burn=100, widths=np.pi/4., loglike_fct_params=params, debug=False, step_out=True)

                    samples_circ_std_dev = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*samples), axis=0))))
                    samples_all_precisions.append(1./samples_circ_std_dev**2.)

                    # Now sample one angle, and compute the fisher information from the distribution of the recall samples
                    # choose last one
                    # recall_samples[m] = np.median(samples[-100:])
                    recall_samples[m] = samples[-1]


                fisher_info_curve_mean = np.mean(fisher_info_curve)
                fisher_info_curve_std = np.std(fisher_info_curve)
                fisher_info_prec_mean = np.mean(fisher_info_prec)
                fisher_info_prec_std = np.std(fisher_info_prec)

                samples_precisions_mean = np.mean(samples_all_precisions)
                samples_precisions_std = np.std(samples_all_precisions)

                recall_samples_precision = 1./np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2

                # Save it
                effects_kappa.append([fisher_info_curve_mean, fisher_info_prec_mean, fisher_info_N(N=N, kappa=kappa, sigma=sigma), fisher_info_Ninf(kappa=kappa, N=N, sigma=sigma), samples_precisions_mean, recall_samples_precision])
                effects_kappa_std.append([fisher_info_curve_std, fisher_info_prec_std, 0, 0, samples_precisions_std, 0])

                # effects_num_points.append((fisher_info_curve_mean, fisher_info_curve_std))

                print "FI curve: %.3f, FI precision: %.3f, Samples: %.3f, Recall precision: %.3f, Theo: %.3f, Theo large N: %.3f" % (fisher_info_curve_mean, fisher_info_prec_mean, samples_precisions_mean, recall_samples_precision, fisher_info_N(N=N, kappa=kappa, sigma=sigma), fisher_info_Ninf(kappa=kappa, N=N, sigma=sigma))

            # plot_mean_std_area(num_points_space, np.array(effects_num_points)[:, 0], np.array(effects_num_points)[:, 1])
            effects_kappa = np.array(effects_kappa)
            effects_kappa_std = np.array(effects_kappa_std)
            

            # No small N / big kappa effect on Fisher information.
            plot_multiple_mean_std_area(kappa_space, effects_kappa.T, effects_kappa_std.T)

            plt.legend(['Curvature', 'Posterior precision', 'Theo', 'Theo large N', 'Samples', 'Recall precision'])



    if False:
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

        kappa_space = np.linspace(0.001, 20., 5)
        # kappa_space = np.linspace(2., 2., 1)

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

            for m, data in enumerate(dataset):
                print m
                
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
        sigma = 0.2
        M = 500
        num_points = 1000

        # stimuli_used = np.random.rand(M)*np.pi*2.
        # stimuli_used = np.random.rand(M)*np.pi/2. + np.pi
        stimuli_used = np.ones(M)*1.2

        dataset = np.zeros((M, N))
        for i, stim in enumerate(stimuli_used):
            dataset[i] = stimuli_used[i] + sigma*np.random.randn(N)


        def loglik_sampler(angle, params):
            sigma = params['sigma']
            data = params['data']
            return -1./(2.*sigma**2.0)*np.sum((data - angle)**2.)

        def loglik_fullspace(all_angles, params):
            lik = np.zeros(all_angles.size)
            for i, angle in enumerate(all_angles):
                lik[i] = loglik_sampler(angle, params)

            return lik

        all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
        dx = np.diff(all_angles)[0]

        slicesampler = SliceSampler()

        fisher_info_curve = np.zeros(M)
        samples_all_precisions = np.zeros(M)
        recall_samples = np.zeros(M)

        for m, data in enumerate(dataset):
            print m

            params = dict(sigma=sigma, data=data)
            log_posterior = loglik_fullspace(all_angles, params)
            
            log_posterior[np.isinf(log_posterior)] = 0.0
            log_posterior[np.isnan(log_posterior)] = 0.0

            # Take curvature at ML value
            ml_index = np.argmax(log_posterior)
            curv_logp = -np.gradient(np.gradient(log_posterior))/dx**2.
            fisher_info_curve[m] = curv_logp[ml_index]

            # Precision from samples
            samples, _ = slicesampler.sample_1D_circular(500, np.random.rand()*2.*np.pi-np.pi, loglik_sampler, burn=100, widths=np.pi/4., loglike_fct_params=params, debug=False, step_out=True)

            samples_circ_std_dev = np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*samples), axis=0))))
            samples_all_precisions[m] = 1./samples_circ_std_dev**2.

            # Now sample one angle, and compute the fisher information from the distribution of the recall samples
            # choose last one
            recall_samples[m] = np.median(samples[-1:])
        
        recall_samples_precision = 1./np.sqrt(-2.*np.log(np.abs(np.mean(np.exp(1j*(recall_samples - stimuli_used)), axis=0))))**2

        plt.figure()
        plt.boxplot([fisher_info_curve, samples_all_precisions, recall_samples_precision])
        plt.title('Comparison Curvature vs samples estimate vs recall precision. Simple gaussian.')
        plt.xticks([1, 2, 3], ['Curvature', 'Samples', 'Precision'], rotation=45)

        print np.mean(fisher_info_curve/recall_samples_precision)

    if True:
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

            slicesampler = SliceSampler()

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


    plt.show()


