#!/usr/bin/env python
# encoding: utf-8
"""
fisherinformation.py

Created by Loic Matthey on 2012-06-03.
Copyright (c) 2012 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np

from utils import *
from statisticsmeasurer import *
from randomfactorialnetwork import *
from datagenerator import *

def main():
    pass

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

    if True:
        # Do the optimisation
        target_experimental_precisions  = np.array([5.0391, 3.4834, 2.9056, 2.2412, 1.7729])/2.

        N_sqrt = 20.

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
        stim_space = np.linspace(0, 2.*np.pi, precision, endpoint=False)
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
                FI_T[T_i] = kappa1**2.*np.mean(np.sum(np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))*np.linalg.solve(covariances_all[T_i], np.sin(stim_space[:, 0] - mu[:, np.newaxis])*np.exp(kappa1*np.cos(stim_space[:, 0] - mu[:, np.newaxis]) + kappa2*np.cos(stim_space[:, 1] - gamma[:, np.newaxis]))), axis=0))
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
            sigma_x = 1.0

            #### See effect of kappa
            kappa_space = np.linspace(0.05, 30.0, 100.)

            kappa_MSE_experimental = np.zeros_like(kappa_space)
            FI_all_kappa = np.zeros((kappa_space.size, T_all.size))
            kappa_powerlaw_params = np.zeros((kappa_space.size, 2))

            covariances_all = np.zeros((T_all.size, int(N_sqrt**2.), int(N_sqrt**2.)))

            for i, kappa in enumerate(kappa_space):
                print "%.f%%, Doing kappa: %.2f" % (i*100./kappa_space.size, kappa)
                
                for T_i, T in enumerate(T_all):
                    time_weights_parameters = dict(weighting_alpha=1.0, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
                    rn = RandomFactorialNetwork.create_full_conjunctive(int(N_sqrt**2.), R=2, scale_moments=(kappa, 0.0001), ratio_moments=(1.0, 0.0001))
                    data_gen_noise = DataGeneratorRFN(3000, T, rn, sigma_y=sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=T-1)
                    stat_meas = StatisticsMeasurer(data_gen_noise)
                    covariances_all[T_i] = stat_meas.model_parameters['covariances'][2][-1]

                FI_all_kappa[i] = compute_FI(sigma_x, kappa, kappa, T_all, stim_space, mu, gamma, covariances_all)
                
                # sigma_x_MSE_experimental[i] = MSE_from_experimental(FI_all_kappa[i], target_experimental_precisions)

                kappa_MSE_experimental[i] = MSE_powerlaw_experimental(FI_all_kappa[i], target_experimental_precisions)
                kappa_powerlaw_params[i] = fit_powerlaw(np.arange(1, 6), FI_all_kappa[i])

            plt.figure()
            plt.semilogy(kappa_space, kappa_MSE_experimental)

            # Show the best sigma
            best_kappa_ind = np.argmin(kappa_MSE_experimental)

            print "Best kappa: %.2f. MSE: %.2f" % (kappa_space[best_kappa_ind], kappa_MSE_experimental[best_kappa_ind])

            # Precision curve
            plt.figure()
            plt.plot(T_all, FI_all_kappa[best_kappa_ind])
            plt.plot(T_all, target_experimental_precisions)
            plt.legend(['Model', 'Experimental'])

            # 2D effect of sigma, not good
            # plt.figure()
            # plt.imshow(FI_all_kappa.T, norm=plt.matplotlib.colors.LogNorm())

            # 3D effect of kappa, nice enough. Smoother than sigmax
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(kappa_space, T_all)
            ax.plot_surface(X, Y, np.log(FI_all_kappa.T))

            # Effect of kappa on the power law exponent: null.
            plt.figure()
            plt.plot(kappa_space, kappa_powerlaw_params[:, 0])

            # Effect of kappa on the power law y0 value: same as FI[0]
            plt.figure()
            plt.semilogy(kappa_space, kappa_powerlaw_params[:, 1])
            
            # Fisher information for single object, as function of kappa
            plt.figure()
            plt.plot(kappa_space**2., FI_all_kappa[:, 0])
            plt.xlabel('kappa')
            plt.xlim([0, 10])
            plt.ylim([0, 10000])
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








    plt.show()


