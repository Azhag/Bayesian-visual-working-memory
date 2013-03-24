    #!/usr/bin/env python
# encoding: utf-8
"""
computations_marginalfisherinfo_marginalposterior_1d.py

Created by Loic Matthey on 2012-06-03.
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

if __name__ == '__main__':

    #### 
    #   1D two stimuli
    ####

    N     = 100
    kappa = 5.0
    sigma = 0.3
    amplitude = 1.0
    min_distance = 0.001

    dataio = DataIO(label='compute_fimarg', calling_function='')
    additional_comment = ''

    def population_code_response(theta, pref_angles=None, N=100, kappa=0.1, amplitude=1.0):
        if pref_angles is None:
            pref_angles = np.linspace(0., 2*np.pi, N, endpoint=False)

        return amplitude*np.exp(kappa*np.cos(theta - pref_angles))/(2.*np.pi*scsp.i0(kappa))

    pref_angles = np.linspace(-np.pi, np.pi, N, endpoint=False)

    ## Estimate likelihood
    num_points = 500
    # num_points_space = np.arange(50, 1000, 200)
    # effects_num_points = []

    # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
    all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)

    theta1_space = np.array([0.])
    theta2_space = all_angles

    def enforce_distance(theta1, theta2, min_distance=0.1):
        return np.abs(wrap_angles(theta1 - theta2)) > min_distance

    #### Compute Theo Inverse Fisher Info

    if True:
        ### Loop over min_distance and kappa
        # min_distance_space = np.linspace(0.0, 1.5, 10)
        min_distance_space = np.array([min_distance])
        # min_distance_space = np.array([0.001])
        # kappa_space = np.linspace(0.05, 30., 40.)
        # kappa_space = np.array([kappa])

        sigma_space = np.array([sigma])
        # sigma_space = np.array([0.1, 0.25, 0.5])
        # sigma_space = np.linspace(0.1, 1.0, 11)

        dim1_size = min_distance_space.size
        dim2_size = sigma_space.size

        inv_FI_search = np.zeros((dim1_size, dim2_size))
        FI_search = np.zeros((dim1_size, dim2_size))
        inv_FI_1_search = np.zeros((dim1_size, dim2_size))

        search_progress = progress.Progress(dim1_size*dim2_size)

        print "Doing from marginal FI"

        for m, min_distance in enumerate(min_distance_space):
            # for k, kappa in enumerate(kappa_space):
            for k, sigma in enumerate(sigma_space):

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
                FI_search[m, k] = np.mean(np.ma.masked_equal(FI_all, 0))

                inv_FI_1_search[m, k] = np.mean(inv_FI_1)

                search_progress.increment()


        print 1./inv_FI_search
        print FI_search
        print 1./inv_FI_1_search

        # Some plots
        # pcolor_2d_data(inv_FI_search, x=min_distance_space, y=kappa_space, log_scale=True)

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

    if False:
        ## Redo sampling, by putting distance constraint into prior
        # Compute p(r | theta_1) = \int p(r | theta_1, theta_2) p(theta_2 | theta_1)
        
        min_distance_space = np.array([0.0001])
        # min_distance_space = np.array([1.0])
        # min_distance_space = np.linspace(0.0, 1.5, 3)
        
        # Number of samples
        num_samples = 30000
        num_samples_test = 1000

        num_points = 85
        all_angles = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        theta1_space = all_angles

        theta1_to_plot = int(theta1_space.size/2)

        nb_bins_prob_est = 61
        bins_prob_est = None
        binsmid_prob_est = None

        dataset2_samples = np.empty((num_samples, N))
        dataset1_test = np.empty((theta1_space.size, N, num_samples_test))
        dataset2_test = np.empty((theta1_space.size, N, num_samples_test))
        theta2_used = np.empty((theta1_space.size, num_samples))

        prob_r_theta1_2obj = np.empty((theta1_space.size, N, nb_bins_prob_est))
        prob_r_theta1_1obj = np.empty((theta1_space.size, N, nb_bins_prob_est))

        posteriormeanvar_theta1_samples_1obj = np.empty((theta1_space.size, num_samples_test, 2))
        posteriormeanvar_theta1_samples_2obj = np.empty((theta1_space.size, num_samples_test, 2))

        mean_fisher_info_curve_1obj_mindist = np.zeros(min_distance_space.size)
        mean_fisher_info_curve_1obj_old_mindist = np.zeros(min_distance_space.size)
        mean_fisher_info_curve_2obj_mindist = np.zeros(min_distance_space.size)
        mean_fisher_info_curve_2obj_old_mindist = np.zeros(min_distance_space.size)
        fisher_info_curve_1obj_mindist = np.empty((min_distance_space.size, theta1_space.size, num_samples_test))
        fisher_info_curve_2obj_mindist = np.empty((min_distance_space.size, theta1_space.size, num_samples_test))
        theta2_test_used_sorted_mindist = np.empty((min_distance_space.size, theta1_space.size, num_samples_test))
        diff_thetas_mindist = np.empty((min_distance_space.size, num_samples_test))
        posteriormeanvar_theta1_samples_1obj_mindist = np.empty((min_distance_space.size, theta1_space.size, num_samples_test, 2))
        posteriormeanvar_theta1_samples_2obj_mindist = np.empty((min_distance_space.size, theta1_space.size, num_samples_test, 2))
        posteriormeanvar_singlebump_2obj = np.empty((min_distance_space.size, 2, num_samples_test))
        posteriormeanvar_singlebump_gauss_2obj = np.empty((min_distance_space.size, 2, num_samples_test))
            
        print "Estimating from marginal probabilities"
        print "kappa: %.2f" % kappa
        print "sigma: %.2f" % sigma
        print "samples: %d, %d" % (num_samples, num_samples_test)

        additional_comment = 'kappa %.2f, sigma %.2f, samples: %d, %d, Min_dist: %s' % (kappa, sigma, num_samples, num_samples_test, min_distance_space)

        for mm, min_distance in enumerate(min_distance_space):
            print "- min_dist %f" % min_distance

            for i in progress.ProgressDisplay(np.arange(theta1_space.size), display=progress.SINGLE_LINE):
                ## One object
                dataset1_samples = population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)[:, np.newaxis] + sigma*np.random.randn(N, num_samples)

                ## Test dataset
                dataset1_test[i] = population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude)[:, np.newaxis] + sigma*np.random.randn(N, num_samples_test)
                
                ## Two objects
                for sample_i in np.arange(num_samples):
                    # Sample new theta2
                    theta2_rand = 2*np.random.rand()*np.pi - np.pi
                
                    while ~enforce_distance(theta1_space[i], theta2_rand, min_distance=min_distance):
                        # enforce minimal distance
                        theta2_rand = 2*np.random.rand()*np.pi - np.pi
                    
                    theta2_used[i, sample_i] = theta2_rand
                    dataset2_samples[sample_i] = (population_code_response(theta1_space[i], pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude) + population_code_response(theta2_rand, pref_angles=pref_angles, N=N, kappa=kappa, amplitude=amplitude))

                dataset2_test[i] = dataset2_samples[:num_samples_test].T
                
                dataset2_samples += sigma*np.random.randn(num_samples, N)
                dataset2_test[i] += sigma*np.random.randn(N, num_samples_test)

                if bins_prob_est is None:
                    # First initialisation of the bins for the probability estimation
                    bins_prob_est = np.linspace(1.05*np.min(dataset2_samples), 1.05*np.max(dataset2_samples), nb_bins_prob_est+1)
                    binsmid_prob_est = (bins_prob_est+np.diff(bins_prob_est)[0]/2.)[:-1]

                # Get histogram estimate of p(r_i | theta_1, theta_2)
                for n in np.arange(N):
                    prob_r_theta1_2obj[i, n] = np.histogram(dataset2_samples[:, n], bins=bins_prob_est, density=True)[0]
                    prob_r_theta1_1obj[i, n] = np.histogram(dataset1_samples[n], bins=bins_prob_est, density=True)[0]
            

            # Check which theta2 we actually sampled
            theta2_test_used = theta2_used[:, :num_samples_test]
            theta2_test_used_sorted = np.argsort(theta2_test_used, axis=1)
            diff_thetas = (theta2_test_used[theta1_to_plot, theta2_test_used_sorted[theta1_to_plot]] - theta1_space[theta1_to_plot])
            theta2_test_used_sorted_mindist[mm] = theta2_test_used_sorted
            diff_thetas_mindist[mm] = diff_thetas

            # Compute data likelihood
            loglikelihood_theta1_samples_1obj = np.zeros((theta1_space.size, num_samples_test, theta1_space.size))
            loglikelihood_theta1_samples_2obj = np.zeros((theta1_space.size, num_samples_test, theta1_space.size))

            for i in progress.ProgressDisplay(np.arange(theta1_space.size), display=progress.SINGLE_LINE):
                index_probs = np.argmin((binsmid_prob_est[:, np.newaxis, np.newaxis] - dataset1_test[i, :, :])**2, axis=0)
                for s in np.arange(num_samples_test):
                    lik = np.log(prob_r_theta1_1obj[:, np.arange(N), index_probs[:, s]])
                    lik[np.isinf(lik)] = 0.0
                    # lik = np.ma.masked_invalid(lik)

                    # Now combine likelihood of all neurons
                    loglikelihood_theta1_samples_1obj[i, s] = np.sum(lik, axis=-1)

                    # Estimate the true mean and variance of the current posteriors
                    posterior_curr = np.exp(loglikelihood_theta1_samples_1obj[i, s])
                    posterior_curr /= np.trapz(posterior_curr, theta1_space)
                    posteriormeanvar_theta1_samples_1obj[i, s, 0] = np.trapz(posterior_curr*theta1_space, theta1_space)
                    posteriormeanvar_theta1_samples_1obj[i, s, 1] = np.trapz(posterior_curr*(theta1_space - posteriormeanvar_theta1_samples_1obj[i, s, 0])**2., theta1_space)

                ## 2 objects
                index_probs_2 = np.argmin((binsmid_prob_est[:, np.newaxis, np.newaxis] - dataset2_test[i, :, :])**2, axis=0)
                for s in np.arange(num_samples_test):
                    lik = np.log(prob_r_theta1_2obj[:, np.arange(N), index_probs_2[:, s]])
                    lik[np.isinf(lik)] = 0.0
                    # lik = np.ma.masked_invalid(lik)

                    # Now combine likelihood of all neurons
                    loglikelihood_theta1_samples_2obj[i, s] = np.sum(lik, axis=-1)

                    posterior_curr = np.exp(loglikelihood_theta1_samples_2obj[i, s])
                    posterior_curr /= np.trapz(posterior_curr, theta1_space)
                    posteriormeanvar_theta1_samples_2obj[i, s, 0] = np.trapz(posterior_curr*theta1_space, theta1_space)
                    posteriormeanvar_theta1_samples_2obj[i, s, 1] = np.trapz(posterior_curr*(theta1_space - posteriormeanvar_theta1_samples_2obj[i, s, 0])**2., theta1_space)


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

            for i in np.arange(theta1_space.size):
            # for i in np.array([theta1_to_plot]):
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
                # ml_indices_2obj[i] = i
                curv_logp2 = -np.gradient(np.gradient(loglikelihood_theta1_samples_2obj[i])[1])[1]/dx**2.
                fisher_info_curve_2obj[i] = curv_logp2[np.arange(num_samples_test), ml_indices_2obj[i].astype(int)]

                likelihood_chosen_theta1_samples_2obj = np.mean(loglikelihood_theta1_samples_2obj[i], axis=0)
                ml_indices_2obj_old[i] = np.argmax(likelihood_chosen_theta1_samples_2obj)
                curv_logp = -np.gradient(np.gradient(likelihood_chosen_theta1_samples_2obj))/dx**2.
                fisher_info_curve_2obj_old[i] = curv_logp[ml_indices_2obj_old[i]]


            ## Find the variance of only one of the bump
            # Cut out everything away from the center bump, for the theta1_to_plot index (which should be the middle point)
            # then get the variance of that instead of the full posterior.
            singlebump_post = np.exp(loglikelihood_theta1_samples_2obj[theta1_to_plot, theta2_test_used_sorted[theta1_to_plot]])
            singlebump_post[:, :25] = 0.0
            singlebump_post[:, 55:] = 0.0
            singlebump_post /= np.trapz(singlebump_post, theta1_space)[:, np.newaxis]
            posteriormeanvar_singlebump_2obj[mm, 0] = np.trapz(singlebump_post*theta1_space, theta1_space)
            posteriormeanvar_singlebump_2obj[mm, 1] = np.trapz(singlebump_post*(theta1_space - posteriormeanvar_singlebump_2obj[mm, 0][:, np.newaxis])**2., theta1_space)

            for s in np.arange(num_samples_test):
                # Compare with gaussian fit
                stats = fit_gaussian(theta1_space, singlebump_post[s], should_plot=False, return_fitted_data=False)
                posteriormeanvar_singlebump_gauss_2obj[mm, 0, s] = stats[0]
                posteriormeanvar_singlebump_gauss_2obj[mm, 1, s] = stats[1]

            mean_fisher_info_curve_1obj_old_mindist[mm] = np.mean(fisher_info_curve_1obj_old[10:-10])
            mean_fisher_info_curve_2obj_old_mindist[mm] = np.mean(fisher_info_curve_2obj_old[10:-10])
            fisher_info_curve_1obj_mindist[mm] = fisher_info_curve_1obj
            fisher_info_curve_2obj_mindist[mm] = fisher_info_curve_2obj
            posteriormeanvar_theta1_samples_1obj_mindist[mm] = posteriormeanvar_theta1_samples_1obj
            posteriormeanvar_theta1_samples_2obj_mindist[mm] = posteriormeanvar_theta1_samples_2obj


            print "1obj old", mean_fisher_info_curve_1obj_old_mindist
            print "1obj", np.mean(np.mean(fisher_info_curve_1obj_mindist[:, 10:-10], axis=-1), axis=-1)
            print "2obj, old", mean_fisher_info_curve_2obj_old_mindist
            print "2obj", np.mean(np.mean(fisher_info_curve_2obj_mindist[:, 10:-10], axis=-1), axis=-1)
            print "var post 1obj", np.mean(np.mean(posteriormeanvar_theta1_samples_1obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)
            print "1/E[var 1obj] post (no borders)", 1./np.mean(np.mean(posteriormeanvar_theta1_samples_1obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)
            print "E[1/var 1obj] post (no borders)", np.mean(np.mean(1./posteriormeanvar_theta1_samples_1obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)
            print "var post 2obj", np.mean(np.mean(posteriormeanvar_theta1_samples_2obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)
            print "1/E[var 2obj] post (no borders)", 1./np.mean(np.mean(posteriormeanvar_theta1_samples_2obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)
            print "E[1/var 2obj] post (no borders)", np.mean(np.mean(1./posteriormeanvar_theta1_samples_2obj_mindist[:, 10:-10, :, 1], axis=-1), axis=-1)


        ## First show how the loglikelihoods for all samples vary as a function of the position of theta2
        loglikelihood_theta1_samples_2obj_sortedfiltered = loglikelihood_theta1_samples_2obj[theta1_to_plot, theta2_test_used_sorted[theta1_to_plot]]
        pcolor_2d_data(loglikelihood_theta1_samples_2obj_sortedfiltered - np.mean(loglikelihood_theta1_samples_2obj_sortedfiltered, axis=1)[:, np.newaxis], y=theta1_space, x=diff_thetas, xlabel='$\\theta_2-\\theta_1$', ylabel='$p(\\theta_1 | r)$', ticks_interpolate=10)
        plt.plot(np.argmax(loglikelihood_theta1_samples_2obj_sortedfiltered, axis=1), 'bo', markersize=5)

        dataio.save_current_figure('loglik2d_kappa%.1fsigma%.1fmindist%.1f{label}_{unique_id}.pdf' % (kappa, sigma, min_distance))


        # pcolor_2d_data(curv_logp[theta2_test_used_sorted])
        # plot(ml_indices_2obj[theta1_to_plot, theta2_test_used_sorted], 'ro', markersize=5)
        plt.figure()
        plt.plot(diff_thetas, fisher_info_curve_2obj[theta1_to_plot, theta2_test_used_sorted[theta1_to_plot]])

        dataio.save_current_figure('fi2objsorted_kappa%.1fsigma%.1fmindist%.1f{label}_{unique_id}.pdf' % (kappa, sigma, min_distance))

        plt.figure()
        plt.plot(theta1_space, loglikelihood_theta1_samples_2obj_sortedfiltered[(num_samples_test/2-20):(num_samples_test/2+20):2].T)
        dataio.save_current_figure('posterior_2objsorted_kappa%.1fsigma%.1fmindist%.1f{label}_{unique_id}.pdf' % (kappa, sigma, min_distance))        


    plt.show()

    say_finished(additional_comment=additional_comment)

