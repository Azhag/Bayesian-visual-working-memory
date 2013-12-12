#!/usr/bin/env python
# encoding: utf-8
"""
launchers_fisherinformation.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from datapbs import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
import progress

import launchers

def launcher_do_fisher_information_estimation(args):
    '''
        Estimate the Fisher information from the posterior.

        Get its dependance upon M and rcscale
    '''

    print "Fisher Information estimation from Posterior."

    N = args.N
    T = args.T
    # K = args.K
    M = args.M
    R = args.R
    weighting_alpha = args.alpha
    code_type = args.code_type
    rc_scale = args.rc_scale
    rc_scale2 = args.rc_scale2
    ratio_conj = args.ratio_conj
    sigma_x = args.sigmax
    sigma_y = args.sigmay
    selection_method = args.selection_method

    stimuli_generation = 'constant'
    # stimuli_generation = 'random'

    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    if args.subaction == '':
        args.subaction = 'M_dependence'

    if args.subaction == 'M_dependence':

        M_space = np.arange(10, 500, 20)
        FI_rc_theo = np.zeros((M_space.size, 2), dtype=float)

        for i, M in enumerate(M_space):

            # Build the random network
            time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
            cued_feature_time = T-1

            if code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(rc_scale, 0.00001), ratio_moments=(1.0, 0.00001))
            elif code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(M, R=R, scale=rc_scale, ratio=40.)
            elif code_type == 'mixed':
                conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            elif code_type == 'wavelet':
                random_network = RandomFactorialNetwork.create_wavelet(M, R=R, scales_number=5)
            else:
                raise ValueError('Code_type is wrong!')

            # Construct the real dataset
            # print "Building the database"
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)

            # Measure the noise structure
            # print "Measuring noise structure"
            data_gen_noise = DataGeneratorRFN(5000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)

            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

            ### Estimate the Fisher Information
            print "theoretical FI, M %d" % M
            FI_rc_theo[i, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=sampler.noise_covariance)
            FI_rc_theo[i, 1] = random_network.compute_fisher_information_theoretical(sigma=sigma_x+sigma_y)

        # TODO DOESN'T work now...
        # https://github.com/Azhag/Bayesian-visual-working-memory/issues/10

        # Plot results
        # plot_mean_std_area(M_space, FI_M_effect, FI_M_effect_std)
        plt.figure()
        plt.plot(M_space, FI_rc_theo[:, 0] - FI_rc_theo[:, 1])
        plt.title('FI dependence on M')
        plt.xlabel('M')
        plt.ylabel('Theo finite - Theo large M')

    elif args.subaction == 'samples_dependence':

        # samples_space = np.linspace(50, 1000, 11)
        samples_space = np.linspace(500., 500., 1.)
        single_point_estimate = False
        num_repet_sample_estimate = 5

        print 'selection_method: %s' % selection_method
        print "Stimuli_generation: %s" % stimuli_generation

        FI_samples_curv = np.zeros((samples_space.size, 3), dtype=float)
        FI_samples_curv_quantiles = np.zeros((samples_space.size, 3), dtype=float)
        FI_samples_curv_all = []
        FI_samples_samples = np.zeros((samples_space.size, 3), dtype=float)
        FI_samples_samples_quantiles = np.zeros((samples_space.size, 3), dtype=float)
        FI_samples_samples_all = []
        FI_samples_precision = np.zeros(samples_space.size, dtype=float)
        FI_samples_precision_quantiles = np.zeros((samples_space.size, 3), dtype=float)
        FI_samples_precision_all = []

        for i, num_samples in enumerate(samples_space):
            print "Doing for %d num_samples" % num_samples

            # Build the random network
            time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
            cued_feature_time = T-1

            if code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(M, R=R, scale=rc_scale, ratio=40.)
            elif code_type == 'mixed':
                conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            elif code_type == 'wavelet':
                random_network = RandomFactorialNetwork.create_wavelet(M, R=R, scales_number=5)
            else:
                raise ValueError('Code_type is wrong!')

            # Construct the real dataset
            # print "Building the database"
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=stimuli_generation)

            # Measure the noise structure
            # print "Measuring noise structure"
            data_gen_noise = DataGeneratorRFN(5000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=stimuli_generation)
            stat_meas = StatisticsMeasurer(data_gen_noise)

            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, samples %.3f" % num_samples
            print "from curvature..."
            # (FI_M_effect[i], FI_M_effect_std[i])=sampler.estimate_fisher_info_from_posterior_avg(num_points=200, full_stats=trUe)
            # (_, FI_samples_curv[i, 1], FI_samples_curv[i, 0])=sampler.estimate_fisher_info_from_posterior_avg(num_points=500, full_stats=trUe)

            if single_point_estimate:
                # Should estimate everything at specific theta/datapoint?
                FI_samples_curv[i, 0] = sampler.estimate_precision_from_posterior(num_points=num_samples)
            else:
                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=num_samples, full_stats=True)
                (FI_samples_curv[i, 0], FI_samples_curv[i, 1])=(fi_curv_dict['median'], fi_curv_dict['std'])
                FI_samples_curv_quantiles[i] = spst.mstats.mquantiles(fi_curv_dict['all'])

                FI_samples_curv_all.append(fi_curv_dict['all'])

            print FI_samples_curv[i]
            print FI_samples_curv_quantiles[i]

            # FI_M_effect[i] = sampler.estimate_fisher_info_from_posterior(n=0, num_points=500)
            # prec_samples = sampler.estimate_precision_from_samples(n=0, num_samples=1000, num_repetitions=10)
            # (FI_samples_samples[i, 0], FI_samples_samples[i, 1])=(prec_samples['mean'], prec_samples['std'])

            if True:
                print "from samples..."
                if single_point_estimate:
                    prec_samples_dict =  sampler.estimate_precision_from_samples(num_samples=num_samples, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_samples_samples[i, 0], FI_samples_samples[i, 1])=(prec_samples_dict['mean'], prec_samples_dict['std'])
                    FI_samples_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])
                else:
                    prec_samples_dict = sampler.estimate_precision_from_samples_avg(num_samples=num_samples, full_stats=True, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_samples_samples[i, 0], FI_samples_samples[i, 1], FI_samples_samples[i, 2])=(prec_samples_dict['median'], prec_samples_dict['std'], np.max(prec_samples_dict['all']))
                    FI_samples_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])

                    FI_samples_samples_all.append(prec_samples_dict['all'])

            print FI_samples_samples[i]
            print FI_samples_samples_quantiles[i]

            print "from precision of recall..."
            sampler.sample_theta(num_samples=num_samples, burn_samples=100, selection_method=selection_method, selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
            FI_samples_precision[i] = sampler.get_precision()
            FI_samples_precision_quantiles[i] = spst.mstats.mquantiles(FI_samples_precision[i])
            FI_samples_precision_all.append(FI_samples_precision[i])

            print FI_samples_precision[i]

        FI_samples_samples_all = np.array(FI_samples_samples_all)
        FI_samples_curv_all = np.array(FI_samples_curv_all)
        FI_samples_precision_all = np.array(FI_samples_precision_all)

        # Save the results
        dataio.save_variables(['FI_samples_curv', 'FI_samples_samples', 'FI_samples_precision', 'FI_samples_curv_quantiles', 'FI_samples_samples_quantiles', 'FI_samples_precision_quantiles', 'samples_space', 'FI_samples_samples_all', 'FI_samples_curv_all', 'FI_samples_precision_all'], locals())

        # Plot results
        ax = plot_mean_std_area(samples_space, FI_samples_curv[:, 0], FI_samples_curv[:, 1])
        plot_mean_std_area(samples_space, FI_samples_samples[:, 0], FI_samples_samples[:, 1], ax_handle=ax)
        # ax = plot_mean_std_area(samples_space, FI_samples_samples[:, 2], 0.0*FI_samples_samples[:, 1], ax_handle=ax)
        ax = plot_mean_std_area(samples_space, FI_samples_precision, 0.0*FI_samples_precision, ax_handle=ax)

        plt.title('FI dependence on num_samples')
        plt.xlabel('num samples')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision'])

        dataio.save_current_figure('FI_numsamples_comparison_mean_std-{unique_id}.pdf')

        ax2 = plot_median_quantile_area(samples_space, quantiles=FI_samples_curv_quantiles)
        ax2 = plot_median_quantile_area(samples_space, quantiles=FI_samples_samples_quantiles, ax_handle=ax2)
        ax2 = plot_median_quantile_area(samples_space, quantiles=FI_samples_precision_quantiles, ax_handle=ax2)

        plt.title('FI dependence, quantiles shown')
        plt.xlabel('num samples')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision'])

        dataio.save_current_figure('FI_numsamples_comparison_median_std-{unique_id}.pdf')

        if not single_point_estimate:

            for num_samples_i, num_samples in enumerate(samples_space):
                # Show the precision from posterior estimate against the FI from posterior estimate
                plt.figure()
                print num_samples_i, FI_samples_curv_all.shape, FI_samples_samples_all.shape

                plt.plot(FI_samples_curv_all[num_samples_i], FI_samples_samples_all[num_samples_i], 'x')

                idx = np.linspace(FI_samples_curv_all[num_samples_i].min()*0.95, FI_samples_curv_all[num_samples_i].max()*1.05, 100.)

                plt.plot(idx, idx, ':k')
                plt.axis('tight')
                plt.xlabel('Curvature estimate')
                plt.ylabel('Samples estimate')
                plt.title('Comparison Curvature vs samples estimate of FI. %d samples' % num_samples)

                dataio.save_current_figure('FI_numsamples_comparison_curv_samples_allpoints_%d-{unique_id}.pdf' % num_samples)

                # Show the boxplot of each estimate, per number of samples
                plt.figure()
                plt.boxplot([FI_samples_curv_all[num_samples_i], FI_samples_samples_all[num_samples_i].flatten(), FI_samples_precision_all[num_samples_i]])
                plt.title('Comparison Curvature vs samples estimate. %d samples' % num_samples)
                plt.xticks([1, 2, 3], ['Curvature', 'Samples', 'Precision'], rotation=45)

                dataio.save_current_figure('FI_numsamples_comparison_curv_samples_%d-{unique_id}.pdf' % num_samples)


    elif args.subaction == 'rcscale_dependence':
        single_point_estimate = False
        num_repet_sample_estimate = 5

        print "stimuli_generation: %s" % stimuli_generation

        rcscale_space = np.linspace(0.5, 10.0, 10)
        # rcscale_space = np.linspace(args.rc_scale, args.rc_scale, 1.)

        FI_rc_curv = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_curv_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_curv_all = []
        FI_rc_samples = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_samples_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_samples_all = []
        FI_rc_precision = np.zeros(rcscale_space.size, dtype=float)
        FI_rc_precision_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_precision_all = []
        FI_rc_theo = np.zeros((rcscale_space.size, 2), dtype=float)
        FI_rc_theo_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_theo_all = []

        for i, rc_scale in enumerate(rcscale_space):


            # Build the random network
            time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
            cued_feature_time = T-1

            if code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R, scale_moments=(rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(M, R=R, scale=rc_scale, ratio=40.)
            elif code_type == 'mixed':
                conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            elif code_type == 'wavelet':
                random_network = RandomFactorialNetwork.create_wavelet(M, R=R, scales_number=5)
            else:
                raise ValueError('Code_type is wrong!')

            # Construct the real dataset
            # print "Building the database"
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=stimuli_generation)

            # Measure the noise structure
            # print "Measuring noise structure"
            data_gen_noise = DataGeneratorRFN(5000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=stimuli_generation)
            stat_meas = StatisticsMeasurer(data_gen_noise)

            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, rcscale %.3f" % rc_scale
            print "from curvature..."
            # (FI_M_effect[i], FI_M_effect_std[i])=sampler.estimate_fisher_info_from_posterior_avg(num_points=200, full_stats=trUe)
            # (_, FI_rc_curv[i, 1], FI_rc_curv[i, 0])=sampler.estimate_fisher_info_from_posterior_avg(num_points=500, full_stats=trUe)
            if single_point_estimate:
                # Should estimate everything at specific theta/datapoint?
                FI_rc_curv[i, 0] = sampler.estimate_precision_from_posterior(num_points=500)
            else:
                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg_randomsubset(subset_size=N/10, num_points=500, full_stats=True)
                (FI_rc_curv[i, 0], FI_rc_curv[i, 1])=(fi_curv_dict['median'], fi_curv_dict['std'])
                FI_rc_curv_quantiles[i] = spst.mstats.mquantiles(fi_curv_dict['all'])

                FI_rc_curv_all.append(fi_curv_dict['all'])

            print FI_rc_curv[i]
            print FI_rc_curv_quantiles[i]

            # FI_M_effect[i] = sampler.estimate_fisher_info_from_posterior(n=0, num_points=500)
            # prec_samples = sampler.estimate_precision_from_samples(n=0, num_samples=1000, num_repetitions=10)
            # (FI_rc_samples[i, 0], FI_rc_samples[i, 1])=(prec_samples['mean'], prec_samples['std'])

            if True:
                print "from samples..."

                if single_point_estimate:
                    prec_samples_dict =  sampler.estimate_precision_from_samples(num_samples=300, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1])=(prec_samples_dict['mean'], prec_samples_dict['std'])
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])
                else:
                    prec_samples_dict = sampler.estimate_precision_from_samples_avg_randomsubset(subset_size=N/10, num_samples=300, full_stats=True, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1], FI_rc_samples[i, 2])=(prec_samples_dict['median'], prec_samples_dict['std'], np.max(prec_samples_dict['all']))
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])

                    FI_rc_samples_all.append(prec_samples_dict['all'])


                print FI_rc_samples[i]
                print FI_rc_samples_quantiles[i]


            # Compute theoretical values
            print "theoretical FI"
            FI_rc_theo[i, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=sampler.noise_covariance)
            FI_rc_theo[i, 1] = random_network.compute_fisher_information_theoretical(sigma=sigma_x+sigma_y)
            FI_rc_theo_quantiles[i] = spst.mstats.mquantiles(FI_rc_theo[i, 0])
            FI_rc_theo_all.append(FI_rc_theo[i])

            print FI_rc_theo

            print "from precision of recall..."
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=100, selection_method=selection_method, selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
            FI_rc_precision[i] = sampler.get_precision()
            FI_rc_precision_quantiles[i] = spst.mstats.mquantiles(FI_rc_precision[i])
            FI_rc_precision_all.append(FI_rc_precision[i])

            print FI_rc_precision[i]

        FI_rc_curv_all = np.array(FI_rc_curv_all)
        FI_rc_samples_all = np.array(FI_rc_samples_all)
        FI_rc_precision_all = np.array(FI_rc_precision_all)
        FI_rc_theo_all = np.array(FI_rc_theo_all)


        # Save the results
        dataio.save_variables(['FI_rc_curv', 'FI_rc_samples', 'FI_rc_precision', 'FI_rc_curv_quantiles', 'FI_rc_samples_quantiles', 'FI_rc_precision_quantiles', 'rcscale_space', 'FI_rc_curv_all', 'FI_rc_samples_all', 'FI_rc_precision_all', 'FI_rc_theo', 'FI_rc_theo_quantiles', 'FI_rc_theo_all'], locals())

        # Plot results
        ax = plot_mean_std_area(rcscale_space, FI_rc_curv[:, 0], FI_rc_curv[:, 1])
        plot_mean_std_area(rcscale_space, FI_rc_samples[:, 0], FI_rc_samples[:, 1], ax_handle=ax)
        # ax = plot_mean_std_area(rcscale_space, FI_rc_samples[:, 2], 0.0*FI_rc_samples[:, 1], ax_handle=ax)
        ax = plot_mean_std_area(rcscale_space, FI_rc_precision, 0.0*FI_rc_precision, ax_handle=ax)
        ax = plot_mean_std_area(rcscale_space, FI_rc_theo[:, 0], 0.0*FI_rc_theo[:, 0], ax_handle=ax)
        ax = plot_mean_std_area(rcscale_space, FI_rc_theo[:, 1], 0.0*FI_rc_theo[:, 1], ax_handle=ax)

        plt.title('FI dependence on rcscale')
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision', 'Theoretical sum', 'Theoretical large N'])

        dataio.save_current_figure("FI_rcscale_comparison_mean_std_{unique_id}.pdf")

        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_curv_quantiles)
        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_samples_quantiles, ax_handle=ax2)
        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_precision_quantiles, ax_handle=ax2)
        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_theo_quantiles, ax_handle=ax2)

        plt.title('FI dependence, quantiles shown')
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision', 'Theoretical sum'])

        dataio.save_current_figure('FI_rcscale_comparison_median_std_{unique_id}.pdf')

        if False and not single_point_estimate:

            for rc_scale_i, rc_scale in enumerate(rcscale_space):
                # Show the precision from posterior estimate against the FI from posterior estimate
                plt.figure()

                plt.rcParams['font.size'] = 16

                plt.plot(FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i], 'bx')

                idx = np.linspace(FI_rc_curv_all[rc_scale_i].min()*0.95, FI_rc_curv_all[rc_scale_i].max()*1.05, 100.)

                plt.plot(idx, idx, ':k')
                plt.axis('tight')
                plt.xlabel('Curvature estimate')
                plt.ylabel('Samples estimate')
                plt.title('Comparison Curvature vs samples estimate of FI. Rscale: %.2f' % rc_scale)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_allpoints_%d-{unique_id}.pdf' % rc_scale)

                # Show the boxplot of each estimate, per number of samples
                plt.figure()
                b = plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i].flatten(), FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
                # plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
                for key in ['medians', 'boxes', 'whiskers', 'caps']:
                    for line in b[key]:
                        line.set_linewidth(2)
                plt.title('Comparison Curvature vs samples estimate. Rscale: %.2f' % rc_scale)
                plt.xticks([1, 2, 3, 4, 5], ['Curvature', 'Samples', 'Precision', 'Theo', 'Theo large N'], rotation=45)
                # plt.xticks([1, 2, 3, 4], ['Curvature', 'Precision', 'Theo', 'Theo large N'], rotation=45)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_%d-{label}_{unique_id}.pdf' % rc_scale)

    else:
        raise ValueError('Wrong subaction!')


    return locals()




def init_everything(parameters):

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = parameters['T']-1

    if parameters['code_type'] == 'conj':
        random_network = RandomFactorialNetwork.create_full_conjunctive(parameters['M'], R=parameters['R'], scale_moments=(parameters['rc_scale'], 0.0001), ratio_moments=(1.0, 0.0001))
    elif parameters['code_type'] == 'feat':
        random_network = RandomFactorialNetwork.create_full_features(parameters['M'], R=parameters['R'], scale=parameters['rc_scale'], ratio=40.)
    elif parameters['code_type'] == 'mixed':
        conj_params = dict(scale_moments=(parameters['rc_scale'], 0.001), ratio_moments=(1.0, 0.0001))
        feat_params = dict(scale=parameters['rc_scale2'], ratio=40.)

        random_network = RandomFactorialNetwork.create_mixed(parameters['M'], R=parameters['R'], ratio_feature_conjunctive=parameters['ratio_conj'], conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif parameters['code_type'] == 'wavelet':
        random_network = RandomFactorialNetwork.create_wavelet(parameters['M'], R=parameters['R'], scales_number=5)
    else:
        raise ValueError('Code_type is wrong!')

    # Construct the real dataset
    # print "Building the database"
    data_gen = DataGeneratorRFN(parameters['N'], parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])

    # Measure the noise structure
    # print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(5000, parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])
    stat_meas = StatisticsMeasurer(data_gen_noise)

    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

    return (random_network, data_gen, stat_meas, sampler)


def launcher_do_compare_fisher_info_theo(args):
    '''
        Compare the finite N and large N limit versions of the Fisher information
    '''

    all_parameters = vars(args)
    print all_parameters

    # Create DataIO
    #  (complete label with current variable state)
    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))

    # Fix some parameters
    all_parameters['stimuli_generation'] = 'constant'

    # Compute precision as well?
    do_precision = all_parameters['do_precision']

    rcscale_space = np.linspace(0.5, 20.0, 10)
    M_space = np.arange(5, 30, 2)**2.

    result_FI_rc_theo_finiteN = np.zeros((rcscale_space.size, M_space.size))
    result_FI_rc_theo_largeN = np.zeros((rcscale_space.size, M_space.size))
    result_precision = np.zeros((rcscale_space.size, M_space.size))

    search_progress = progress.Progress(rcscale_space.size*M_space.size)
    save_every = 1
    run_counter = 0
    ax = None
    ax_2 = None
    ax_3 = None


    for i, rc_scale in enumerate(rcscale_space):
        for j, M in enumerate(M_space):
            print "Rcscale: %.2f, M: %d. %.1f%% %s/%s" % (rc_scale, M, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            all_parameters['rc_scale']  = rc_scale
            all_parameters['M']         = M

            (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

            # Compute the Fisher info
            result_FI_rc_theo_finiteN[i, j] = random_network.compute_fisher_information(cov_stim=sampler.noise_covariance)
            result_FI_rc_theo_largeN[i, j] = random_network.compute_fisher_information_theoretical(sigma=all_parameters['sigmax'])

            print result_FI_rc_theo_finiteN, '\n', result_FI_rc_theo_largeN

            # Compute precision
            if do_precision:
                print 'Precision...'
                sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=True)
                result_precision[i, j] = sampler.get_precision()

                print result_precision


            search_progress.increment()
            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables_default(locals())

                # Plots
                plt.figure(1)
                plt.plot(rcscale_space, result_FI_rc_theo_finiteN, '--', rcscale_space, result_FI_rc_theo_largeN, linewidth=2)
                plt.legend(['Finite N', 'Large N'])
                plt.xlabel('kappa')

                dataio.save_current_figure('FI_compare_theo_finite-lines-{label}_{unique_id}.pdf')

                ax, _ = pcolor_2d_data((result_FI_rc_theo_largeN/result_FI_rc_theo_finiteN - 1.0)**2., x=rcscale_space, y=M_space, log_scale=True, ax_handle=ax)

                dataio.save_current_figure('FI_compare_theo_finite-2d-{label}_{unique_id}.pdf')

                ax_2, _ = pcolor_2d_data(result_precision, x=rcscale_space, y=M_space, ax_handle=ax_2)
                dataio.save_current_figure('FI_compare_theo_finite-precision2d-{label}_{unique_id}.pdf')

                ax_3, _ = pcolor_2d_data((result_precision - result_FI_rc_theo_finiteN)**2.0+1e-10, x=rcscale_space, y=M_space, ax_handle=ax_3)
                dataio.save_current_figure('FI_compare_theo_finite-precisionvstheo-{label}_{unique_id}.pdf')


            run_counter += 1




    return locals()



def launcher_do_fisher_information_param_search(args):
    '''
        Get the fisher information for varying values of sigmax and rc_scale.

        - First see how different FI estimators behave.
        - Then build a constraint between sigmax/rc_scale based on the experimental value for 1 object.
    '''

    all_parameters = vars(args)
    data_to_plot = {}

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo', 'FI_rc_truevar']
    save_every = 5
    run_counter = 0

    rcscale_space = np.linspace(0.1, 20.0, 10.)
    # rcscale_space = np.linspace(2., 2., 1.)

    sigma_space = np.linspace(0.1, 0.8, 10.)
    # sigma_space = np.linspace(0.1, 0.1, 1.)

    FI_rc_curv = np.zeros((rcscale_space.size, sigma_space.size, 2), dtype=float)
    FI_rc_precision = np.zeros((rcscale_space.size, sigma_space.size), dtype=float)
    FI_rc_theo = np.zeros((rcscale_space.size, sigma_space.size, 2), dtype=float)
    FI_rc_truevar = np.zeros((rcscale_space.size, sigma_space.size, 2), dtype=float)

    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*sigma_space.size)

    for j, sigma in enumerate(sigma_space):
        for i, rc_scale in enumerate(rcscale_space):
            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, rcscale %.3f, sigma %.3f. %.2f%%, %s left - %s" % (rc_scale, sigma, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            # Current parameter values
            all_parameters['rc_scale']  = rc_scale
            all_parameters['sigmax']    = sigma

            # Instantiate the sampler
            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

            print "from curvature..."
            fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=1000, full_stats=True)
            (FI_rc_curv[i, j, 0], FI_rc_curv[i, j, 1]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
            print FI_rc_curv[i, j]

            print "theoretical FI"

            FI_rc_theo[i, j, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=sampler.noise_covariance)
            FI_rc_theo[i, j, 1] = random_network.compute_fisher_information_theoretical(sigma=all_parameters['sigmax'], kappa1=all_parameters['rc_scale'], kappa2=all_parameters['rc_scale'])
            print FI_rc_theo[i, j]

            print "true variance..."
            fi_truevar_dict = sampler.estimate_truevariance_from_posterior_avg(full_stats=True)
            (FI_rc_truevar[i, j, 0], FI_rc_truevar[i, j, 1]) =  (fi_truevar_dict['mean'], fi_truevar_dict['std'])
            print FI_rc_truevar[i, j]

            print "from precision of recall..."
            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
            FI_rc_precision[i, j] = sampler.get_precision()
            print FI_rc_precision[i, j]

            search_progress.increment()

            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables(variables_to_save, locals())

                # plots
                for curr_data in variables_to_save:
                    data_to_plot[curr_data] = locals()[curr_data]

                plots_fisher_info_param_search(data_to_plot, dataio)

            run_counter += 1


    return locals()


def launcher_do_fisher_information_M_effect(args):
    '''
        Get the fisher information for varying values of sigmax and rc_scale.

        - First see how different FI estimators behave.
        - Then build a constraint between sigmax/rc_scale based on the experimental value for 1 object.
    '''

    all_parameters = vars(args)
    data_to_plot = {}

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'M_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo', 'FI_rc_truevar']
    save_every = 1
    run_counter = 0

    # rcscale_space = np.linspace(0.1, 20.0, 10.)
    rcscale_space = np.linspace(4., 4., 3.)

    # M_space = np.arange(30, 30, 2, dtype=int)**2.
    M_space = np.array([900])

    FI_rc_curv = np.zeros((rcscale_space.size, M_space.size, 2), dtype=float)
    FI_rc_precision = np.zeros((rcscale_space.size, M_space.size), dtype=float)
    FI_rc_theo = np.zeros((rcscale_space.size, M_space.size, 2), dtype=float)
    FI_rc_truevar = np.zeros((rcscale_space.size, M_space.size, 2), dtype=float)

    # Show the progress in a nice way
    search_progress = progress.Progress(rcscale_space.size*M_space.size)

    print rcscale_space, M_space

    for j, M in enumerate(M_space):
        for i, rc_scale in enumerate(rcscale_space):
            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, rcscale %.3f, M %.3f. %.2f%%, %s left - %s" % (rc_scale, M, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

            # Current parameter values
            all_parameters['rc_scale']  = rc_scale
            all_parameters['M']         = M

            # Instantiate the sampler
            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

            print "from curvature..."
            fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=1000, full_stats=True)
            (FI_rc_curv[i, j, 0], FI_rc_curv[i, j, 1]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
            print FI_rc_curv[i, j]

            print "theoretical FI"
            FI_rc_theo[i, j, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=sampler.noise_covariance)
            FI_rc_theo[i, j, 1] = random_network.compute_fisher_information_theoretical(sigma=all_parameters['sigmax'], kappa1=all_parameters['rc_scale'], kappa2=all_parameters['rc_scale'])
            print FI_rc_theo[i, j]

            print "true variance..."
            fi_truevar_dict = sampler.estimate_truevariance_from_posterior_avg(full_stats=True)
            (FI_rc_truevar[i, j, 0], FI_rc_truevar[i, j, 1]) =  (fi_truevar_dict['mean'], fi_truevar_dict['std'])
            print FI_rc_truevar[i, j]

            print "from precision of recall..."
            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
            FI_rc_precision[i, j] = sampler.get_precision()
            print FI_rc_precision[i, j]

            search_progress.increment()

            if run_counter % save_every == 0 or search_progress.done():
                dataio.save_variables(variables_to_save, locals())

                # plots
                for curr_data in variables_to_save:
                    data_to_plot[curr_data] = locals()[curr_data]

                # plots_fisher_info_param_search(data_to_plot, dataio)

            run_counter += 1


    return locals()



def launcher_do_fisher_information_param_search_pbs(args):
    '''
        Get the fisher information for varying values of sigmax and rc_scale.

        - First see how different FI estimators behave.
        - Then build a constraint between sigmax/rc_scale based on the experimental value for 1 object.
    '''

    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    print all_parameters

    data_to_plot = {}


    dataio = DataIO(output_folder=all_parameters['output_directory'], label=all_parameters['label'].format(**all_parameters))
    variables_to_save = ['repet_i', 'num_repetitions']

    do_samples = True
    save_every = 5
    run_counter = 0

    num_repetitions = all_parameters['num_repetitions']

    # rcscale_space = np.linspace(0.5, 15.0, 21.)
    # rcscale_space = np.linspace(all_parameters['rc_scale'], all_parameters['rc_scale'], 1.)

    # sigma_space = np.linspace(0.01, 1.1, 20.)
    # sigma_space = np.linspace(all_parameters['sigmax'], all_parameters['sigmax'], 1.)

    result_FI_rc_curv_mult = np.empty((2, num_repetitions), dtype=float)*np.nan
    result_FI_rc_curv_all  = np.empty((all_parameters['N'], num_repetitions), dtype=float)*np.nan
    result_FI_rc_precision_mult = np.empty((num_repetitions), dtype=float)*np.nan
    result_FI_rc_theo_mult = np.empty((2, num_repetitions), dtype=float)*np.nan
    result_FI_rc_truevar_mult = np.empty((2, num_repetitions), dtype=float)*np.nan
    result_FI_rc_samples_mult = np.empty((2, num_repetitions), dtype=float)*np.nan
    result_FI_rc_samples_all = np.empty((all_parameters['N'], num_repetitions), dtype=float)*np.nan

    # Show the progress in a nice way
    search_progress = progress.Progress(num_repetitions)

    for repet_i in xrange(num_repetitions):
        ### Estimate the Fisher Information
        print "Estimating the Fisher Information, sigmax %.3f (%d/%d). %.2f%%, %s left - %s" % (all_parameters['sigmax'], repet_i+1, num_repetitions, search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

        # Current parameter values
        # all_parameters['rc_scale']  = rc_scale
        # all_parameters['sigmax']    = sigma


        ### WORK UNIT
        (random_network, data_gen, stat_meas, sampler) = launchers.init_everything(all_parameters)

        print "from curvature..."
        fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=1000, full_stats=True)
        (result_FI_rc_curv_mult[0, repet_i], result_FI_rc_curv_mult[1, repet_i]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
        result_FI_rc_curv_all[:, repet_i] = fi_curv_dict['all']

        print result_FI_rc_curv_mult[:, repet_i]

        print "theoretical FI"
        result_FI_rc_theo_mult[0, repet_i] = sampler.estimate_fisher_info_theocov(use_theoretical_cov=False, kappa_different=True)
        # result_FI_rc_theo_mult[1, repet_i] = random_network.compute_fisher_information_theoretical(sigma=all_parameters['sigmax'])
        result_FI_rc_theo_mult[1, repet_i] = sampler.estimate_fisher_info_theocov_largen(use_theoretical_cov=True)

        print result_FI_rc_theo_mult[:, repet_i]

        print "true variance..."
        fi_truevar_dict = sampler.estimate_truevariance_from_posterior_avg(full_stats=True)
        (result_FI_rc_truevar_mult[0, repet_i], result_FI_rc_truevar_mult[1, repet_i]) =  (fi_truevar_dict['mean'], fi_truevar_dict['std'])
        print result_FI_rc_truevar_mult[:, repet_i]

        if do_samples:
            prec_samples_dict = sampler.estimate_precision_from_samples_avg_randomsubset(subset_size=all_parameters['N']/10, num_samples=all_parameters['num_samples'], full_stats=True, num_repetitions=10, selection_method='last')
            (result_FI_rc_samples_mult[0, repet_i], result_FI_rc_samples_mult[1, repet_i]) = (prec_samples_dict['mean'], prec_samples_dict['std'])
            result_FI_rc_samples_all[:, repet_i] = prec_samples_dict['all'].flatten()

        print "from precision of recall..."
        sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=all_parameters['burn_samples'], selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['selection_num_samples'], integrate_tc_out=False, debug=False)
        result_FI_rc_precision_mult[repet_i] = sampler.get_precision()
        print result_FI_rc_precision_mult[repet_i]
        ### DONE WORK UNIT

        search_progress.increment()
        if run_counter % save_every == 0 or search_progress.done():
            dataio.save_variables_default(locals(), variables_to_save)

            # # plots
            # for curr_data in variables_to_save:
            #     data_to_plot[curr_data] = locals()[curr_data]
            # plots_fisher_info_param_search(data_to_plot, dataio)

        run_counter += 1

    return locals()



def plots_fisher_info_param_search(data_to_plot, dataio=None, save_figures=True, fignum=None):
    '''
        Create and save a few plots for the fisher information parameter search
    '''

    # Sanity check, verify that we have all the data we will be plotting
    required_variables = ['rcscale_space', 'sigma_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo']

    assert set(required_variables) <= set(data_to_plot), "This dataset is not complete. Require %s" % required_variables

    # 2D plots
    # pcolor_2d_data(dist_2d, x=x['space'], y=y['space'], xlabel=x['label'], ylabel=y['label'], title=title)

    plt.hold(False)

    pcolor_2d_data(data_to_plot['FI_rc_curv'][..., 0], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='FI from curvature, sigma/rcscale', fignum=fignum, colorbar=False)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_2d_curve_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1

    pcolor_2d_data(data_to_plot['FI_rc_precision'], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='FI recall precision, sigma/rcscale', fignum=fignum, colorbar=False)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_2d_precision_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1

    pcolor_2d_data(data_to_plot['FI_rc_theo'][..., 0], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='FI Theoretical sum, sigma/rcscale', fignum=fignum, colorbar=False)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_2d_theo_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1

    pcolor_2d_data(data_to_plot['FI_rc_theo'][..., 1], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='FI Theoretical large N, sigma/rcscale', fignum=fignum, colorbar=False)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_2d_theoN_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1

    if 'FI_rc_truevar' in data_to_plot:
        pcolor_2d_data(1./data_to_plot['FI_rc_truevar'][..., 0], x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='Variance estimated from posterior, sigma/rcscale', fignum=fignum, colorbar=False)
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_2d_truevar_{label}_{unique_id}.pdf")
        if fignum:
            fignum += 1

    # 1D plots
    if False:
        for i, sigma in enumerate(data_to_plot['sigma_space']):

            ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_curv'][:, i, 0], data_to_plot['FI_rc_curv'][:, i, 1], fignum=fignum)
            plt.hold(True)
            ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_precision'][:, i], 0.0*data_to_plot['FI_rc_precision'][:, i], ax_handle=ax)
            ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][:, i, 0], 0.0*data_to_plot['FI_rc_theo'][:, i, 0], ax_handle=ax)
            ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][:, i, 1], 0.0*data_to_plot['FI_rc_theo'][:, i, 1], ax_handle=ax)

            ax.set_title('FI dependence on rcscale, sigma %.2f' % sigma)
            ax.set_xlabel('rcscale')
            ax.set_ylabel('FI')
            ax.legend(['Curvature', 'Recall precision', 'Theoretical sum', 'Theoretical large N'])

            ax.get_figure().canvas.draw()

            if fignum:
                fignum += 1

            if save_figures:
                dataio.save_current_figure("FI_paramsearch_rcscale_sigmafixed%.2f_mean_std_{label}_{unique_id}.pdf" % sigma)

            plt.hold(False)


    plt.figure(fignum)
    plt.plot(data_to_plot['rcscale_space'], data_to_plot['FI_rc_curv'][..., 0])
    plt.xlabel('rcscale')
    plt.ylabel('FI')
    plt.title('FI curve, for different sigma')
    if fignum:
        fignum += 1
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_curv_rcscalesigma_{label}_{unique_id}.pdf")


    plt.figure(fignum)
    plt.plot(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][..., 0])
    plt.xlabel('rcscale')
    plt.ylabel('FI')
    plt.title('FI theo, for different sigma')
    if fignum:
        fignum += 1
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_theo_rcscalesigma_{label}_{unique_id}.pdf")


    plt.figure(fignum)
    plt.plot(data_to_plot['rcscale_space'], data_to_plot['FI_rc_precision'])
    plt.xlabel('rcscale')
    plt.ylabel('FI')
    plt.title('FI precision, for different sigma')
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_precision_rcscalesigma_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1


    if np.any(data_to_plot['FI_rc_curv'][..., 0]>0):
        plt.figure(fignum)
        plt.semilogy(data_to_plot['rcscale_space'], data_to_plot['FI_rc_curv'][..., 0])
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.title('FI curve, for different sigma')
        if fignum:
            fignum += 1
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_semilogy_curv_rcscalesigma_{label}_{unique_id}.pdf")
    else:
        print "FI_rc_curve all 0"

    if np.any(data_to_plot['FI_rc_precision']>0):
        plt.figure(fignum)
        plt.semilogy(data_to_plot['rcscale_space'], data_to_plot['FI_rc_precision'])
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.title('FI precision, for different sigma')
        if fignum:
            fignum += 1
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_semilogy_precision_rcscalesigma_{label}_{unique_id}.pdf")
    else:
        print "FI_rc_precision all 0"

    if np.any(data_to_plot['FI_rc_theo'][..., 0]>0):
        plt.figure(fignum)
        plt.semilogy(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][..., 0])
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.title('FI theo, for different sigma')
        if fignum:
            fignum += 1
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_semilogy_theo_rcscalesigma_{label}_{unique_id}.pdf")
    else:
        print "FI_rc_theo all 0"

    if np.any(data_to_plot['FI_rc_theo'][..., 1]>0):
        plt.figure(fignum)
        plt.semilogy(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][..., 1])
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.title('FI theo large n, for different sigma')
        if fignum:
            fignum += 1
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_semilogy_theolargen_rcscalesigma_{label}_{unique_id}.pdf")
    else:
        print "FI_rc_theo large n all 0"


    if 'FI_rc_truevar' in data_to_plot:
        curvtruevar_ratio = data_to_plot['FI_rc_curv'][..., 0]*data_to_plot['FI_rc_truevar'][..., 0]
        curvtruevar_ratio_dist = (curvtruevar_ratio - 1.0)**2.
        curvtruevar_ratio_dist_filtered = curvtruevar_ratio_dist.copy()
        curvtruevar_ratio_dist_filtered[curvtruevar_ratio_dist > 1.0] = np.nan
        curvtruevar_ratio_filtered = curvtruevar_ratio.copy()
        curvtruevar_ratio_filtered[curvtruevar_ratio_dist > 5.0] = np.nan


        pcolor_2d_data(curvtruevar_ratio_dist_filtered, x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='curv/truevar distance, sigma/rcscale', fignum=fignum, colorbar=True)
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_2d_curvtruevardist_{label}_{unique_id}.pdf")
        if fignum:
            fignum += 1

        pcolor_2d_data(curvtruevar_ratio_filtered, x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='curv/truevar, sigma/rcscale', fignum=fignum, colorbar=True)
        if save_figures:
            dataio.save_current_figure("FI_paramsearch_2d_curvtruevar_{label}_{unique_id}.pdf")
        if fignum:
            fignum += 1

    plt.hold(True)


def plots_ratio_checkers_fisherinfo(data_to_plot, dataio=None, save_figures=True, fignum=None):
    '''
        Create several plots to check the relationship between the recall precision and the theoretical FI
    '''

    required_variables = ['rcscale_space', 'sigma_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo']

    assert set(required_variables) <= set(data_to_plot), "This dataset is not complete. Require %s" % required_variables

    ratio_recall_curv = data_to_plot['FI_rc_curv'][..., 0]/data_to_plot['FI_rc_precision']
    ratio_recall_theo = data_to_plot['FI_rc_theo'][..., 0]/data_to_plot['FI_rc_precision']

    # Remove all parameters tuples that deviate too much from a 2.0 ratio (sign of wrong inference)
    filter_threshold = 5.0
    ratio_distance_2 = (ratio_recall_curv - 2.0)**2.
    ratio_distance_2[ratio_distance_2 > filter_threshold] = np.nan

    ratio_recall_curv_filtered = ratio_recall_curv.copy()
    ratio_recall_curv_filtered[(ratio_recall_curv - 2.0)**2. > filter_threshold] = np.nan

    plt.figure(fignum)
    plt.plot(data_to_plot['rcscale_space'], nanmean(ratio_recall_curv, axis=1), hold=False)
    plt.plot(data_to_plot['rcscale_space'], nanmean(ratio_recall_theo, axis=1), hold=True)
    plt.xlabel('rcscale')
    plt.ylabel('ratio')
    plt.title('Ratio between precision and curvature/theo FI')
    plt.legend(['Curv/precision', 'Theo/precision'])
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_ratio_precisioncurv_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1


    plt.figure(fignum)
    plt.plot(data_to_plot['rcscale_space'], ratio_recall_curv, hold=False)
    plt.plot(data_to_plot['rcscale_space'], ratio_recall_theo, hold=True)
    plt.xlabel('rcscale')
    plt.ylabel('ratio')
    plt.title('Ratio between precision and curvature/theo FI, full')
    plt.legend(['Curv/precision', 'Theo/precision'])
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_ratio_precisioncurv_full_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1


    pcolor_2d_data(ratio_recall_curv, x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='curv/precision, sigma/rcscale', fignum=fignum, colorbar=True)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_ratio_precisioncurv_2d_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1

    pcolor_2d_data(ratio_recall_curv_filtered, x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='curv/precision filtered, sigma/rcscale', fignum=fignum, colorbar=True)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_ratiofiltered_precisioncurv_2d_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1


    pcolor_2d_data(ratio_distance_2, x=data_to_plot['rcscale_space'], y=data_to_plot['sigma_space'], xlabel='rc scale', ylabel='sigma', title='curv/precision distance, sigma/rcscale', fignum=fignum, colorbar=True)
    if save_figures:
        dataio.save_current_figure("FI_paramsearch_ratiodistance_precisioncurv_2d_{label}_{unique_id}.pdf")
    if fignum:
        fignum += 1



def plots_M_effect_multipleruns(data_to_plot, dataio=None, save_figures=False, fignum=None):
    '''
        Create some plots to show the effect of M on the ratio and the FI obtained from the curvature,
         theoretical formula and recall precision estimation.

        Checks and assumes multiple runs per each rc_scale. Looks
    '''

    ## Multiple runs are on dimension 0, associated with rcscale...

    # Plots of FI
    ax = plot_mean_std_area(data_to_plot['M_space'], np.mean(data_to_plot['FI_rc_curv'][..., 0], axis=0), np.mean(data_to_plot['FI_rc_curv'][..., 1], axis=0))
    ax = plot_mean_std_area(data_to_plot['M_space'], np.mean(data_to_plot['FI_rc_theo'][..., 0], axis=0), np.std(data_to_plot['FI_rc_theo'][..., 0], axis=0), ax_handle=ax)
    ax = plot_mean_std_area(data_to_plot['M_space'], np.mean(data_to_plot['FI_rc_precision'], axis=0), np.std(data_to_plot['FI_rc_precision'], axis=0), ax_handle=ax)
    # ax = plot_mean_std_area(data_to_plot['M_space'], 1./np.mean(data_to_plot['FI_rc_truevar'][..., 0], axis=0)**2., np.mean(data_to_plot['FI_rc_truevar'][..., 1], axis=0), ax_handle=ax)

    plt.legend(['Curv', 'Theory', 'Precision', 'True variance'])
    plt.xlabel('M')
    plt.ylabel('FI')

    # Plots of ratios
    # ratio_recall_curv = data_to_plot['FI_rc_curv'][..., 0]/data_to_plot['FI_rc_precision']
    ratio_recall_theo = data_to_plot['FI_rc_theo'][..., 0]/data_to_plot['FI_rc_precision']
    plot_mean_std_area(data_to_plot['M_space'], np.mean(ratio_recall_theo, axis=0), np.std(ratio_recall_theo, axis=0))
    plt.title('Ratio curv/prec as function of M')
    plt.xlabel('M')
    plt.ylabel('Ratio')



def launcher_reload_fisher_information_param_search(args):
    '''
        Reload data created from launcher_do_fisher_information_param_search
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_param_search"


    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()


    # Plots
    plots_fisher_info_param_search(loaded_data, dataio, save_figures=False)
    plots_ratio_checkers_fisherinfo(loaded_data, save_figures=False)

    return locals()



def launcher_reload_fisher_information_M_effect(args):
    '''
        Reload data created from launcher_do_fisher_information_M_effect
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_M_effect"


    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()

    # Small hack to use the same plotting than before...
    loaded_data['sigma_space'] = loaded_data['M_space']

    # Plots
    plots_fisher_info_param_search(loaded_data, dataio, save_figures=False)
    plots_ratio_checkers_fisherinfo(loaded_data, save_figures=False)

    return locals()



def launcher_reload_fisher_information_M_effect_multipleruns(args):
    '''
        Reload data created from launcher_do_fisher_information_M_effect, with multiple runs for each rcscale
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_M_effect"


    # dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()

    # Handle unfinished runs
    # variables_to_load = ['FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo', 'FI_rc_truevar']
    variables_to_load = ['FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo']
    for var_load in variables_to_load:
        loaded_data[var_load] = np.ma.masked_equal(loaded_data[var_load], 0.0)

    # Plots
    plots_M_effect_multipleruns(loaded_data)

    return locals()





def launcher_reload_fisher_information_param_search_pbs(args):
    '''
        Reload data created from launcher_do_fisher_information_param_search
    '''

    # Need to find a way to provide the dataset_infos nicely...
    dataset_infos = dict(label='New PBS runs, different loading method. Uses the 2D fisher information as a constraint between sigma and rcscale. Also checks the ratio between recall precision and FI curve.',
                    # files='Data/constraint/allfi_M400N300/allfi_*-launcher_do_fisher_information_param_search_pbs-*.npy',
                    # files='Data/constraint/allfi_N200samples300/allfi_*-launcher_do_fisher_information_param_search_pbs-*.npy',
                    files='Data/constraint/allfi_M900N300/allfi_*-launcher_do_fisher_information_param_search_pbs-*.npy',
                    loading_type='args',
                    parameters=('rc_scale', 'sigmax'),
                    variables_to_load=['FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo'],
                    variables_description=('FI curve', 'FI recall precision', 'FI theo'),
                    post_processing=None,
                    )

    # Reload everything
    data_pbs = DataPBS(dataset_infos=dataset_infos, debug=True)


    # Extract the information properly.
    extracted_data = {}
    # Put the data
    for var_loaded in data_pbs.dataset_infos['variables_to_load']:
        extracted_data[var_loaded] = np.squeeze(data_pbs.dict_arrays[var_loaded]['results'])

    # Put the axes
    extracted_data['rcscale_space'] = data_pbs.loaded_data['parameters_uniques']['rc_scale']
    extracted_data['sigma_space'] = data_pbs.loaded_data['parameters_uniques']['sigmax']


    # Plots
    plots_fisher_info_param_search(extracted_data, save_figures=False)
    plots_ratio_checkers_fisherinfo(extracted_data, save_figures=False)

    max_div = 100.
    # constrained_fi = 36.94
    constrained_fi = 9.04
    # build_constraint(extracted_data['FI_rc_precision'], constrained_value=constrained_fi, max_divergence=max_div, x=dict(space=extracted_data['rcscale_space'], label='Rc scale'), y=dict(space=extracted_data['sigma_space'], label='Sigma'), title='Precision')

    # build_constraint(extracted_data['FI_rc_theo'][..., 0], constrained_value=constrained_fi*2., max_divergence=max_div, x=dict(space=extracted_data['rcscale_space'], label='Rc scale'), y=dict(space=extracted_data['sigma_space'], label='Sigma'), title='Theo sum')


    return locals()



def launcher_reload_fi_param_search_constraint_building(args):
    '''
        Reload data created from launcher_do_fisher_information_param_search,

        Finds the relationship between sigma and rcscale then, constrained to be close to the
        experimentally measured human FI.
    '''

    # Check that a filename was provided
    input_filename = args.input_filename
    assert input_filename is not '', "Give a file with saved results from launcher_do_fisher_information_param_search"

    # dataio = DataIO(output_folder=args.output_directory, label=args.label)

    # Reload everything
    loaded_data = np.load(input_filename).item()

    if 'M_space' in loaded_data:
        loaded_data['sigma_space'] = loaded_data['M_space']

    max_div = 100.
    constrained_fi = 36.94

    # Check the theoretical data
    build_constraint(loaded_data['FI_rc_theo'][..., 0], constrained_value=constrained_fi*2., max_divergence=max_div, x=dict(space=loaded_data['rcscale_space'], label='Rc scale'), y=dict(space=loaded_data['sigma_space'], label='Sigma'), title='Theo sum')


    build_constraint(loaded_data['FI_rc_theo'][..., 1], constrained_value=constrained_fi*2., max_divergence=max_div, x=dict(space=loaded_data['rcscale_space'], label='Rc scale'), y=dict(space=loaded_data['sigma_space'], label='Sigma'), title='Theo large N')

    # Check the curvature
    if np.any(loaded_data['FI_rc_curv'][..., 0] > 0):
        build_constraint(loaded_data['FI_rc_curv'][..., 0], constrained_value=constrained_fi*2., max_divergence=max_div, x=dict(space=loaded_data['rcscale_space'], label='Rc scale'), y=dict(space=loaded_data['sigma_space'], label='Sigma'), title='Curv')

    # Check the precision
    if np.any(loaded_data['FI_rc_precision'] > 0):
        build_constraint(loaded_data['FI_rc_precision'], constrained_value=constrained_fi, max_divergence=max_div, x=dict(space=loaded_data['rcscale_space'], label='Rc scale'), y=dict(space=loaded_data['sigma_space'], label='Sigma'), title='Precision')

    return locals()



def build_constraint(data_2d, constrained_value=0.0, max_divergence=None, x=None, y=None, title='', fignum=None):
    """
        Returns x-y constrained values, out of the big X-Y dataset, constrained on the
         provided value. Will only return values up to max_divergence away from the constrain (NaN if further away)
    """

    # Compute the distance from the constrained value
    dist_2d = (data_2d - constrained_value)**2.

    # Find the mapping
    y_constrained = y['space'][np.nanargmin(dist_2d, axis=1)]
    x_constrained = x['space'][np.nanargmin(dist_2d, axis=0)]

    # Constrain the maximum error
    if max_divergence:
        dist_2d[dist_2d > max_divergence] = np.nan

    if not x is None and not y is None:
        pcolor_2d_data(dist_2d, x=x['space'], y=y['space'], xlabel=x['label'], ylabel=y['label'], title="%s: Distance to %.2f" % (title, constrained_value), fignum=fignum, label_format="%.1f")
    else:
        pcolor_2d_data(dist_2d, title=title, fignum=fignum)


    plt.figure()
    plt.plot(x['space'], y_constrained)
    plt.xlabel(x['label'])
    plt.ylabel(y['label'])
    plt.title('%s: %s vs %s, constrained' % (title, y['label'], x['label']))


    plt.figure()
    plt.plot(y['space'], x_constrained)
    plt.xlabel(y['label'])
    plt.ylabel(x['label'])
    plt.title('%s: %s vs %s, constrained' % (title, x['label'], y['label']))

    # Fit a line
    lin_fit = fit_line(x['space'], y_constrained, title='%s: Line fit to closest constrained' % title)

    print lin_fit




















