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
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *


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

    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    if args.subaction == '':
        args.subaction = 'M_dependence'

    if args.subaction == 'M_dependence':

        M_space = np.arange(10, 500, 20)
        FI_M_effect = np.zeros_like(M_space, dtype=float)
        FI_M_effect_std = np.zeros_like(M_space, dtype=float)

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
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
            
            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, M %d" % M
            (_, FI_M_effect_std[i], FI_M_effect[i])=sampler.t_all_avg(num_points=200, return_std=trUe)


        # Plot results
        plot_mean_std_area(M_space, FI_M_effect, FI_M_effect_std)
        plt.title('FI dependence on M')
        plt.xlabel('M')
        plt.ylabel('FI')

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
            sampler.sample_theta(num_samples=num_samples, burn_samples=100, selection_method=selection_method, selection_num_samples=num_samples, integrate_tc_out=False, debug=False)
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
        num_repet_sample_estimate = 1

        print "stimuli_generation: %s" % stimuli_generation

        rcscale_space = np.linspace(0.5, 10.0, 10)
        # rcscale_space = np.linspace(4., 4., 1.)
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
                FI_rc_curv[i, 0] = sampler.estimate_precision_from_posterior(num_points=num_samples)
            else:
                fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=1000, full_stats=True)
                (FI_rc_curv[i, 0], FI_rc_curv[i, 1])=(fi_curv_dict['median'], fi_curv_dict['std'])
                FI_rc_curv_quantiles[i] = spst.mstats.mquantiles(fi_curv_dict['all'])

                FI_rc_curv_all.append(fi_curv_dict['all'])

            print FI_rc_curv[i]
            print FI_rc_curv_quantiles[i]
            
            # FI_M_effect[i] = sampler.estimate_fisher_info_from_posterior(n=0, num_points=500)
            # prec_samples = sampler.estimate_precision_from_samples(n=0, num_samples=1000, num_repetitions=10)
            # (FI_rc_samples[i, 0], FI_rc_samples[i, 1])=(prec_samples['mean'], prec_samples['std'])
            
            if False:
                print "from samples..."
                
                if single_point_estimate:
                    prec_samples_dict =  sampler.estimate_precision_from_samples(num_samples=300, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1])=(prec_samples_dict['mean'], prec_samples_dict['std'])
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])
                else:
                    prec_samples_dict = sampler.estimate_precision_from_samples_avg(num_samples=300, full_stats=True, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1], FI_rc_samples[i, 2])=(prec_samples_dict['median'], prec_samples_dict['std'], np.max(prec_samples_dict['all']))
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])

                    FI_rc_samples_all.append(prec_samples_dict['all'])


                print FI_rc_samples[i]
                print FI_rc_samples_quantiles[i]
            

            # Compute theoretical values
            print "theoretical FI"
            FI_rc_theo[i, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=stat_meas.model_parameters['covariances'][-1, 0])
            FI_rc_theo[i, 1] = random_network.compute_fisher_information_theoretical(sigma=sigma_x+sigma_y)
            FI_rc_theo_quantiles[i] = spst.mstats.mquantiles(FI_rc_theo[i, 0])
            FI_rc_theo_all.append(FI_rc_theo[i])

            print FI_rc_theo

            print "from precision of recall..."
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=100, selection_method=selection_method, selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
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

        if not single_point_estimate:

            for rc_scale_i, rc_scale in enumerate(rcscale_space):
                # Show the precision from posterior estimate against the FI from posterior estimate
                plt.figure()

                # plt.plot(FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i], 'x')

                idx = np.linspace(FI_rc_curv_all[rc_scale_i].min()*0.95, FI_rc_curv_all[rc_scale_i].max()*1.05, 100.)

                plt.plot(idx, idx, ':k')
                plt.axis('tight')
                plt.xlabel('Curvature estimate')
                plt.ylabel('Samples estimate')
                plt.title('Comparison Curvature vs samples estimate of FI. Rscale: %d' % rc_scale)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_allpoints_%d-{unique_id}.pdf' % rc_scale)

                # Show the boxplot of each estimate, per number of samples
                plt.figure()
                # plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i].flatten(), FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
                plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_precision_all[rc_scale_i], FI_rc_theo_all[rc_scale_i, 0], FI_rc_theo_all[rc_scale_i, 1]])
                plt.title('Comparison Curvature vs samples estimate. Rscale: %d' % rc_scale)
                # plt.xticks([1, 2, 3, 5], ['Curvature', 'Samples', 'Precision', 'Theo', 'Theo large N'], rotation=45)
                plt.xticks([1, 2, 3, 4], ['Curvature', 'Precision', 'Theo', 'Theo large N'], rotation=45)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_%d-{unique_id}.pdf' % rc_scale)
    
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

        random_network = RandomFactorialNetwork.create_mixed(parameters['M'], R=parameters['R'], ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
    elif code_type == 'wavelet':
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


def launcher_do_fisher_information_param_search(args):
    '''
        Get the fisher information for varying values of sigmax and rc_scale.

        - First see how different FI estimators behave.
        - Then build a constraint between sigmax/rc_scale based on the experimental value for 1 object.
    '''

    all_parameters = vars(args)

    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    variables_to_save = ['rcscale_space', 'sigma_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo']

    rcscale_space = np.linspace(0.5, 10.0, 10.)
    # rcscale_space = np.linspace(4., 4., 1.)
    
    sigma_space = np.linspace(0.01, 1.5, 10.)
    # sigma_space = np.linspace(0.2, 0.2, 1.)

    FI_rc_curv = np.zeros((rcscale_space.size, sigma_space.size, 2), dtype=float)
    FI_rc_precision = np.zeros((rcscale_space.size, sigma_space.size), dtype=float)
    FI_rc_theo = np.zeros((rcscale_space.size, sigma_space.size, 2), dtype=float)
    
    for i, rc_scale in enumerate(rcscale_space):
        for j, sigma in enumerate(sigma_space):
            ### Estimate the Fisher Information
            print "Estimating the Fisher Information, rcscale %.3f, sigma %.3f. %.2f%%" % (rc_scale, sigma, 100.*(i*rcscale_space.size+j)/(rcscale_space.size*sigma_space.size))
            
            # Current parameter values
            all_parameters['rc_scale']  = rc_scale
            all_parameters['sigmax']    = sigma

            # Instantiate the sampler
            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
            
            print "from curvature..."
            fi_curv_dict = sampler.estimate_fisher_info_from_posterior_avg(num_points=1000, full_stats=True)
            (FI_rc_curv[i, j, 0], FI_rc_curv[i, j, 1]) = (fi_curv_dict['mean'], fi_curv_dict['std'])
            print FI_rc_curv
            
            print "theoretical FI"

            FI_rc_theo[i, j, 0] = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=stat_meas.model_parameters['covariances'][-1, 0])
            FI_rc_theo[i, j, 1] = random_network.compute_fisher_information_theoretical(sigma=all_parameters['sigmax'], kappa1=all_parameters['rc_scale'], kappa2=all_parameters['rc_scale'])
            print FI_rc_theo

            print "from precision of recall..."
            sampler.sample_theta(num_samples=all_parameters['num_samples'], burn_samples=100, selection_method=all_parameters['selection_method'], selection_num_samples=all_parameters['num_samples'], integrate_tc_out=False, debug=False)
            FI_rc_precision[i, j] = sampler.get_precision()
            print FI_rc_precision

            dataio.save_variables(variables_to_save, locals())

    # Plots
    data_to_plot = {}
    for curr_data in variables_to_save:
        data_to_plot[curr_data] = locals()[curr_data]

    plots_fisher_info_param_search(data_to_plot, dataio)

    return locals()


def plots_fisher_info_param_search(data_to_plot, dataio):
    '''
        Create and save a few plots for the fisher information parameter search
    '''
    
    # Sanity check, verify that we have all the data we will be plotting
    required_variables = ['rcscale_space', 'sigma_space', 'FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo']

    assert set.intersection(set(data_to_plot), set(required_variables)) == set(required_variables), "This dataset is not complete. Require %s" % required_variables

    # 2D plots
    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(data_to_plot['FI_rc_curv'][:, :, 0].T, interpolation='nearest', origin='lower left')
    ax.set_yticks(np.arange(data_to_plot['rcscale_space'].size))
    ax.set_yticklabels(data_to_plot['rcscale_space'])
    ax.set_xticks(np.arange(data_to_plot['sigma_space'].size))
    ax.set_xticklabels(data_to_plot['sigma_space'])
    f.colorbar(im)
    plt.title('FI from curvature, sigma/rcscale')
    ax.axis('tight')
    dataio.save_current_figure("FI_paramsearch_2d_curve_{unique_id}.pdf")

    
    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(data_to_plot['FI_rc_precision'].T, interpolation='nearest', origin='lower left')
    ax.set_yticks(np.arange(data_to_plot['rcscale_space'].size))
    ax.set_yticklabels(data_to_plot['rcscale_space'])
    ax.set_xticks(np.arange(data_to_plot['sigma_space'].size))
    ax.set_xticklabels(data_to_plot['sigma_space'])
    f.colorbar(im)
    plt.title('FI Recall precision, sigma/rcscale')
    ax.axis('tight')
    dataio.save_current_figure("FI_paramsearch_2d_precision_{unique_id}.pdf")

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(data_to_plot['FI_rc_theo'][:, :, 0].T, interpolation='nearest', origin='lower left')
    ax.set_yticks(np.arange(data_to_plot['rcscale_space'].size))
    ax.set_yticklabels(data_to_plot['rcscale_space'])
    ax.set_xticks(np.arange(data_to_plot['sigma_space'].size))
    ax.set_xticklabels(data_to_plot['sigma_space'])
    f.colorbar(im)
    plt.title('FI Theoretical sum, sigma/rcscale')
    ax.axis('tight')
    dataio.save_current_figure("FI_paramsearch_2d_theo_{unique_id}.pdf")

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(data_to_plot['FI_rc_theo'][:, :, 1].T, interpolation='nearest', origin='lower left')
    ax.set_yticks(np.arange(data_to_plot['rcscale_space'].size))
    ax.set_yticklabels(data_to_plot['rcscale_space'])
    ax.set_xticks(np.arange(data_to_plot['sigma_space'].size))
    ax.set_xticklabels(data_to_plot['sigma_space'])
    f.colorbar(im)
    plt.title('FI Theoretical large N, sigma/rcscale')
    ax.axis('tight')
    dataio.save_current_figure("FI_paramsearch_2d_theoN_{unique_id}.pdf")

    # 1D plots
    for i, sigma in enumerate(data_to_plot['sigma_space']):
        ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_curv'][:, i, 0], data_to_plot['FI_rc_curv'][:, i, 1])
        ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_precision'][:, i], 0.0*data_to_plot['FI_rc_precision'][:, i], ax_handle=ax)
        ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][:, i, 0], 0.0*data_to_plot['FI_rc_theo'][:, i, 0], ax_handle=ax)
        ax = plot_mean_std_area(data_to_plot['rcscale_space'], data_to_plot['FI_rc_theo'][:, i, 1], 0.0*data_to_plot['FI_rc_theo'][:, i, 1], ax_handle=ax)

        plt.title('FI dependence on rcscale, sigma %.2f' % sigma)
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Recall precision', 'Theoretical sum', 'Theoretical large N'])

        dataio.save_current_figure("FI_paramsearch_rcscale_sigmafixed_mean_std_{unique_id}.pdf")















