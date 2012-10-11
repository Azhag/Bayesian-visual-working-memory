#!/usr/bin/env python
# encoding: utf-8
"""
launchers.py


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


def do_simple_run(args):
    
    print "Simple run"
    
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


    # Build the random network
    # sigma_y = 0.02
    # sigma_y = 0.2
    # sigma_x = 0.1
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = T-1

    # 'conj', 'feat', 'mixed'
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
    print "Building the database"
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation='constant')
    
    # Measure the noise structure
    print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation='constant')
    stat_meas = StatisticsMeasurer(data_gen_noise)
    # stat_meas = StatisticsMeasurer(data_gen)
    
    print "Sampling..."
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
    
    print "Inferring optimal angles, for t=%d" % sampler.tc[0]
    # sampler.set_theta_max_likelihood(num_points=500, post_optimise=True)
    
    if args.inference_method == 'sample':
        # Sample thetas
        sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
    elif args.inference_method == 'max_lik':
        # Just use the ML value for the theta
        sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
    elif args.inference_method == 'none':
        # Do nothing
        pass
        
    sampler.print_comparison_inferred_groundtruth()
    
    return locals()
    


def do_neuron_number_precision(args):
    '''
        Check the effect of the number of neurons on the coding precision.
    '''
    N = args.N
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    code_type = args.code_type
    output_dir = os.path.join(args.output_directory, args.label)
    
    R = 2
    param1_space = np.array([10, 20, 50, 100, 150, 200, 300, 500, 1000, 1500])
    # param1_space = np.array([10, 50, 100, 300])
    # param1_space = np.array([300])
    # num_repetitions = 5
    # num_repetitions = 3

    # After searching the big N/scale space, get some relation for scale = f(N)
    fitted_scale_space = np.array([4.0, 4.0, 1.5, 1.0, 0.8, 0.55, 0.45, 0.35, 0.2, 0.2])
    # fitted_scale_space = np.array([0.4])
    
    output_string = unique_filename(prefix=strcat(output_dir, 'neuron_number_precision'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.2
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform')  # alpha 0.8
    cued_feature_time = T-1

    print "Doing do_neuron_number_precision"
    print "param1_space: %s" % param1_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing M=%d, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            if code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(fitted_scale_space[param1_i], 0.001), ratio_moments=(1.0, 0.2))
            elif code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=fitted_scale_space[param1_i])
            elif code_type == 'mixed':
                random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)

            # random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(0.4, 0.02), ratio_moments=(1.0, 0.05))
            # random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(fitted_scale_space[param1_i], 0.001), ratio_moments=(1.0, 0.2))
            # random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=0.3)
            # random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'fitted_scale_space': fitted_scale_space, 'output_string': output_string})
            
    # Plot
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]')
    
    print all_precisions

    print "Done: %s" % output_string

    return locals()


def plot_neuron_number_precision(args):
    '''
        Plot from results of a do_neuron_number_precision
    '''
    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_neuron_number_precision"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    if 'M_space' in loaded_data:
        param1_space = loaded_data['M_space']
    elif 'param1_space' in loaded_data:
        param1_space = loaded_data['param1_space']

    
    # Do the plot(s)
    f = plt.figure()
    ax = f.add_subplot(111)
    if np.any(np.std(all_precisions, 1) == 0.0):
        plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
        ax.set_ylabel('Std dev [rad]')
    else:
        plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
        ax.set_ylabel('Precision [rad]^-1')
    ax.set_xlabel('Number of neurons in population')
    

    return locals()

    # For multiple plot, run this, then save all_precisions, and redo same for other parameters.
    #
    # all_precisions1 = all_vars['all_precisions']
    # all_precisions2 = all_vars['all_precisions']
    #
    # ax = semilogy_mean_std_area(param1_space1, np.mean(all_precisions1, 1), np.std(all_precisions1, 1))
    # ax = semilogy_mean_std_area(param1_space1, np.mean(all_precisions2, 1), np.std(all_precisions2, 1), ax_handle=ax)
    # legend(['Features', 'Conjunctive'])


def plot_multiple_neuron_number_precision(args):
    input_filenames = ['Data/Used/feat_new_neuron_number_precision-5067b28c-0fd1-4586-a1be-1a5ab0a820f4.npy', 'Data/Used/conj_good_neuron_number_precision-4da143e7-bbd4-432d-8603-195348dd7afa.npy']

    f = plt.figure()
    ax = f.add_subplot(111)

    for file_n in input_filenames:
        loaded_data = np.load(file_n).item()
        loaded_precision = loaded_data['all_precisions']
        if 'M_space' in loaded_data:
            param1_space = loaded_data['M_space']
        elif 'param1_space' in loaded_data:
            param1_space = loaded_data['param1_space']

        ax = semilogy_mean_std_area(param1_space, np.mean(loaded_precision, 1), np.std(loaded_precision, 1), ax_handle=ax)
    

def do_size_receptive_field(args):
    '''
        Check the effect of the size of the receptive fields of neurons on the coding precision.
    '''

    N = args.N
    M = args.M
    T = args.T
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    code_type = args.code_type
    rc_scale = args.rc_scale
    
    output_dir = os.path.join(args.output_directory, args.label)
    
    R = 2

    # param1_space = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5])
    param1_space = np.logspace(np.log10(0.05), np.log10(1.5), 12)

    output_string = unique_filename(prefix=strcat(output_dir, 'size_receptive_field'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.2
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')  # alpha 0.5
    cued_feature_time = T-1

    print "Doing do_size_receptive_field"
    print "Scale_space: %s" % param1_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing Scale=%.2f, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R,  scale_moments=(param1_space[param1_i], 0.001), ratio_moments=(1.0, 0.1))
            # random_network = RandomFactorialNetwork.create_full_features(M_space[param1_i], R=R)
            # random_network = RandomFactorialNetwork.create_mixed(M_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            
            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Size of receptive fields')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string

    return locals()


def plot_size_receptive_field(args):
    '''
        Plot from results of a size_receptive_field
    '''

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from size_receptive_field"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    param1_space = loaded_data['param1_space']

    # Do the plot(s)
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]^-1')

    return locals()

    # For multiple plot, run this, then save all_precisions, and redo same for other parameters.
    #
    # all_precisions1 = all_vars['all_precisions']
    # all_precisions2 = all_vars['all_precisions']
    #
    # ax = semilogy_mean_std_area(M_space1, np.mean(all_precisions1, 1), np.std(all_precisions1, 1))
    # ax = semilogy_mean_std_area(M_space1, np.mean(all_precisions2, 1), np.std(all_precisions2, 1), ax_handle=ax)
    # legend(['Features', 'Conjunctive'])


def do_size_receptive_field_number_neurons(args):
    '''
        Check the effect of the size of the receptive fields of neurons on the coding precision
        Also change the number of neurons.
    '''


    N = args.N
    T = args.T
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    output_dir = os.path.join(args.output_directory, args.label)
    num_repetitions = args.num_repetitions
    code_type = args.code_type

    output_string = unique_filename(prefix=strcat(output_dir, 'size_receptive_field_number_neurons'))
    R = 2

    param1_space = np.array([10, 20, 50, 75, 100, 200, 300, 500, 1000])
    # param2_space = np.array([0.05, 0.1, 0.15, 0.17, 0.2, 0.25, 0.3, 0.5, 1.0, 1.5])
    # param2_space = np.array([0.05, 0.1, 0.15, 0.2, 0.5, 1.0])
    param2_space = np.logspace(np.log10(0.05), np.log10(4.0), 12)

    # num_repetitions = 10

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_size_receptive_field_number_neurons"
    print "M_space: %s" % param1_space
    print "Scale_space: %s" % param2_space
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for param2_i in np.arange(param2_space.size):
            for repet_i in np.arange(num_repetitions):
                #### Get multiple examples of precisions, for different number of neurons. #####
                print "Doing M=%d, Scale=%.2f, %d/%d" % (param1_space[param1_i], param2_space[param2_i], repet_i+1, num_repetitions)

                if code_type == 'conj':
                    random_network = RandomFactorialNetwork.create_full_conjunctive(param1_space[param1_i], R=R, scale_moments=(param2_space[param2_i], 0.001), ratio_moments=(1.0, 0.2))
                elif code_type == 'feat':
                    random_network = RandomFactorialNetwork.create_full_features(param1_space[param1_i], R=R, scale=param2_space[param2_i])
                elif code_type == 'mixed':
                    random_network = RandomFactorialNetwork.create_mixed(param1_space[param1_i], R=R, ratio_feature_conjunctive=0.2)
                
                # Construct the real dataset
                data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                
                # Measure the noise structure
                data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                stat_meas = StatisticsMeasurer(data_gen_noise)
                # stat_meas = StatisticsMeasurer(data_gen)
                
                sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
                
                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                
                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[param1_i, param2_i, repet_i]

            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'param2_space': param2_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})

            
    
    
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), param1_space)
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Scale of receptive field')
    f.colorbar(im)

    # f = plt.figure()
    # ax = f.add_subplot(111)
    # plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    # ax.set_xlabel('Number of neurons in population')
    # ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string

    return locals()


def plot_size_receptive_field_number_neurons(args):

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_size_receptive_field_number_neurons"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    param1_space = loaded_data['param1_space']
    param2_space = loaded_data['param2_space']

    ### Simple imshow
    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), param1_space)
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Scale of receptive field')
    f.colorbar(im)

    ### Fill a nicer plot, interpolating between the sampled points
    param1_space_int = np.linspace(param1_space.min(), param1_space.max(), 100)
    param2_space_int = np.linspace(param2_space.min(), param2_space.max(), 100)

    all_points = np.array(cross(param1_space, param2_space))
    all_precisions_flat = 1./np.mean(all_precisions, 2).flatten()

    all_precisions_int = spint.griddata(all_points, all_precisions_flat, (param1_space_int[None, :], param2_space_int[:, None]), method='nearest')

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    cs = ax1.contourf(param1_space_int, param2_space_int, all_precisions_int, 20)   # cmap=plt.cm.jet
    ax1.set_xlabel('Number of neurons')
    ax1.set_ylabel('Scale of receptive field')
    ax1.set_title('Precision wrt scale/number of neurons')
    ax1.scatter(all_points[:, 0], all_points[:, 1], marker='o', c='b', s=5)
    ax1.set_xlim(param1_space_int.min(), param1_space_int.max())
    ax1.set_ylim(param2_space_int.min(), param2_space_int.max())
    f1.colorbar(cs)

    ### Print the 1D plot for each N
    
    for i in np.arange(param1_space.size):
        f = plt.figure()
        ax = f.add_subplot(111)
        plot_mean_std_area(param2_space, np.mean(1./all_precisions[i], 1), np.std(1./all_precisions[i], 1), ax_handle=ax)
        ax.set_xlabel('Scale of filter')
        ax.set_ylabel('Precision [rad]^-1')
        ax.set_title('Optim scale for N=%d' % param1_space[i])
    # plot_square_grid(np.tile(param2_space, (param1_space.size, 1)), np.mean(1./all_precisions, 2))

    ### Plot the optimal scale for all N
    optimal_scale_N = param2_space[np.argmax(np.mean(1./all_precisions, 2), 1)]
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.plot(param1_space, optimal_scale_N)
    ax2.set_xlabel('Number of neurons')
    ax2.set_ylabel('Optimal scale')

    return locals()

    
def do_memory_curve(args):
    '''
        Get the memory curve
    '''

    
    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.ones((args.T, args.num_repetitions))*np.nan
    for repet_i in np.arange(args.num_repetitions):
        #### Get multiple examples of precisions, for different number of neurons. #####
        
        if args.code_type == 'conj':
            random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
        elif args.code_type == 'feat':
            random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
        elif args.code_type == 'mixed':
            conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
            feat_params = dict(scale=args.rc_scale2, ratio=40.)
            random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
        else:
            raise ValueError('Code_type is wrong!')

        
        # Construct the real dataset
        data_gen = DataGeneratorRFN(args.N, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
        
        # Measure the noise structure
        data_gen_noise = DataGeneratorRFN(3000, args.T, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
        stat_meas = StatisticsMeasurer(data_gen_noise)
        # stat_meas = StatisticsMeasurer(data_gen)
        
        sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters)
        
        for t in np.arange(args.T):
            print "Doing T=%d,  %d/%d" % (t, repet_i+1, args.num_repetitions)

            # Change the cued feature
            sampler.change_cued_features(t)

            # Cheat here, just use the ML value for the theta
            # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

            # Sample the new theta
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.selection_num_samples, integrate_tc_out=False, debug=False)
            
            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=True)
            # all_precisions[t, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]
    
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Object')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def do_multiple_memory_curve(args):
    '''
        Get the memory curves, for 1...T objects
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'multiple_memory_curve'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))

    # Construct different datasets, with t objects
    for t in np.arange(args.T):

        for repet_i in np.arange(args.num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)
                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters)
            
            for tc in np.arange(t+1):
                print "Doing T=%d, Tc=%d,  %d/%d" % (t+1, tc, repet_i+1, args.num_repetitions)

                # Change the cued feature
                sampler.change_cued_features(tc)

                # Cheat here, just use the ML value for the theta
                # sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                # sampler.set_theta_max_likelihood_tc_integratedout(num_points=200, post_optimise=True)

                # Sample thetas
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=20, selection_method='median', selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)

                # Save the precision
                all_precisions[t, tc, repet_i] = sampler.get_precision(remove_chance_level=True)
                # all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'output_string': output_string})

    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        plot_mean_std_area(t_space_aligned_right, np.mean(all_precisions[t], 1)[:t+1], np.std(all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def do_multiple_memory_curve_simult(args):
    '''
        Get the memory curves, for 1...T objects, simultaneous presentation
        (will force alpha=1, and only recall one object for each T, more independent)
    '''

    # Build the random network
    alpha = 1.
    time_weights_parameters = dict(weighting_alpha=alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    # Initialise the output file
    dataio = DataIO(output_folder=args.output_directory, label=args.label)
    output_string = dataio.filename

    # List of variables to save
    # Try not save when one of those is not set. 
    variables_to_output = ['all_precisions', 'args', 'num_repetitions', 'output_string', 'power_law_params', 'repet_i']

    print "Doing do_multiple_memory_curve"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.num_repetitions))
    power_law_params = np.zeros(2)

    # Construct different datasets, with t objects
    for repet_i in np.arange(args.num_repetitions):

        for t in np.arange(args.T):

            #### Get multiple examples of precisions, for different number of neurons. #####
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.0001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)
                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=t, stimuli_generation=args.stimuli_generation)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters, cued_feature_time=t, stimuli_generation=args.stimuli_generation)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=t)
            
            print "  doing T=%d %d/%d" % (t+1, repet_i+1, args.num_repetitions)

            if args.inference_method == 'sample':
                # Sample thetas
                sampler.sample_theta(num_samples=args.num_samples, burn_samples=100, selection_method=args.selection_method, selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
            elif args.inference_method == 'max_lik':
                # Just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            else:
                raise ValueError('Wrong value for inference_method')

            # Save the precision
            all_precisions[t, repet_i] = sampler.get_precision(remove_chance_level=False)
            # all_precisions[t, repet_i] = 1./sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[t, repet_i]
            
            # Save to disk, unique filename
            dataio.save_variables(variables_to_output, locals())

            # np.save(output_string, {'all_precisions': all_precisions, 'args': args, 'num_repetitions': args.num_repetitions, 'output_string': output_string, 'power_law_params': power_law_params, 'repet_i': repet_i})
        
        xx = np.tile(np.arange(1, args.T+1, dtype='float'), (repet_i+1, 1)).T
        power_law_params = fit_powerlaw(xx, all_precisions[:, :(repet_i+1)], should_plot=True)

        print '====> Power law fits: exponent: %.4f, bias: %.4f' % (power_law_params[0], power_law_params[1])

        # Save to disk, unique filename
        dataio.save_variables(variables_to_output, locals())

    print all_precisions

    
    # Save to disk, unique filename
    dataio.save_variables(variables_to_output, locals())
    

    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(np.arange(args.T), np.mean(all_precisions, 1), np.std(all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of objects')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def plot_multiple_memory_curve(args):

    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(T):
        t_space_aligned_right = (T - np.arange(t+1))[::-1]
        # plot_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
        # plt.semilogy(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-', linewidth=2, markersize=8)
        plt.plot(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], 'o-')
    x_labels = ['-%d' % x for x in np.arange(T)[::-1]]
    x_labels[-1] = 'Last'

    ax.set_xticks(t_space_aligned_right)
    ax.set_xticklabels(x_labels)
    ax.set_xlim((0.8, T+0.2))
    ax.set_ylim((1.01, 40))
    # ax.set_xlabel('Recall time')
    # ax.set_ylabel('Precision [rad]')
    legends=['%d items' % (x+1) for x in np.arange(T)]
    legends[0] = '1 item'
    plt.legend(legends, loc='best', numpoints=1, fancybox=True)

    return locals()




def plot_multiple_memory_curve_simult(args):
    input_filename = args.input_filename

    assert input_filename is not '', "Give a file with saved results from do_multiple_memory_curve"

    loaded_data = np.load(input_filename).item()

    all_precisions = loaded_data['all_precisions']
    T = loaded_data['args'].T
    
    # Average over repetitions, and then get mean across T
    # mean_precision = np.zeros(T)
    # std_precision = np.zeros(T)
    # for t in np.arange(T):
        # mean_precision[t] = np.mean(all_precisions[t][:t+1])
        # std_precision[t] = np.std(all_precisions[t][:t+1])

    mean_precision = np.mean(all_precisions, axis=1)
    std_precision = np.std(all_precisions, axis=1)

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.semilogy(np.arange(1, T+1), mean_precision, 'o-')
    plt.xticks(np.arange(1, T+1))
    plt.xlim((0.9, T+0.1))
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')

    f = plt.figure()
    ax = f.add_subplot(111)
    plt.plot(np.arange(1, T+1), mean_precision)
    ax.set_xlabel('Number of stored items')
    ax.set_ylabel('Precision [rad]')

    plot_mean_std_area(np.arange(1, T+1), mean_precision, std_precision)
    plt.xlabel('Number of stored items')
    plt.ylabel('Precision [rad]^-0.5')


    return locals()


def do_mixed_ratioconj(args):
    '''
        For a mixed population, check the effect of increasing the ratio of conjunctive cells
        on the performance.
    '''

    N = args.N
    M = args.M
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    output_dir = os.path.join(args.output_directory, args.label)
    rc_scale = args.rc_scale
    rc_scale2 = args.rc_scale2

    R = 2
    args.R = 2
    args.code_type = 'mixed'
    param1_space = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
    
    output_string = unique_filename(prefix=strcat(output_dir, 'mixed_ratioconj'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_mixed_ratioconj"
    print "param1_space: %s" % param1_space
    print "rc_scales: %.3f %.3f" % (rc_scale, rc_scale2)
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for repet_i in np.arange(num_repetitions):
            #### Get multiple examples of precisions, for different number of neurons. #####
            print "Doing M=%.3f, %d/%d" % (param1_space[param1_i], repet_i+1, num_repetitions)

            # Construct the network with appropriate parameters
            conj_params = dict(scale_moments=(rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
            feat_params = dict(scale=rc_scale2, ratio=40.)

            random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=param1_space[param1_i], conjunctive_parameters=conj_params, feature_parameters=feat_params)
            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
            
            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
            
            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()[1]

            print "-> %.5f" % all_precisions[param1_i, repet_i]
        
        # Save to disk, unique filename
        np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})
            
    # Plot
    f = plt.figure()
    ax = f.add_subplot(111)
    plot_mean_std_area(param1_space, np.mean(1./all_precisions, 1), np.std(1./all_precisions, 1), ax_handle=ax)
    ax.set_xlabel('Number of neurons in population')
    ax.set_ylabel('Precision [rad]')
    
    print all_precisions

    print "Done: %s" % output_string

    return locals()


def do_mixed_two_scales(args):
    '''
        Search over the space of conjunctive scale and feature scale.
        Should be called with different values of conj_ratio (e.g. from PBS)
    '''

    N = args.N
    M = args.M
    # K = args.K
    num_samples = args.num_samples
    weighting_alpha = args.alpha
    num_repetitions = args.num_repetitions
    T = args.T
    output_dir = os.path.join(args.output_directory, args.label)
    # rc_scale = args.rc_scale
    # rc_scale2 = args.rc_scale2
    ratio_conj = args.ratio_conj

    R = 2
    args.R = 2
    args.code_type = 'mixed'
    param1_space = np.logspace(np.log10(0.05), np.log10(4.0), num_samples)
    param2_space = np.logspace(np.log10(0.05), np.log10(4.0), num_samples)
    
    output_string = unique_filename(prefix=strcat(output_dir, 'mixed_two_scales'))

    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.01
    time_weights_parameters = dict(weighting_alpha=weighting_alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = T-1

    print "Doing do_mixed_ratioconj"
    print "param1_space: %s" % param1_space
    print "param2_space: %s" % param2_space
    print "ratio_conj: %.3f" % (ratio_conj)
    print "File: %s" % output_string

    all_precisions = np.zeros((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in np.arange(param1_space.size):
        for param2_i in np.arange(param2_space.size):
            for repet_i in np.arange(num_repetitions):
                #### Get multiple examples of precisions, for different number of neurons. #####
                print "Doing scales=(%.3f, %.3f), %d/%d" % (param1_space[param1_i], param2_space[param2_i], repet_i+1, num_repetitions)

                # Construct the network with appropriate parameters
                conj_params = dict(scale_moments=(param1_space[param1_i], 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=param2_space[param2_i], ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
                
                # Construct the real dataset
                data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                
                # Measure the noise structure
                data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, sigma_x=sigma_x, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
                stat_meas = StatisticsMeasurer(data_gen_noise)
                # stat_meas = StatisticsMeasurer(data_gen)
                
                sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)
                
                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)
                
                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[param1_i, param2_i, repet_i]
            
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'param1_space': param1_space, 'param2_space': param2_space, 'args': args, 'num_repetitions': num_repetitions, 'output_string': output_string})
          
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(np.mean(1./all_precisions, 2).T, origin='lower', aspect='auto')
    # im.set_extent((param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
    im.set_interpolation('nearest')
    # ax.xaxis.set_major_locator(plttic.NullLocator())
    # ax.yaxis.set_major_locator(plttic.NullLocator())
    plt.xticks(np.arange(param1_space.size), np.around(param1_space, 2))
    plt.yticks(np.arange(param2_space.size), np.around(param2_space, 2))
    ax.set_xlabel('Scale of conjunctive neurons')
    ax.set_ylabel('Scale of feature neurons')
    f.colorbar(im)
    
    print "Done: %s" % output_string

    return locals()


def do_save_responses_simultaneous(args):
    '''
        Simulate simultaneous presentations, with 1...T objects.
        Outputs the responses and target/non-targets, to be fitted in Matlab (TODO: convert EM fit in Python code)
    '''

    output_dir = os.path.join(args.output_directory, args.label)
    output_string = unique_filename(prefix=strcat(output_dir, 'save_responses_simultaneous'))

    # Build the random network
    time_weights_parameters = dict(weighting_alpha=args.alpha, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')

    print "Doing do_save_responses_simultaneous"
    print "max_T: %s" % args.T
    print "File: %s" % output_string

    all_precisions = np.zeros((args.T, args.T, args.num_repetitions))
    all_responses = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_targets = np.zeros((args.T, args.T, args.num_repetitions, args.N))
    all_nontargets = np.zeros((args.T, args.T, args.num_repetitions, args.N, args.T-1))
    
    for repet_i in np.arange(args.num_repetitions):
        
        # Construct different datasets, with t objects
        for t in np.arange(args.T):
            
            if args.code_type == 'conj':
                random_network = RandomFactorialNetwork.create_full_conjunctive(args.M, R=args.R, scale_moments=(args.rc_scale, 0.0001), ratio_moments=(1.0, 0.001))
            elif args.code_type == 'feat':
                random_network = RandomFactorialNetwork.create_full_features(args.M, R=args.R, scale=args.rc_scale, ratio=40.)
            elif args.code_type == 'mixed':
                conj_params = dict(scale_moments=(args.rc_scale, 0.001), ratio_moments=(1.0, 0.0001))
                feat_params = dict(scale=args.rc_scale2, ratio=40.)

                random_network = RandomFactorialNetwork.create_mixed(args.M, R=args.R, ratio_feature_conjunctive=args.ratio_conj, conjunctive_parameters=conj_params, feature_parameters=feat_params)
            else:
                raise ValueError('Code_type is wrong!')

            
            # Construct the real dataset
            data_gen = DataGeneratorRFN(args.N, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            
            # Measure the noise structure
            data_gen_noise = DataGeneratorRFN(3000, t+1, random_network, sigma_y=args.sigmay, sigma_x=args.sigmax, time_weights_parameters=time_weights_parameters)
            stat_meas = StatisticsMeasurer(data_gen_noise)
            # stat_meas = StatisticsMeasurer(data_gen)
            
            sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters)
            
            for tc in np.arange(t+1):
                print "Doing T=%d, Tc=%d,  %d/%d" % (t+1, tc, repet_i+1, args.num_repetitions)

                # Change the cued feature
                sampler.change_cued_features(tc)

                # Sample the new theta
                sampler.sample_theta(num_samples=args.num_samples, selection_num_samples=args.selection_num_samples, integrate_tc_out=False, selection_method='median')

                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[t, tc, repet_i] = sampler.compute_angle_error()[1]

                print "-> %.5f" % all_precisions[t, tc, repet_i]

                # Save the responses, targets and nontargets
                print t
                print tc
                (all_responses[t, tc, repet_i], all_targets[t, tc, repet_i], all_nontargets[t, tc, repet_i, :, :t])=sampler.collect_responses()

        
            # Save to disk, unique filename
            np.save(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})
            sio.savemat(output_string, {'all_precisions': all_precisions, 'all_responses': all_responses, 'all_targets': all_targets, 'all_nontargets': all_nontargets, 'args': args, 'num_repetitions': args.num_repetitions, 'T': args.T, 'output_string': output_string})

    
    print all_precisions

    f = plt.figure()
    ax = f.add_subplot(111)
    for t in np.arange(args.T):
        t_space_aligned_right = (args.T - np.arange(t+1))[::-1]
        semilogy_mean_std_area(t_space_aligned_right, np.mean(1./all_precisions[t], 1)[:t+1], np.std(1./all_precisions[t], 1)[:t+1], ax_handle=ax)
    ax.set_xlabel('Recall time')
    ax.set_ylabel('Precision [rad]')
    
    print "Done: %s" % output_string
    return locals()


def do_fisher_information_estimation(args):
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

        # rcscale_space = np.linspace(1.5, 6.0, 5)
        rcscale_space = np.linspace(4., 4., 1.)
        FI_rc_curv = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_curv_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_curv_all = []
        FI_rc_samples = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_samples_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_samples_all = []
        FI_rc_precision = np.zeros(rcscale_space.size, dtype=float)
        FI_rc_precision_quantiles = np.zeros((rcscale_space.size, 3), dtype=float)
        FI_rc_precision_all = []

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
            
            if True:
                print "from samples..."
                
                if single_point_estimate:
                    prec_samples_dict =  sampler.estimate_precision_from_samples(num_samples=500, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1])=(prec_samples_dict['mean'], prec_samples_dict['std'])
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])
                else:
                    prec_samples_dict = sampler.estimate_precision_from_samples_avg(num_samples=500, full_stats=True, num_repetitions=num_repet_sample_estimate, selection_method=selection_method)
                    (FI_rc_samples[i, 0], FI_rc_samples[i, 1], FI_rc_samples[i, 2])=(prec_samples_dict['median'], prec_samples_dict['std'], np.max(prec_samples_dict['all']))
                    FI_rc_samples_quantiles[i] = spst.mstats.mquantiles(prec_samples_dict['all'])

                    FI_rc_samples_all.append(prec_samples_dict['all'])


            print FI_rc_samples[i]
            print FI_rc_samples_quantiles[i]
            
            print "from precision of recall..."
            sampler.sample_theta(num_samples=args.num_samples, burn_samples=100, selection_method=selection_method, selection_num_samples=args.num_samples, integrate_tc_out=False, debug=False)
            FI_rc_precision[i] = sampler.get_precision()
            FI_rc_precision_quantiles[i] = spst.mstats.mquantiles(FI_rc_precision[i])
            FI_rc_precision_all.append(FI_rc_precision[i])

            print FI_rc_precision[i]

        FI_rc_curv_all = np.array(FI_rc_curv_all)
        FI_rc_samples_all = np.array(FI_rc_samples_all)
        FI_rc_precision_all = np.array(FI_rc_precision_all)


        # Save the results
        dataio.save_variables(['FI_rc_curv', 'FI_rc_samples', 'FI_rc_precision', 'FI_rc_curv_quantiles', 'FI_rc_samples_quantiles', 'FI_rc_precision_quantiles', 'rcscale_space', 'FI_rc_curv_all', 'FI_rc_samples_all', 'FI_rc_precision_all'], locals())

        # Plot results
        ax = plot_mean_std_area(rcscale_space, FI_rc_curv[:, 0], FI_rc_curv[:, 1])
        plot_mean_std_area(rcscale_space, FI_rc_samples[:, 0], FI_rc_samples[:, 1], ax_handle=ax)
        # ax = plot_mean_std_area(rcscale_space, FI_rc_samples[:, 2], 0.0*FI_rc_samples[:, 1], ax_handle=ax)
        ax = plot_mean_std_area(rcscale_space, FI_rc_precision, 0.0*FI_rc_precision, ax_handle=ax)

        plt.title('FI dependence on rcscale')
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision'])

        dataio.save_current_figure("FI_rcscale_comparison_mean_std_{unique_id}.pdf")

        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_curv_quantiles)
        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_samples_quantiles, ax_handle=ax2)
        ax2 = plot_median_quantile_area(rcscale_space, quantiles=FI_rc_precision_quantiles, ax_handle=ax2)

        plt.title('FI dependence, quantiles shown')
        plt.xlabel('rcscale')
        plt.ylabel('FI')
        plt.legend(['Curvature', 'Samples', 'Recall precision'])

        dataio.save_current_figure('FI_rcscale_comparison_median_std_{unique_id}.pdf')

        if not single_point_estimate:

            for rc_scale_i, rc_scale in enumerate(rcscale_space):
                # Show the precision from posterior estimate against the FI from posterior estimate
                plt.figure()

                plt.plot(FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i], 'x')

                idx = np.linspace(FI_rc_curv_all[rc_scale_i].min()*0.95, FI_rc_curv_all[rc_scale_i].max()*1.05, 100.)

                plt.plot(idx, idx, ':k')
                plt.axis('tight')
                plt.xlabel('Curvature estimate')
                plt.ylabel('Samples estimate')
                plt.title('Comparison Curvature vs samples estimate of FI. Rscale: %d' % rc_scale)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_allpoints_%d-{unique_id}.pdf' % rc_scale)

                # Show the boxplot of each estimate, per number of samples
                plt.figure()
                plt.boxplot([FI_rc_curv_all[rc_scale_i], FI_rc_samples_all[rc_scale_i].flatten(), FI_rc_precision_all[rc_scale_i]])
                plt.title('Comparison Curvature vs samples estimate. Rscale: %d' % rc_scale)
                plt.xticks([1, 2, 3], ['Curvature', 'Samples', 'Precision'], rotation=45)

                dataio.save_current_figure('FI_rc_comparison_curv_samples_%d-{unique_id}.pdf' % rc_scale)
    
    else:
        raise ValueError('Wrong subaction!')


    return locals()

