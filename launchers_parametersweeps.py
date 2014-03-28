#!/usr/bin/env python
# encoding: utf-8
"""
launchers_parametersweeps.py


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



def launcher_do_neuron_number_precision(args):
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

    all_precisions = np.nan*np.empty((param1_space.size, num_repetitions))
    for param1_i in xrange(param1_space.size):
        for repet_i in xrange(num_repetitions):
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

            sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)

            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()['std']

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


def launcher_plot_neuron_number_precision(args):
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


def launcher_plot_multiple_neuron_number_precision(args):
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


def launcher_do_size_receptive_field(args):
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

    all_precisions = np.nan*np.empty((param1_space.size, num_repetitions))
    for param1_i in xrange(param1_space.size):
        for repet_i in xrange(num_repetitions):
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

            sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)

            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()['std']


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


def launcher_plot_size_receptive_field(args):
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


def launcher_do_size_receptive_field_number_neurons(args):
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

    all_precisions = np.nan*np.empty((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in xrange(param1_space.size):
        for param2_i in xrange(param2_space.size):
            for repet_i in xrange(num_repetitions):
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

                sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)

                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()['std']

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


def launcher_plot_size_receptive_field_number_neurons(args):

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

    for i in xrange(param1_space.size):
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




def launcher_do_mixed_ratioconj(args):
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

    all_precisions = np.nan*np.empty((param1_space.size, num_repetitions))
    for param1_i in xrange(param1_space.size):
        for repet_i in xrange(num_repetitions):
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

            sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

            # Cheat here, just use the ML value for the theta
            sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)

            # Save the precision
            # all_precisions[param1_i, repet_i] = sampler.get_precision()
            all_precisions[param1_i, repet_i] = sampler.compute_angle_error()['std']

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


def launcher_do_mixed_two_scales(args):
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

    all_precisions = np.nan*np.empty((param1_space.size, param2_space.size, num_repetitions))
    for param1_i in xrange(param1_space.size):
        for param2_i in xrange(param2_space.size):
            for repet_i in xrange(num_repetitions):
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

                sampler = Sampler(data_gen, n_parameters=stat_meas.model_parameters, tc=cued_feature_time, sigma_output=0.0)

                # Cheat here, just use the ML value for the theta
                sampler.set_theta_max_likelihood(num_points=200, post_optimise=True)

                # Save the precision
                # all_precisions[param1_i, repet_i] = sampler.get_precision()
                all_precisions[param1_i, param2_i, repet_i] = sampler.compute_angle_error()['std']

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


