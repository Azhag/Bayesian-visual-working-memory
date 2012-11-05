#!/usr/bin/env python
# encoding: utf-8
"""
launchers_experimentalvolume.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import os

import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from datapbs import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from submitpbs import *
import progress

PBS_SUBMIT=True

def launcher_do_generate_constrained_param_experimental_theo(args):
    '''
        Generate parameters consistent with the experimentally found FI.
        Uses the theoretical FI for our given population code to verify constraint.

        Should then spawn simulations, or find a way to use those appropriately.
    '''

    all_parameters = vars(args)
    
    variables_to_save = ['dict_parameters_range', 'constrained_parameters', 'pbs_submission_infos']
    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    num_samples = all_parameters['num_samples']

    # Experimental FI informations
    experimental_fi = 36.94
    max_mse_fi = 100.0

    # PBS submission informations
    pbs_submit_during_parameters_generation = True
    
    # pbs_submission_infos = dict(description='Testing the random parameter generator.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_fisher_information_param_search_pbs', output_directory='.', M=400, sigmax=0.2, rc_scale=4.0, N=300, T=1, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', num_repetitions=all_parameters['num_repetitions'], label='allfi_randomparams_M400N300samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'random_search_fi'))

    pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=225, sigmax=0.2, rc_scale=4.0, N=200, T=6, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M225N200samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained'))
    

    if all_parameters['search_type'] == 'grid':
        rcscale_space   =   np.linspace(0.01, 20.0, 30.)
        sigma_space     =   np.linspace(0.01, 0.8, 30.)
        # M_space         =   np.arange(10, 50, 5, dtype=int)**2.
        
        dict_parameters_range = dict(rc_scale=rcscale_space, sigmax=sigma_space)
        # dict_parameters_range = dict(rc_scale=rcscale_space, sigmax=sigma_space, M=M_space)
    
        # Generate the parameters
        if pbs_submit_during_parameters_generation:
            # Submit during the generation, when we find a new set of parameters.
            constrained_parameters = generate_constrained_parameters_grid(dict_parameters_range, all_parameters, experimental_fisherinfo=experimental_fi, max_mse=max_mse_fi, pbs_submission_infos=pbs_submission_infos)
        else:
            # Submit at the end, just create the list for now.
            constrained_parameters = generate_constrained_parameters_grid(dict_parameters_range, all_parameters, experimental_fisherinfo=experimental_fi, max_mse=max_mse_fi)

    elif all_parameters['search_type'] == 'random':

        # Define the parameters to optimize over
        rcscale_range = dict(low=0.01, high=15.0, dtype=float)
        sigmax_range = dict(low=0.01, high=0.8, dtype=float)
        # M_sqrt_range = dict(low=10, high=30, dtype=int)

        # Define which parameters to sample. Use the same key as the parameter argument name for submission
        dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range)

        # dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range, M_sqrt=M_sqrt_range)

        # Generate the parameters
        if pbs_submit_during_parameters_generation:
            # Submit during the generation, when we find a new set of parameters.
            constrained_parameters = generate_constrained_parameters_random(num_samples, dict_parameters_range, all_parameters, experimental_fisherinfo=experimental_fi, max_mse=max_mse_fi, pbs_submission_infos=pbs_submission_infos)
        else:
            # Submit at the end, just create the list for now.
            constrained_parameters = generate_constrained_parameters_random(num_samples, dict_parameters_range, all_parameters, experimental_fisherinfo=experimental_fi, max_mse=max_mse_fi)
    else:
        raise ValueError('Wrong search_type')
    

    dataio.save_variables(variables_to_save, locals())

    # Plot them
    plt.figure()
    plt.plot([x['rc_scale'] for x in constrained_parameters], [x['sigmax'] for x in constrained_parameters], 'x')
    plt.xlabel('rc_scale')
    plt.ylabel('sigmax')
    plt.title('Obtained constrained parameters')
    dataio.save_current_figure("experimentalvolume_generatedparameters_{unique_id}.pdf")

    # Create and submit the scripts if not already done
    if not pbs_submit_during_parameters_generation:

        submit_pbs = SubmitPBS(working_directory=pbs_submission_infos['simul_out_dir'], memory=pbs_submission_infos['memory'], walltime=pbs_submission_infos['walltime'], debug=True)

        for found_parameters in constrained_parameters:
            # Create a script for each of them
            submit_pbs.create_submit_job_parameters(pbs_submission_infos, found_parameters, submit=PBS_SUBMIT)


    return locals()




def init_everything(parameters):

    # Build the random network
    random_network = init_random_network(parameters)
        
    # Construct the real dataset
    time_weights_parameters = dict(weighting_alpha=parameters['alpha'], weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = parameters['T']-1

    # print "Building the database"
    data_gen = DataGeneratorRFN(parameters['N'], parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])
    
    # Measure the noise structure
    # print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(5000, parameters['T'], random_network, sigma_y=parameters['sigmay'], sigma_x=parameters['sigmax'], time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, stimuli_generation=parameters['stimuli_generation'])
    stat_meas = StatisticsMeasurer(data_gen_noise)
    
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

    return (random_network, data_gen, stat_meas, sampler)



def init_random_network(parameters):

    # Build the random network
    
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

    return random_network
    


def generate_constrained_parameters_grid(dict_parameters_range, all_parameters, experimental_fisherinfo=36.94, max_mse=100., pbs_submission_infos=None):
    '''
        Takes a dictionary of parameters, with their list of values, and generates a list of all the combinations
        that satisfy the experimental constraint.

        if pbs_submission_infos is provided, will create a script and submit it to PBS when an acceptable set of parameters if found
    '''

    candidate_parameters = []

    # Get all cross combinations of parameters
    cross_comb = cross([dict_parameters_range[param].tolist() for param in dict_parameters_range])
    # Convert them back into dictionaries
    candidate_parameters = [dict(zip(dict_parameters_range.keys(), x)) for x in cross_comb]

    # Now filter them
    constrained_parameters = []
    for new_parameters in progress.ProgressDisplay(candidate_parameters, display=progress.SINGLE_LINE):
        if check_experimental_constraint(new_parameters, all_parameters, experimental_fisherinfo=experimental_fisherinfo, max_mse=max_mse):
            constrained_parameters.append(new_parameters)

            # Submit to PBS if required
            if pbs_submission_infos:
                submit_pbs = SubmitPBS(working_directory=pbs_submission_infos['simul_out_dir'], memory=pbs_submission_infos['memory'], walltime=pbs_submission_infos['walltime'], debug=True)
                
                submit_pbs.create_submit_job_parameters(pbs_submission_infos, new_parameters, submit=PBS_SUBMIT)


    return constrained_parameters



def generate_constrained_parameters_random(num_samples, dict_parameters_range, all_parameters, experimental_fisherinfo=36.94, max_mse=10.0, pbs_submission_infos=None):
    '''
        Takes a dictionary of parameters (which should contain low/high values for each), and 
        generates num_samples possible parameters, within experimentally found Fisher Information.
    '''

    constrained_parameters = []
    
    fill_parameters_progress = progress.Progress(num_samples)

    tested_parameters = 0

    # Provide as many experimentally constrained parameters as desired
    while len(constrained_parameters) < num_samples:
        print "Parameters tested %d, found %d. %.2f%%, %s left - %s" % (tested_parameters, len(constrained_parameters), fill_parameters_progress.percentage(), fill_parameters_progress.time_remaining_str(), fill_parameters_progress.eta_str())

        # Sample new parameter values
        new_parameters = {}
        
        for curr_param in dict_parameters_range:    
            new_parameters[curr_param] = dict_parameters_range[curr_param]['low'] + np.random.rand()*(dict_parameters_range[curr_param]['high'] - dict_parameters_range[curr_param]['low'])

        # Check if the new parameters are within the constraints
        if check_experimental_constraint(new_parameters, all_parameters, dict_parameters_range, experimental_fisherinfo=experimental_fisherinfo, max_mse=max_mse):
            # Yes, all good

            # Append to our parameters
            constrained_parameters.append(new_parameters)

            # If desired, generate a script and submits it to PBS
            if pbs_submission_infos:
                submit_pbs = SubmitPBS(working_directory=pbs_submission_infos['simul_out_dir'], memory=pbs_submission_infos['memory'], walltime=pbs_submission_infos['walltime'], debug=True)
                
                submit_pbs.create_submit_job_parameters(pbs_submission_infos, new_parameters, submit=PBS_SUBMIT)


            fill_parameters_progress.increment()

        tested_parameters += 1


    return constrained_parameters



def check_experimental_constraint(parameters, all_parameters, dict_parameters_range, experimental_fisherinfo=0.0, max_mse=10.0):
    '''
        Takes parameters, and returns True if the theoretical Fisher Info is within the provided bounds
        for the experimental FI
    '''

    print parameters

    # Set the parameters we wish to assess
    for param_name in parameters:
        if param_name == 'M_sqrt':
            all_parameters['M'] = dict_parameters_range[param_name]['dtype'](parameters[param_name]**2.)
        else:
            all_parameters[param_name] = dict_parameters_range[param_name]['dtype'](parameters[param_name])

    # Initalize everything... If we could use the theoretical FI or covariance, would be better...
    # (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
    random_network = init_random_network(all_parameters)
    computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=1, beta=1.0, precision=50)

    # cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
    
    # if cov_div > 1.0:
    #     print cov_div
    #     print all_parameters

    #     pcolor_2d_data(computed_cov)
    #     pcolor_2d_data(sampler.noise_covariance)
    #     plt.show()

    #     raise ValueError('Big divergence between measured and theoretical divergence!')


    # theoretical_fi = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=sampler.noise_covariance)
    theoretical_fi = random_network.compute_fisher_information(stimulus_input=(0.0, 0.0), cov_stim=computed_cov)

    # Check if in the bounds or not
    return (theoretical_fi - experimental_fisherinfo)**2. < max_mse
    
    





