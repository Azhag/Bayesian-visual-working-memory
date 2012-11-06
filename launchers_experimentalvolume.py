#!/usr/bin/env python
# encoding: utf-8
"""
launchers_experimentalvolume.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

import os

import matplotlib.pyplot as plt
import scipy.interpolate as spint

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from datapbs import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from submitpbs import *

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
    max_mse_fi = 250.0

    # PBS submission informations
    pbs_submit_during_parameters_generation = True
    
    # pbs_submission_infos = dict(description='Testing the random parameter generator.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_fisher_information_param_search_pbs', output_directory='.', M=400, sigmax=0.2, rc_scale=4.0, N=300, T=1, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', num_repetitions=all_parameters['num_repetitions'], label='allfi_randomparams_M400N300samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'random_search_fi'))

    # pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=225, sigmax=0.2, rc_scale=4.0, N=200, T=6, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M225N200samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained'))
    
    pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information (use 2x experimental one here)', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=400, sigmax=0.2, rc_scale=4.0, N=200, T=6, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M400N200samples300_filterfitwo'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained_expfitwo'))

    # Instantiate a SubmitPBS, handling the parameter generation and PBS submission.
    submit_pbs = SubmitPBS(pbs_submission_infos=pbs_submission_infos, debug=True)


    if all_parameters['search_type'] == 'random':

        # Define the parameters to optimize over
        rcscale_range = dict(sampling_type='uniform', low=0.01, high=15.0, dtype=float)
        sigmax_range = dict(sampling_type='uniform', low=0.01, high=0.5, dtype=float)
        # M_sqrt_range = dict(sampling_type='randint', low=10, high=30, dtype=int)

        # Define which parameters to sample. Use the same key as the parameter argument name for submission
        dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range)

        # dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range, M_sqrt=M_sqrt_range)

        # Generate the parameters
        # Submit during the generation, when we find a new set of parameters.
        constrained_parameters = submit_pbs.generate_submit_constrained_parameters_random(num_samples, dict_parameters_range, filtering_function=check_experimental_constraint, filtering_function_parameters=dict(all_parameters=all_parameters, experimental_fisherinfo=2*experimental_fi, max_mse=max_mse_fi), pbs_submission_infos=pbs_submission_infos, submit_jobs=pbs_submit_during_parameters_generation)
     
    elif all_parameters['search_type'] == 'grid':

        rcscale_range     =   dict(range=np.linspace(0.01, 20.0, 10.), dtype=float)
        sigmax_range      =   dict(range=np.linspace(0.01, 0.8, 10.), dtype=float)
        # M_range           =   dict(range=np.arange(10, 50, 5, dtype=int)**2., dtype=int)
        
        dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range)
        # dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range, M=M_range)
    
        # Generate the parameters
        # Submit during the generation, when we find a new set of parameters (pbs_submit_during_parameters_generation=True)
        constrained_parameters = submit_pbs.generate_submit_constrained_parameters_grid(dict_parameters_range, filtering_function=check_experimental_constraint, filtering_function_parameters=dict(all_parameters=all_parameters, experimental_fisherinfo=2*experimental_fi, max_mse=max_mse_fi), pbs_submission_infos=pbs_submission_infos, submit_jobs=pbs_submit_during_parameters_generation)

    else:
        raise ValueError('Wrong search_type')
    
    # Clean dict_parameters_range for pickling
    for param, param_dict in dict_parameters_range.items():
        param_dict['sampling_fct'] = None

    dataio.save_variables(variables_to_save, locals())

    # Plot them
    plt.figure()
    plt.plot([x['rc_scale'] for x in constrained_parameters], [x['sigmax'] for x in constrained_parameters], 'x')
    plt.xlabel('rc_scale')
    plt.ylabel('sigmax')
    plt.title('Obtained constrained parameters')
    dataio.save_current_figure("experimentalvolume_generatedparameters_{unique_id}.pdf")

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
    

def check_experimental_constraint(parameters, dict_parameters_range, function_parameters=None):
    '''
        Takes parameters, and returns True if the theoretical Fisher Info is within the provided bounds
        for the experimental FI
    '''

    print parameters

    all_parameters = function_parameters['all_parameters']
    experimental_fisherinfo = function_parameters['experimental_fisherinfo']
    max_mse = function_parameters['max_mse']


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
    

#########

def launcher_do_reload_constrained_parameters(args):
    '''
        Reload outputs run with the automatic parameter generator.

        Should handle random sampling of the parameter space.
    '''


    if args.subaction == '1':
        dataset_infos = dict(label='PBS run with the automatic parameter generator. Random samples of the parameters, should interpolate. Test of the FI match for now (which already looks wrong, great...)',
                        files='Data/param_generator/test_fi_scripts/random_search_fi/*launcher_do_multiple_memory_curve_simult*.npy',
                        loading_type='args',
                        parameters=('rc_scale', 'sigmax'),
                        variables_to_load=['FI_rc_curv_mult', 'FI_rc_precision_mult', 'FI_rc_theo_mult', 'FI_rc_truevar_mult'],
                        variables_description=['FI curve', 'FI recall precision', 'FI theo'],
                        post_processing=plots_randomsamples_fi
                        )
    elif args.subaction == '2':
        dataset_infos = dict(label='PBS run with the automatic parameter generator. Random samples of the parameters, should interpolate. Memory curves results',
                    files='Data/param_generator/memorycurves_constrainedfi/*launcher_do_multiple_memory_curve_simult*.npy',
                    # files='Data/param_generator/memorycurves_constrainedfi_expfitwo/*launcher_do_multiple_memory_curve_simult*.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax'],
                    variables_to_load=['all_precisions', 'power_law_params'],
                    variables_description=['Precisions', 'Power law parameters'],
                    post_processing=plots_randomsamples_memorycurves
                    )
    else:
        raise ValueError('Set subaction to the data you want to reload')

    
    # Reload everything
    data_pbs = DataPBS(dataset_infos=dataset_infos, debug=True)

    # Do the plots
    if dataset_infos['post_processing']:
        dataset_infos['post_processing'](data_pbs)


    return locals()


def plots_randomsamples_fi(data_pbs):
    '''
        Some plots for reloaded data from PBS, random samples
    '''

    # Extract the data and average it
    FI_rc_theo_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_theo_mult']['results_flat'], axis=-1))
    FI_rc_curv_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_curv_mult']['results_flat'], axis=-1))
    FI_rc_precision_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_precision_mult']['results_flat'], axis=-1))
    parameters_allpoints = np.array(data_pbs.dict_arrays['FI_rc_theo_mult']['parameters_flat'])

    ### Draw nice plots, interpolating between the sampled points
    contourf_interpolate_data(parameters_allpoints, FI_rc_theo_mean[:, 0], xlabel='rc scale', ylabel='sigma x', title='FI Theo', interpolation_numpoints=200, interpolation_method='linear')

    contourf_interpolate_data(parameters_allpoints, FI_rc_curv_mean[:, 0], xlabel='rc scale', ylabel='sigma x', title='FI Curv', interpolation_numpoints=200, interpolation_method='linear')

    contourf_interpolate_data(parameters_allpoints, FI_rc_precision_mean, xlabel='rc scale', ylabel='sigma x', title='FI recall precision', interpolation_numpoints=200, interpolation_method='linear')



def plots_randomsamples_memorycurves(data_pbs):
    '''
        Plots for memory curves run
    '''
    
    interpolation_method = 'linear'
    # interpolation_method = 'nearest'


    all_precisions_mean = np.mean(data_pbs.dict_arrays['all_precisions']['results_flat'], axis=-1)
    parameters_allpoints = np.array(data_pbs.dict_arrays['all_precisions']['parameters_flat'])

    # Random filter because of stupid runs...
    # filter_good_fi = parameters_allpoints[:, 1] < parameters_allpoints[:, 0]*0.07735 - 0.1
    # parameters_allpoints = np.array([parameters_allpoints[i] for i in xrange(parameters_allpoints.shape[0]) if filter_good_fi[i]])
    # all_precisions_mean = np.array([all_precisions_mean[i] for i in xrange(all_precisions_mean.shape[0]) if filter_good_fi[i]])

    # Plot the precision for 1 object, should be ~= to the experimental FI
    contourf_interpolate_data(parameters_allpoints, all_precisions_mean[:, 0], xlabel='rc scale', ylabel='sigma x', title='Precision 1 object', interpolation_numpoints=500, interpolation_method=interpolation_method)

    # Plot the distance to the experimental FI for 1 object
    experimental_fi = 35.94
    # experimental_fi = 18.08
    max_mse_fi = 800.0
    dist_experimental_fi_1obj = (all_precisions_mean[:, 0] - experimental_fi)**2.

    # Filter points too bad...
    # dist_experimental_fi_1obj[dist_experimental_fi_1obj > max_mse_fi] = np.nan
    # dist_experimental_fi_1obj = np.ma.masked_where(dist_experimental_fi_1obj > max_mse_fi, dist_experimental_fi_1obj)
    
    contourf_interpolate_data(parameters_allpoints, dist_experimental_fi_1obj, xlabel='rc scale', ylabel='sigma x', title='Precision 1 object MSE from Experimental FI', interpolation_numpoints=500, interpolation_method=interpolation_method)

    #### Check how good the fit is for the full memory curve
    experimental_fit_full = np.load('Data/experimental_data/processed_experimental.npy').item()
    experimental_memory_curve = np.mean(experimental_fit_full['data_simult']['precision_subject_nitems_theo'], axis=0)

    dist_memory_fits = np.sum(np.abs(all_precisions_mean[:, :-1] - experimental_memory_curve), axis=1)

    contourf_interpolate_data(parameters_allpoints, dist_memory_fits, xlabel='rc scale', ylabel='sigma x', title='Distance between fits and experimental memory curves', interpolation_numpoints=500, interpolation_method=interpolation_method)




