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

import progress

def launcher_do_generate_constrained_param_experimental_theo(args):
    '''
        Generate parameters consistent with the experimentally found FI.
        Uses the theoretical FI for our given population code to verify constraint.

        Should then spawn simulations, or find a way to use those appropriately.
    '''

    all_parameters = vars(args)
    
    variables_to_save = ['dict_parameters_range', 'constrained_parameters', 'pbs_submission_infos', 'filtering_function_params']
    dataio = DataIO(output_folder=args.output_directory, label=args.label)

    num_samples = all_parameters['num_samples']

    # Experimental FI informations
    # experimental_fi = 35.94
    # experimental_fi = 9.04
    experimental_fi = 8.81007762
    max_mse_fi = 500.0

    # PBS submission informations
    pbs_submit_during_parameters_generation = True

    rcscale_space = np.linspace(0.01, 15.0, 10.)
    sigma_space = np.linspace(0.01, 0.8, 10.)
    M_space = np.arange(10, 30, 4, dtype=int)**2.
    
    # pbs_submission_infos = dict(description='Testing the random parameter generator.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_fisher_information_param_search_pbs', output_directory='.', M=625, sigmax=0.2, rc_scale=4.0, N=300, T=1, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', num_repetitions=all_parameters['num_repetitions'], label='allfi_randomparams_M625N300samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'random_search_fi'))
    
    # pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=225, sigmax=0.2, rc_scale=4.0, N=200, T=6, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M225N200samples300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained'))

    # pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=225, sigmax=0.2, rc_scale=4.0, N=200, T=6, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M225N200samples300fi9numrepet5'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained_fi9_corr'))
    
    # pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information (use 2x experimental one here). M=625', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=625, sigmax=0.2, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_randomparams_M625N200samples300_newruns'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained_M625_newruns'))

    # pbs_submission_infos = dict(description='Memory curves for correct experimental FI. M=625', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_multiple_memory_curve_simult', code_type='conj', output_directory='.', M=625, sigmax=0.2, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='memorycurves_correctfi_randomparams_M625N200samples300_121112'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'memory_curves_constrained_M625_correctfi'))

    # pbs_submission_infos = dict(description='3D Grid search for FI Theo+var. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_memorycurve_theoretical_pbs', code_type='conj', output_directory='.', M=400, sigmax=0.1, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='3dvolume_memorycurves_theovar_grid_151112'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), '3dvolume_memcurve_theovar_151112'))
    # rcscale_space = np.linspace(0.01, 15.0, 10.)
    # sigma_space = np.linspace(0.01, 0.8, 10.)
    # # M_space = np.arange(10, 30, 4, dtype=int)**2.
    # M_space = np.arange(27, 35, 2, dtype=int)**2.

    # pbs_submission_infos = dict(description='3D Grid search for FI Theo+var. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_memorycurve_theoretical_pbs_theoonly', code_type='conj', output_directory='.', M=400, sigmax=0.1, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='3dvolume_memorycurves_theoonlybigspace_grid_151112'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), '3dvolume_memcurve_theoonlybigspace_reruns_bis_151112'))
    # rcscale_space = np.linspace(0.01, 10.0, 51.)
    # sigma_space = np.linspace(0.01, 0.8, 50.)

    # pbs_submission_infos = dict(description='Mixed populations. 3D Grid search for FI Theo+var. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_memorycurve_theoretical_pbs', code_type='mixed', output_directory='.', M=625, sigmax=0.1, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', rc_scale2=0.4, ratio_conj=0.5, feat_ratio=-100, selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='mixedpop05_3dvolume_memorycurves_theovar_grid_211112_featratio_100'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'mixedpop05_3dvolume_memcurve_theovar_211112_diffspace'))
    # pbs_submission_infos = dict(description='Mixed populations. 3D Grid search for FI Theo+var. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_memorycurve_theoretical_pbs', code_type='mixed', output_directory='.', M=625, sigmax=0.1, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', rc_scale2=0.4, ratio_conj=0.1, feat_ratio=-100, selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='mixedpop01_3dvolume_memorycurves_theovar_grid_221112_featratio_100'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'mixedpop01_3dvolume_memcurve_theovar_221112'))
    pbs_submission_infos = dict(description='Mixed populations. 3D Grid search for FI Theo+var. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_memorycurve_theoretical_pbs', code_type='mixed', output_directory='.', M=625, sigmax=0.1, rc_scale=4.0, N=200, T=5, alpha=1.0, sigmay=0.0001, num_samples=300, selection_method='last', rc_scale2=0.4, ratio_conj=0.25, feat_ratio=-100, selection_num_samples=300, inference_method='sample', num_repetitions=all_parameters['num_repetitions'], label='mixedpop025_3dvolume_memorycurves_theovar_grid_231112_featratio_300'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'mixedpop025_specialspace_3dvolume_memcurve_theovar_231112'))
    rcscale_space = np.linspace(0.01, 10.0, 10.)
    sigma_space = np.linspace(0.01, 3.0, 10.)
    rcscale2_space = np.linspace(0.01, 0.3, 10.)
    # # M_space = np.arange(10, 30, 4, dtype=int)**2.
    # M_space = np.arange(27, 35, 2, dtype=int)**2.
    activate_filtering = False

    # Create the filtering all parameters dict, make sure parameters are appropriately shared...
    filtering_all_parameters = all_parameters.copy()
    for key, val in pbs_submission_infos['other_options'].items():
        filtering_all_parameters[key] = val

    filtering_function_params = dict(all_parameters=filtering_all_parameters, experimental_fisherinfo=2*experimental_fi, max_mse=max_mse_fi, use_theoretical_cov=False)


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

        constrained_parameters = submit_pbs.generate_submit_constrained_parameters_random(num_samples, dict_parameters_range, filtering_function=check_experimental_constraint, filtering_function_parameters=filtering_function_params, pbs_submission_infos=pbs_submission_infos, submit_jobs=pbs_submit_during_parameters_generation)
     
    elif all_parameters['search_type'] == 'grid':
        print "Launching grid!"

        rcscale_range     =   dict(range=rcscale_space, dtype=float)
        sigmax_range      =   dict(range=sigma_space, dtype=float)
        M_range           =   dict(range=M_space, dtype=int)
        rcscale2_range    =   dict(range=rcscale2_space, dtype=float)
        
        # dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range)
        # dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range, M=M_range)
        dict_parameters_range = dict(rc_scale=rcscale_range, sigmax=sigmax_range, rc_scale2=rcscale2_range)
    
        # Generate the parameters
        # Submit during the generation, when we find a new set of parameters (pbs_submit_during_parameters_generation=True)
        if activate_filtering:
            constrained_parameters = submit_pbs.generate_submit_constrained_parameters_grid(dict_parameters_range, filtering_function=check_experimental_constraint, filtering_function_parameters=dict(all_parameters=all_parameters, experimental_fisherinfo=2.*experimental_fi, max_mse=max_mse_fi), pbs_submission_infos=pbs_submission_infos, submit_jobs=pbs_submit_during_parameters_generation)
        else:
            constrained_parameters = submit_pbs.generate_submit_constrained_parameters_grid(dict_parameters_range, pbs_submission_infos=pbs_submission_infos, submit_jobs=pbs_submit_during_parameters_generation)

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
    if 'use_theoretical_cov' in function_parameters:
        use_theoretical_cov = function_parameters['use_theoretical_cov']
    else:
        use_theoretical_cov = True
    if 'check_theoretical_cov' in function_parameters:
        check_theoretical_cov = function_parameters['check_theoretical_cov']
    else:
        check_theoretical_cov = False


    # Set the parameters we wish to assess
    for param_name in parameters:
        if param_name == 'M_sqrt':
            all_parameters['M'] = dict_parameters_range[param_name]['dtype'](parameters[param_name]**2.)
        else:
            all_parameters[param_name] = dict_parameters_range[param_name]['dtype'](parameters[param_name])

    # Initalize everything... If we could use the theoretical FI or covariance, would be better...
    if use_theoretical_cov:
        if check_theoretical_cov:
            (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)

            computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=1, beta=1.0, precision=50)

            cov_div = np.mean((sampler.noise_covariance-computed_cov)**2.)
            if cov_div > 0.001:
                print cov_div
                print all_parameters

                pcolor_2d_data(computed_cov)
                pcolor_2d_data(sampler.noise_covariance)
                plt.show()

                raise ValueError('Big divergence between measured and theoretical divergence!')
        else:
            random_network = init_random_network(all_parameters)

            computed_cov = random_network.compute_covariance_KL(sigma_2=(all_parameters['sigmax']**2. + all_parameters['sigmay']**2.), T=1, beta=1.0, precision=50)

    else:
        (random_network, data_gen, stat_meas, sampler) = init_everything(all_parameters)
        # computed_cov = sampler.noise_covariance
        computed_cov = stat_meas.model_parameters['covariances'][-1, 0]


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
                        # files='Data/param_generator/test_fi_scripts/random_search_fi/*.npy',
                        files='Data/param_generator/new_test_fi/random_search_fi/*.npy',
                        launcher_file='Data/param_generator/new_test_fi/launcher_do_generate_constrained_param_experimental_theo-5c73e45e-aa2b-4fe3-ad7a-0f00fa88f9b4.npy',
                        loading_type='args',
                        parameters=('rc_scale', 'sigmax'),
                        variables_to_load=['FI_rc_curv_mult', 'FI_rc_precision_mult', 'FI_rc_theo_mult', 'FI_rc_truevar_mult'],
                        variables_description=['FI curve', 'FI recall precision', 'FI theo'],
                        post_processing=plots_randomsamples_fi
                        )
    elif args.subaction == '2':
        dataset_infos = dict(label='PBS run with the automatic parameter generator. Random samples of the parameters, should interpolate. Memory curves results',
                    # files='Data/param_generator/memorycurves_constrainedfi/*launcher_do_multiple_memory_curve_simult*.npy',
                    files='Data/param_generator/memorycurves_constrainedfi_expfitwo/*launcher_do_multiple_memory_curve_simult*.npy',
                    # files='Data/param_generator/memorycurves_constrainedfi_all/all_memcurv/*.npy',
                    launcher_file='Data/param_generator/memorycurves_constrainedfi/launcher_do_generate_constrained_param_experimental_theo-94db3de3-fd5f-4b86-a48f-904074971468.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax'],
                    variables_to_load=['all_precisions', 'power_law_params'],
                    variables_description=['Precisions', 'Power law parameters'],
                    post_processing=plots_randomsamples_memorycurves
                    )
    elif args.subaction == '3':
        dataset_infos = dict(label='New runs, corrected contraint... Memory curves results',
                    files='Data/param_generator/new_memorycurves/memory_curves_constrained_M625_newruns/*launcher_do_multiple_memory_curve_simult*.npy',
                    launcher_file='Data/param_generator/new_memorycurves/launcher_do_generate_constrained_param_experimental_theo-802881b2-b2b0-4699-bb27-132f502ac7bd.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax'],
                    variables_to_load=['all_precisions', 'power_law_params'],
                    variables_description=['Precisions', 'Power law parameters'],
                    post_processing=plots_randomsamples_memorycurves
                    )
    elif args.subaction == '4':
        dataset_infos = dict(label='New runs, corrected contraint, correct FI (8.81). Memory curves',
                    files='Data/param_generator/new_memorycurves_correctexpfi8.81/memory_curves_constrained_M625_correctfi/memorycurves_correctfi_randomparams_M625N200samples300_121112*.npy',
                    launcher_file='Data/param_generator/new_memorycurves_correctexpfi8.81/launcher_do_generate_constrained_param_experimental_theo-833ff820-8c2e-4a9d-bbec-9583fe4b11c0.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax'],
                    variables_to_load=['all_precisions', 'power_law_params'],
                    variables_description=['Precisions', 'Power law parameters'],
                    post_processing=plots_randomsamples_memorycurves
                    )
    elif args.subaction == '5':
        dataset_infos = dict(label='3D volume. Memory curves. Theoretical and Variances. Used in Cosyne 2013 abstract.',
            files='Data/param_generator/3dvolume_theomemcurves_theovar/3dvolume_memcurve_theovar_151112/3dvolume_memorycurves_theovar_grid_151112-*.npy',
            # files='Data/param_generator/3dvolume_theomemcurves_theovar/3dvolume_memcurve_theovar_151112_extendedruns/3dvolume_memorycurves_theovar_grid_151112-*.npy',
                    launcher_file='Data/param_generator/3dvolume_theomemcurves_theovar/launcher_do_generate_constrained_param_experimental_theo-a55ca6b8-ec1a-468c-a81b-6a38f4086318.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax', 'M'],
                    variables_to_load=['FI_rc_theo_mult', 'FI_rc_var_mult'],
                    variables_description=['FI theory', 'FI var'],
                    post_processing=plots_3dvolume_theovar
                    )
    elif args.subaction == '6':
        dataset_infos = dict(label='3D volume. Memory curves. Theoretical big space',
                    
                    # files='Data/param_generator/3dvolume_theomemcurves_theobigspace/3dvolume_memcurve_theoonlybigspace_151112/3dvolume_memorycurves_theoonlybigspace_grid_151112-*.npy',
                    # launcher_file='Data/param_generator/3dvolume_theomemcurves_theobigspace/launcher_do_generate_constrained_param_experimental_theo-e86df8d5-df0e-4a54-8c6a-59b24de1b869.npy',
                    # files='Data/param_generator/3dvolume_theomemcurves_theobigspace/3dvolume_memcurve_theoonlybigspace_reruns_151112/3dvolume_memorycurves_theoonlybigspace_grid_151112-*.npy',
                    files='Data/param_generator/3dvolume_theomemcurves_theobigspace/3dvolume_memcurve_theoonlybigspace_reruns_bis_151112/3dvolume_memorycurves_theoonlybigspace_grid_151112-*.npy',
                    launcher_file='Data/param_generator/3dvolume_theomemcurves_theobigspace/reruntheobigspace-launcher_do_generate_constrained_param_experimental_theo-ab895bf4-b41d-414b-9afe-9349bf9d5e85.npy',

                    loading_type='args',
                    parameters=['rc_scale', 'sigmax'],
                    variables_to_load=['FI_rc_theo_mult'],
                    variables_description=['FI theory'],
                    post_processing=plots_3dvolume_theo
                    )
    elif args.subaction == '7':
        dataset_infos = dict(label='Mixed population. 3D volume. Memory curves. Theoretical and Variances.',
                    files='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/mixedpop05_3dvolume_memcurve_theovar_161112/mixedpop05_3dvolume_memorycurves_theovar_grid_161112_rcscale2_04_featratio_100-*.npy',
                    launcher_file='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/launcher_do_generate_constrained_param_experimental_theo-5b279312-035e-4126-9399-b94234eb2f01.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax', 'M'],
                    variables_to_load=['FI_rc_theo_mult', 'FI_rc_var_mult'],
                    variables_description=['FI theory', 'FI var'],
                    post_processing=plots_3dvolume_theovar
                    )
    elif args.subaction == '8':
        dataset_infos = dict(label='Mixed population. 3D volume (rcscale, sigma, rcscale2). Memory curves. Theoretical and Variances.',
                    # files='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/mixedpop05_3dvolume_memcurve_theovar_211112/mixedpop05_3dvolume_memorycurves_theovar_grid_211112_featratio_100-*.npy',
                    # files='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/mixedpop05_3dvolume_memcurve_theovar_211112_diffspace/mixedpop05_3dvolume_memorycurves_theovar_grid_211112_featratio_100-*.npy',
                    # files='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/mixedpop01_3dvolume_memcurve_theovar_221112/mixedpop01_3dvolume_memorycurves_theovar_grid_221112_featratio_100-*.npy',
                    files='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/mixedpop025_specialspace_3dvolume_memcurve_theovar_231112/mixedpop025_3dvolume_memorycurves_theovar_grid_231112_featratio_300-*.npy',
                    launcher_file='Data/param_generator/3dvolume_mixedpop_memcurves_theovar/launcher_do_generate_constrained_param_experimental_theo-dfd16533-4151-49c1-b3a4-45f5f50b11f4.npy',
                    loading_type='args',
                    parameters=['rc_scale', 'sigmax', 'rc_scale2'],
                    variables_to_load=['FI_rc_theo_mult', 'FI_rc_var_mult'],
                    variables_description=['FI theory', 'FI var'],
                    post_processing=plots_3dvolume_theovar_mixedpop
                    )
    else:
        raise ValueError('Set subaction to the data you want to reload')

    
    # Reload everything
    data_pbs = DataPBS(dataset_infos=dataset_infos, debug=True)

    # Reload the outputs of launcher used.
    launcher_variables = np.load(dataset_infos['launcher_file']).item()

    # Do the plots
    if dataset_infos['post_processing']:
        dataset_infos['post_processing'](data_pbs, launcher_variables)


    return locals()


def plots_randomsamples_fi(data_pbs, launcher_variables=None):
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

    # Check the condition
    filtering_function_params = launcher_variables['filtering_function_params']
    # filtering_function_params['use_theoretical_cov'] = True
    experimental_fisherinfo = filtering_function_params['experimental_fisherinfo']
    max_mse = filtering_function_params['max_mse']

    measured_fi_dist = (FI_rc_theo_mean[:, 0] - experimental_fisherinfo)**2.
    contourf_interpolate_data(parameters_allpoints, measured_fi_dist)

    # Check with the filter function to see if everything worked as intended...
    dict_parameters_range = launcher_variables['dict_parameters_range']
    parameters_allpoints_dict = [dict(zip(data_pbs.dataset_infos['parameters'], point_act)) for point_act in parameters_allpoints]

    # filtering_fct_output = [check_experimental_constraint(param_dict, dict_parameters_range, function_parameters=filtering_function_params) for param_dict in parameters_allpoints_dict]

    filtering_fct_output = []
    for param_dict_i in progress.ProgressDisplay(range(len(parameters_allpoints_dict))):
        curr_out = check_experimental_constraint(parameters_allpoints_dict[param_dict_i], dict_parameters_range, function_parameters=filtering_function_params)
        print curr_out, measured_fi_dist[param_dict_i] < max_mse, measured_fi_dist[param_dict_i]
        filtering_fct_output.append(curr_out)
    print filtering_fct_output



def plots_randomsamples_memorycurves(data_pbs, launcher_variables=None):
    '''
        Plots for memory curves run
    '''
    
    interpolation_method = 'linear'
    # interpolation_method = 'nearest'
    interpolation_numpoints = 300
    contour_numlevels = 50

    log_distances = False

    all_precisions_mean = np.mean(data_pbs.dict_arrays['all_precisions']['results_flat'], axis=-1)
    parameters_allpoints = np.array(data_pbs.dict_arrays['all_precisions']['parameters_flat'])
    power_law_params = np.array(data_pbs.dict_arrays['power_law_params']['results_flat'])

    # Extract parameters used
    if launcher_variables and 'filtering_function_params' in launcher_variables:
        experimental_fi_constraint = launcher_variables['filtering_function_params']['experimental_fisherinfo']
        experimental_fi = experimental_fi_constraint/2.
        # experimental_fi = experimental_fi_constraint
        max_mse_fi = launcher_variables['filtering_function_params']['max_mse']
    else:
        # experimental_fi = 35.94
        # experimental_fi = 18.08
        # experimental_fi = 9.041
        experimental_fi = 8.81
        max_mse_fi = 800.0
    
    # Random filter because of stupid runs...
    # filter_good_fi = parameters_allpoints[:, 1] < parameters_allpoints[:, 0]*0.07735 - 0.1
    # parameters_allpoints = np.array([parameters_allpoints[i] for i in xrange(parameters_allpoints.shape[0]) if filter_good_fi[i]])
    # all_precisions_mean = np.array([all_precisions_mean[i] for i in xrange(all_precisions_mean.shape[0]) if filter_good_fi[i]])

    # Plot the precision for 1 object, should be ~= to the experimental FI
    contourf_interpolate_data(parameters_allpoints, all_precisions_mean[:, 0], xlabel='rc scale', ylabel='sigma x', title='Precision 1 object, target FI: %.2f' % experimental_fi, interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    # Plot the distance to the experimental FI for 1 object
    dist_experimental_fi_1obj = (all_precisions_mean[:, 0] - experimental_fi)**2.

    # Filter points too bad...
    # dist_experimental_fi_1obj[dist_experimental_fi_1obj > max_mse_fi] = np.nan
    # dist_experimental_fi_1obj = np.ma.masked_where(dist_experimental_fi_1obj > max_mse_fi, dist_experimental_fi_1obj)
    
    if log_distances:
        contourf_interpolate_data(parameters_allpoints, np.log(dist_experimental_fi_1obj), xlabel='rc scale', ylabel='sigma x', title='log MSE between precision 1 object and Experimental FI %.2f' % experimental_fi, interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)
    else:
        contourf_interpolate_data(parameters_allpoints, dist_experimental_fi_1obj, xlabel='rc scale', ylabel='sigma x', title='MSE between precision 1 object and Experimental FI %.2f' % experimental_fi, interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)



    #### Check how good the fit is for the full memory curve
    # experimental_fit_full = np.load('Data/experimental_data/processed_experimental.npy').item()
    # experimental_memory_curve = np.mean(experimental_fit_full['data_simult']['precision_subject_nitems_theo'], axis=0)
    # experimental_memory_curve = np.array([ 9.04094574, 4.91125975, 3.8211774, 2.59703441, 1.89610192])
    # experimental_memory_curve = np.array([ 9.04094574, 4.91125975, 3.8211774, 2.59703441, 1.89610192])*2.
    experimental_memory_curve = np.array([ 8.81007762,  4.7976755,   3.61554792,  2.4624979,   1.78252039])

    dist_memory_fits = np.sum((all_precisions_mean[:, :5] - experimental_memory_curve)**2., axis=1)

    if log_distances:
        contourf_interpolate_data(parameters_allpoints, np.log(dist_memory_fits), xlabel='rc scale', ylabel='sigma x', title='Log distance between fits and experimental memory curves', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)
    else:
        contourf_interpolate_data(parameters_allpoints, dist_memory_fits, xlabel='rc scale', ylabel='sigma x', title='Distance between fits and experimental memory curves', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    plt.figure()
    plt.plot(all_precisions_mean[np.argmin(dist_memory_fits)])
    plt.plot(experimental_memory_curve)
    plt.title('Best fit, memory. Params: %s' % parameters_allpoints[np.argmin(dist_memory_fits)])


    ## Check the powerlaw exponents
    # power_law_params = np.ma.masked_equal(power_law_params, 0.0)

    contourf_interpolate_data(parameters_allpoints, power_law_params[:, 0], xlabel='rc scale', ylabel='sigma x', title='Power law exponent', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    contourf_interpolate_data(parameters_allpoints, power_law_params[:, 1], xlabel='rc scale', ylabel='sigma x', title='Power law magnitude', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    experimental_powerlaw_params = fit_powerlaw(np.arange(1, 6), experimental_memory_curve)
    dist_powerlaw_fits = np.sum((power_law_params - experimental_powerlaw_params)**2., axis=1)
    
    contourf_interpolate_data(parameters_allpoints, dist_powerlaw_fits, xlabel='rc scale', ylabel='sigma x', title='MSE exp/model power law parameters', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    contourf_interpolate_data(parameters_allpoints, np.log(dist_powerlaw_fits), xlabel='rc scale', ylabel='sigma x', title='Log MSE exp/model power law parameters', interpolation_numpoints=interpolation_numpoints, interpolation_method=interpolation_method, contour_numlevels=contour_numlevels)

    plt.figure()
    plt.plot(all_precisions_mean[np.argmin(dist_powerlaw_fits)])
    plt.plot(experimental_memory_curve)
    plt.title('Best fit, powerlaw. Params: %s' % parameters_allpoints[np.argmin(dist_powerlaw_fits)])



def plots_3dvolume_theovar(data_pbs, launcher_variables=None):
    '''
        Reload 3D volume runs from PBS and plot them
    '''

    FI_rc_theo_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_theo_mult']['results'], axis=-1))
    FI_rc_var_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_var_mult']['results'], axis=-1))

    if len(FI_rc_theo_mean.shape) == 3:
        # HACK There was only one M, put it back
        FI_rc_theo_mean = FI_rc_theo_mean[:, :, np.newaxis]
    if len(FI_rc_var_mean.shape) == 4:
        FI_rc_var_mean = FI_rc_var_mean[:, :, np.newaxis]

    rcscale_space = data_pbs.loaded_data['parameters_uniques']['rc_scale']
    sigma_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    
    
    exp_target = np.array([ 8.81007762,  4.7976755,   3.61554792,  2.4624979,   1.78252039])
    interpolation = 'nearest'

    # Check the fits
    dist_theo_exp = np.sum((FI_rc_theo_mean - exp_target)**2., axis=-1)
    log_dist_theo_exp = np.log(dist_theo_exp)

    dist_theo_1obj = (FI_rc_theo_mean[..., 0] - exp_target[0])**2.
    log_dist_theo_1obj = np.log(dist_theo_1obj)
    
    dist_var_exp = np.sum((FI_rc_var_mean[..., 0] - exp_target)**2., axis=-1)
    log_dist_var_exp = np.log(dist_var_exp)

    ratio_theo_var = np.mean((FI_rc_theo_mean/FI_rc_var_mean[..., 0]), axis=-1)

    for m_i, M in enumerate(M_space):
        # Do one pcolor per M
        pcolor_2d_data(log_dist_theo_exp[..., m_i], x=rcscale_space, y=sigma_space, title='Log distance theory to experimental memory curve. M: %d' % M, label_format="%.2f", interpolation=interpolation)
        pcolor_2d_data(log_dist_var_exp[..., m_i], x=rcscale_space, y=sigma_space, title='Log distance variance to experimental memory curve. M: %d' % M, label_format="%.2f", interpolation=interpolation)
        pcolor_2d_data(log_dist_theo_1obj[..., m_i], x=rcscale_space, y=sigma_space, title='Log distance theory 1 object to experimental 1 object. M: %d' % M, label_format="%.2f", interpolation=interpolation)
        pcolor_2d_data(ratio_theo_var[..., m_i], x=rcscale_space, y=sigma_space, title='Ratio theo/var', label_format="%.2f", interpolation=interpolation)

        # Best fits plots
        # Get the list of best values.
        besttheo_mem_argsort = np.argsort(dist_theo_exp[..., m_i], axis=None)
        bestvar_mem_argsort = np.argsort(dist_var_exp[..., m_i], axis=None)
        besttheo_1obj_argsort = np.argsort(dist_theo_1obj[..., m_i], axis=None)

        # Show the results for the best K
        best_k = 3

        for bk_i in np.arange(best_k):
            # Convert the flat best index into indices
            indices_besttheo_mem = np.unravel_index(besttheo_mem_argsort[bk_i], dist_theo_exp[..., m_i].shape)
            indices_bestvar_mem = np.unravel_index(bestvar_mem_argsort[bk_i], dist_var_exp[..., m_i].shape)
            indices_besttheo_1obj = np.unravel_index(besttheo_1obj_argsort[bk_i], dist_theo_1obj[..., m_i].shape)

            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_mem][m_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][m_i, :, 0], 'o-', linewidth=3)
            current_color = ax.get_lines()[-1].get_c()
            ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][m_i, :, 0]-FI_rc_var_mean[indices_bestvar_mem][m_i, :, 1], FI_rc_var_mean[indices_bestvar_mem][m_i, :, 0]+FI_rc_var_mean[indices_bestvar_mem][m_i, :, 1], facecolor=current_color, alpha=0.4)
            plt.title('%d Best fits for M=%d: kappa_theo %.2f, sigma_theo %.2f | kappa_var %.2f, sigma_var %.2f' % (bk_i+1, M, rcscale_space[indices_besttheo_mem[0]], sigma_space[indices_besttheo_mem[1]], rcscale_space[indices_bestvar_mem[0]], sigma_space[indices_bestvar_mem[1]]))
            plt.legend(['Experimental data', 'best Theoretical FI', 'best Posterior variance'])
            plt.xticks(np.arange(1, 6))
            plt.xlim([0.8, 5.2])

            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_mem][m_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_besttheo_mem][m_i, :, 0], 'o-', linewidth=3)
            current_color = ax.get_lines()[-1].get_c()
            ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_besttheo_mem][m_i, :, 0]-FI_rc_var_mean[indices_besttheo_mem][m_i, :, 1], FI_rc_var_mean[indices_besttheo_mem][m_i, :, 0]+FI_rc_var_mean[indices_besttheo_mem][m_i, :, 1], facecolor=current_color, alpha=0.4)
            plt.title('%d Best fits for M=%d: kappa_theo %.2f, sigma_theo %.2f' % (bk_i+1, M, rcscale_space[indices_besttheo_mem[0]], sigma_space[indices_besttheo_mem[1]]))
            plt.legend(['Experimental data', 'best Theoretical FI', 'Posterior variance, theo best'])
            plt.xticks(np.arange(1, 6))
            plt.xlim([0.8, 5.2])

            f = plt.figure()
            ax = f.add_subplot(111)
            ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_1obj][m_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_besttheo_1obj][m_i, :, 0], 'o-', linewidth=3)
            current_color = ax.get_lines()[-1].get_c()
            ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_besttheo_1obj][m_i, :, 0]-FI_rc_var_mean[indices_besttheo_1obj][m_i, :, 1], FI_rc_var_mean[indices_besttheo_1obj][m_i, :, 0]+FI_rc_var_mean[indices_besttheo_1obj][m_i, :, 1], facecolor=current_color, alpha=0.4)
            plt.title('%d Best fits for M=%d: kappa_1obj %.2f, sigma_1obj %.2f' % (bk_i+1, M, rcscale_space[indices_besttheo_1obj[0]], sigma_space[indices_besttheo_1obj[1]]))
            plt.legend(['Experimental data', 'Theoretical FI, 1obj', 'Posterior variance, theo 1obj'])
            plt.xticks(np.arange(1, 6))
            plt.xlim([0.8, 5.2])

        # plt.figure()
        # plt.plot(np.arange(1, 6), FI_rc_var_mean[argmin_indices(dist_var_exp[..., m_i])][m_i, :, 0], np.arange(1, 6), exp_target)
        # plt.title('Best var for M=%d' % M)


def plots_3dvolume_theo(data_pbs, launcher_variables=None):
    '''
        Reconstruct the matrices to be used for launchers_memorycurves.plots_memorycurve_theoretical_3dvolume
    '''

    import launchers_memorycurves

    # Load and setup the data
    FI_rc_theo_mult = np.squeeze(data_pbs.dict_arrays['FI_rc_theo_mult']['results'])
    rcscale_space = data_pbs.loaded_data['parameters_uniques']['rc_scale']
    sigma_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    M_space = data_pbs.loaded_data['datasets_list'][0]['M_space']

    if False and len(FI_rc_theo_mult.shape) == 4:
        # Repetition got squeezed
        FI_rc_theo_mult = FI_rc_theo_mult[..., np.newaxis]

        # Fix some weird mismatches... HACK
        # M[1] <-> M[2], M[3] <-> M[4], M[5] <-> M[6], M[7] <-> M[8]
        tt = FI_rc_theo_mult[:, :, 1].copy()
        FI_rc_theo_mult[:, :, 1] = FI_rc_theo_mult[:, :, 2]
        FI_rc_theo_mult[:, :, 2] = tt
        tt = FI_rc_theo_mult[:, :, 3].copy()
        FI_rc_theo_mult[:, :, 3] = FI_rc_theo_mult[:, :, 4]
        FI_rc_theo_mult[:, :, 4] = tt
        tt = FI_rc_theo_mult[:, :, 5].copy()
        FI_rc_theo_mult[:, :, 5] = FI_rc_theo_mult[:, :, 6]
        FI_rc_theo_mult[:, :, 6] = tt
        tt = FI_rc_theo_mult[:, :, 7].copy()
        FI_rc_theo_mult[:, :, 7] = FI_rc_theo_mult[:, :, 8]
        FI_rc_theo_mult[:, :, 8] = tt
        tt = FI_rc_theo_mult[:, :, 4].copy()
        FI_rc_theo_mult[:, :, 4] = FI_rc_theo_mult[:, :, 5]
        FI_rc_theo_mult[:, :, 5] = tt
        tt = FI_rc_theo_mult[:, :, 8].copy()
        FI_rc_theo_mult[:, :, 8] = FI_rc_theo_mult[:, :, 9]
        FI_rc_theo_mult[:, :, 9] = tt

    # Only keep valid M^2 values
    M_space_true = []
    M_sqrt_seen = []
    M_sqrt_space = np.floor(M_space**0.5).astype(int)

    for m_i, M_sqrt in enumerate(M_sqrt_space):
        if M_sqrt not in M_sqrt_seen:
            # Not seen it, keep this row
            M_space_true.append(m_i)
            M_sqrt_seen.append(M_sqrt)
    M_space_true = np.array(M_space_true)

    # Filter it
    FI_rc_theo_mult = FI_rc_theo_mult[:, :, M_space_true]
    M_space = M_space[M_space_true]

    data_to_plot = dict(FI_rc_theo_mult=FI_rc_theo_mult, rcscale_space=rcscale_space, sigma_space=sigma_space, M_space=M_space)

    # launchers_memorycurves.plots_memorycurve_theoretical_3dvolume(data_to_plot)


    ## Find undone simulations
    # nanparams = np.nonzero(np.any(np.isnan(np.mean(FI_rc_theo_mult, axis=-1)), axis=-1))
    # list_params_dict = [dict(rc_scale=rcscale_space[par[0]], sigmax=sigma_space[par[1]], M=M_space[par[2]]) for par in zip(*nanparams)]
    # this could be resend on PBS...


def plots_3dvolume_theovar_mixedpop(data_pbs, launcher_variables=None):
    '''
        Reload 3D volume runs from PBS and plot them
    '''

    FI_rc_theo_mean = np.squeeze(nanmean(data_pbs.dict_arrays['FI_rc_theo_mult']['results'], axis=-1))
    FI_rc_var_mean = np.squeeze(np.mean(data_pbs.dict_arrays['FI_rc_var_mult']['results'], axis=-1))


    rcscale_space = data_pbs.loaded_data['parameters_uniques']['rc_scale']
    sigma_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    rcscale2_space = data_pbs.loaded_data['parameters_uniques']['rc_scale2']
    
    
    exp_target = np.array([ 8.81007762,  4.7976755,   3.61554792,  2.4624979,   1.78252039])*2.
    interpolation = 'nearest'

    # 3D
    if False:
        import launchers_memorycurves

        FI_rc_theo_mult = np.squeeze(data_pbs.dict_arrays['FI_rc_theo_mult']['results'])
        FI_rc_var_mult = np.squeeze(data_pbs.dict_arrays['FI_rc_var_mult']['results'])[..., 0]

        # data_to_plot = dict(FI_rc_theo_mult=FI_rc_theo_mult, rcscale_space=rcscale_space, sigma_space=sigma_space, M_space=rcscale2_space)
        # launchers_memorycurves.plots_memorycurve_theoretical_3dvolume(data_to_plot)

        data_to_plot = dict(FI_rc_theo_mult=FI_rc_var_mult, rcscale_space=rcscale_space, sigma_space=sigma_space, M_space=rcscale2_space)
        launchers_memorycurves.plots_memorycurve_theoretical_3dvolume(data_to_plot)

    if True:

        # Check the fits
        dist_theo_exp = np.sum((FI_rc_theo_mean - exp_target)**2., axis=-1)
        log_dist_theo_exp = np.log(dist_theo_exp)

        dist_theo_1obj = (FI_rc_theo_mean[..., 0] - exp_target[0])**2.
        log_dist_theo_1obj = np.log(dist_theo_1obj)
        
        dist_var_exp = np.sum((FI_rc_var_mean[..., 0] - exp_target)**2., axis=-1)
        log_dist_var_exp = np.log(dist_var_exp)

        ratio_theo_var = np.mean((FI_rc_theo_mean/FI_rc_var_mean[..., 0]), axis=-1)

        for dim3_i, dim3 in enumerate(rcscale2_space):
            # Do one pcolor per dim3 (rc_scale2)

            pcolor_2d_data(log_dist_theo_exp[..., dim3_i], x=rcscale_space, y=sigma_space, title='Log distance theory to experimental memory curve. dim3: %.2f' % dim3, label_format="%.2f", interpolation=interpolation)
            pcolor_2d_data(log_dist_var_exp[..., dim3_i], x=rcscale_space, y=sigma_space, title='Log distance variance to experimental memory curve. dim3: %.2f' % dim3, label_format="%.2f", interpolation=interpolation)
            # pcolor_2d_data(log_dist_theo_1obj[..., dim3_i], x=rcscale_space, y=sigma_space, title='Log distance theory 1 object to experimental 1 object. rc_scale2: %.2f' % rcscale2, label_format="%.2f", interpolation=interpolation)
            # pcolor_2d_data(ratio_theo_var[..., dim3_i], x=rcscale_space, y=sigma_space, title='Ratio theo/var. rc_scale2: %.2f' % rcscale2, label_format="%.2f", interpolation=interpolation)

            # Best fits plots
            # Get the list of best values.
            besttheo_mem_argsort = np.argsort(dist_theo_exp[..., dim3_i], axis=None)
            bestvar_mem_argsort = np.argsort(dist_var_exp[..., dim3_i], axis=None)
            besttheo_1obj_argsort = np.argsort(dist_theo_1obj[..., dim3_i], axis=None)

            # Show the results for the best K
            best_k = 1

            for bk_i in np.arange(best_k):
                # Convert the flat best index into indices
                indices_besttheo_mem = np.unravel_index(besttheo_mem_argsort[bk_i], dist_theo_exp[..., dim3_i].shape)
                indices_bestvar_mem = np.unravel_index(bestvar_mem_argsort[bk_i], dist_var_exp[..., dim3_i].shape)
                indices_besttheo_1obj = np.unravel_index(besttheo_1obj_argsort[bk_i], dist_theo_1obj[..., dim3_i].shape)

                # Best memory curve for each individual distance function.
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_mem][dim3_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0], 'o-', linewidth=3)
                current_color = ax.get_lines()[-1].get_c()
                ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0]-FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 1], FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0]+FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 1], facecolor=current_color, alpha=0.4)
                plt.title('%d Best fits for dim3=%.2f: kappa_theo %.2f, sigma_theo %.2f | kappa_var %.2f, sigma_var %.2f' % (bk_i+1, dim3, rcscale_space[indices_besttheo_mem[0]], sigma_space[indices_besttheo_mem[1]], rcscale_space[indices_bestvar_mem[0]], sigma_space[indices_bestvar_mem[1]]))
                plt.legend(['Experimental data', 'best Theoretical FI', 'best Posterior variance'])
                plt.xticks(np.arange(1, 6))
                plt.xlim([0.8, 5.2])

                # Optimize based on FI, full memory curves
                # f = plt.figure()
                # ax = f.add_subplot(111)
                # ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_mem][dim3_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_besttheo_mem][dim3_i, :, 0], 'o-', linewidth=3)
                # current_color = ax.get_lines()[-1].get_c()
                # ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_besttheo_mem][dim3_i, :, 0]-FI_rc_var_mean[indices_besttheo_mem][dim3_i, :, 1], FI_rc_var_mean[indices_besttheo_mem][dim3_i, :, 0]+FI_rc_var_mean[indices_besttheo_mem][dim3_i, :, 1], facecolor=current_color, alpha=0.4)
                # plt.title('%d Best fits for dim3=%.2f: kappa_theo %.2f, sigma_theo %.2f' % (bk_i+1, dim3, rcscale_space[indices_besttheo_mem[0]], sigma_space[indices_besttheo_mem[1]]))
                # plt.legend(['Experimental data', 'best Theoretical FI', 'Posterior variance, theo best'])
                # plt.xticks(np.arange(1, 6))
                # plt.xlim([0.8, 5.2])

                # # Optimize based on FI, 1obj case. Shouldn't be too different from above.
                # f = plt.figure()
                # ax = f.add_subplot(111)
                # ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_theo_mean[indices_besttheo_1obj][dim3_i, :], 'o-', np.arange(1, 6), FI_rc_var_mean[indices_besttheo_1obj][dim3_i, :, 0], 'o-', linewidth=3)
                # current_color = ax.get_lines()[-1].get_c()
                # ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_besttheo_1obj][dim3_i, :, 0]-FI_rc_var_mean[indices_besttheo_1obj][dim3_i, :, 1], FI_rc_var_mean[indices_besttheo_1obj][dim3_i, :, 0]+FI_rc_var_mean[indices_besttheo_1obj][dim3_i, :, 1], facecolor=current_color, alpha=0.4)
                # plt.title('%d Best fits for dim3=%.2f: kappa_1obj %.2f, sigma_1obj %.2f' % (bk_i+1, dim3, rcscale_space[indices_besttheo_1obj[0]], sigma_space[indices_besttheo_1obj[1]]))
                # plt.legend(['Experimental data', 'Theoretical FI, 1obj', 'Posterior variance, theo 1obj'])
                # plt.xticks(np.arange(1, 6))
                # plt.xlim([0.8, 5.2])

                # Best variance to experimental fit only
                f = plt.figure()
                ax = f.add_subplot(111)
                ax.plot(np.arange(1, 6), exp_target, 'o-', np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0], 'o-', linewidth=3)
                # current_color = ax.get_lines()[-1].get_c()
                # ax.fill_between(np.arange(1, 6), FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0]-FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 1], FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 0]+FI_rc_var_mean[indices_bestvar_mem][dim3_i, :, 1], facecolor=current_color, alpha=0.4)
                plt.title('%d Best var fits for dim3=%.2f: kappa_var %.2f, sigma_var %.2f' % (bk_i+1, dim3, rcscale_space[indices_bestvar_mem[0]], sigma_space[indices_bestvar_mem[1]]))
                plt.legend(['Experimental data', 'best Posterior variance'])
                plt.xticks(np.arange(1, 6))
                plt.xlim([0.8, 5.2])








