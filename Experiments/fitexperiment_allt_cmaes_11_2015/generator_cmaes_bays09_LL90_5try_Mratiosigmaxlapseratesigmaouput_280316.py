"""
    CMA/ES on new FitExperimentAllT
"""

import os
import numpy as np
import experimentlauncher
import inspect
import utils
import dataio
import copy
import submitpbs

import matplotlib.pyplot as plt

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'


num_repetitions = 5
experiment_id = 'bays09'

run_label = 'cmaes_bays09_ll90_5try_Mratiosigmaxlapseratesigmaout_repetitions{num_repetitions}_280316'
simul_out_dir = os.path.join(os.getcwd(), run_label.format(**locals()))

parameter_generation = 'cma-es'
cma_population_size = 20
cma_sigma0 = 1.0
cma_logger_do_plot = True
cma_use_auto_scaling = True
cma_use_bounds = True

sleeping_period = dict(min=10, max=20)

pbs_submission_infos = dict(description='Fit experiments (bays09), using dist_ll90_allt ResultComputation), using the CMA-ES code. Looks at all t<=T here. Changes M, ratio_conj and sigmax. Only looks at LL score, top 90 percent. Looking for best meaningful parameter bounds now.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fitexperiment_allmetrics',
                                               code_type='mixed',
                                               output_directory='.',
                                               experiment_id=experiment_id,
                                               bic_K=5,
                                               ratio_conj=0.5,
                                               session_id='cmaes_5try_Mratiosigmaxlapsesigmaout_bays09',
                                               result_computation='dist_ll90_allt',
                                               M=100,
                                               sigmax=0.1,
                                               renormalize_sigmax=None,
                                               N=200,
                                               T=1,
                                               sigmay=0.0001,
                                               sigma_output=0.0,
                                               lapse_rate=0.0,
                                               inference_method='none',
                                               num_samples=200,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               slice_width=0.07,
                                               burn_samples=200,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               collect_responses=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='1:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            source_dir=os.environ['WORKDIR_DROP'],
                            submit_label='cmaes_5try_bays',
                            resource=resource,
                            partition=partition,
                            qos='auto')



## Define our filtering function
def filtering_function(new_parameters, dict_parameters_range, function_parameters=None):
    '''
    Given M and ratio_conj, will adapt them so that M_conj is always correct and integer.

    or if should_clamp is False, will not change them
    '''

    M_conj_prior = int(new_parameters['M']*new_parameters['ratio_conj'])
    M_conj_true = int(np.floor(M_conj_prior**0.5)**2)
    M_feat_true = int(np.floor((new_parameters['M']-M_conj_prior)/2.)*2.)
    M_true = M_conj_true + M_feat_true
    ratio_true = M_conj_true/float(M_true)

    if function_parameters['should_clamp']:
        # Clamp them and return true
        new_parameters['M'] = M_true
        new_parameters['ratio_conj'] = ratio_true

        return True
    else:
        return np.allclose(M_true, new_parameters['M'])

filtering_function_parameters = {'should_clamp': True}

sigmax_range      =   dict(sampling_type='uniform', low=0.1, high=0.8, dtype=float)
ratioconj_range   =   dict(sampling_type='uniform', low=0.3, high=1.0, dtype=float)
lapserate_range   =   dict(sampling_type='uniform', low=0.0, high=0.15, dtype=float)
sigmaoutput_range =   dict(sampling_type='uniform', low=0.1, high=0.7, dtype=float)
M_range           =   dict(sampling_type='randint', low=6, high=400, dtype=int)


dict_parameters_range =   dict(M=M_range, lapse_rate=lapserate_range, ratio_conj=ratioconj_range, sigmax=sigmax_range, sigma_output=sigmaoutput_range)

# result_callback_function to track best parameter
best_parameters_seen = dict(result=np.nan, job_name='', parameters=None, submit_best=True, pbs_submission_infos_copy=copy.deepcopy(pbs_submission_infos))
def best_parameters_callback(job, parameters=None):

    if not np.any(np.isnan(job.get_result())) and (np.any(np.isnan(parameters['result'])) or (job.get_result() <= parameters['result'])):
        # New best parameter!
        parameters['result'] = job.get_result()
        parameters['job_name'] = job.job_name
        parameters['parameters'] = job.experiment_parameters
        parameters['best_parameters'] = utils.subdict(job.experiment_parameters, dict_parameters_range.keys())

        print "\n\n>>>>>> Found new best parameters: \n%s %s %s\n\n" % (parameters['best_parameters'], parameters['result'], parameters['job_name'])

        np.save('./outputs/best_params', dict(parameters=parameters))

        # If desired, automatically create additional plots.
        if parameters.get('submit_best', False):

            pbs_submission_infos_copy = parameters['pbs_submission_infos_copy']
            try:
                # Will check the best fitting parameters, and relaunch simulations for them, in order to get new cool plots.

                ## First do Memory curves + EM Fits
                pbs_submission_infos_copy['other_options'].update(dict(
                    action_to_do='launcher_do_memory_curve_marginal_fi_withplots_live',
                    subaction='collect_responses',
                    inference_method='sample',
                    N=300,
                    T=6,
                    num_samples=300,
                    output_directory=os.path.join(simul_out_dir, 'outputs'),
                    selection_method='last',
                    num_repetitions=10,
                    burn_samples=200,
                    stimuli_generation='random',
                    stimuli_generation_recall='random',
                    session_id='cmaes_bays09_5try_rerun_280316',
                    result_computation='filenameoutput',
                    label='lapse%.2fsigmaout%.2f_cmaes_bays09_5try_280316' % (parameters['parameters']['lapse_rate'], parameters['parameters']['sigma_output'])))
                pbs_submission_infos_copy['walltime'] = '40:00:00'
                pbs_submission_infos_copy['submit_label'] = 'bestparam_rerun'

                submit_pbs = submitpbs.SubmitPBS(pbs_submission_infos=pbs_submission_infos_copy, debug=True)

                # Extract the parameters to try
                best_params_resend = [utils.subdict(job.experiment_parameters, dict_parameters_range.keys())]

                # Submit without waiting
                print "Submitting extra job for Plots, parameters:", best_params_resend
                submission_parameters_dict = dict(pbs_submission_infos=pbs_submission_infos_copy, submit_jobs=submit_jobs, wait_jobs_completed=False)
                submit_pbs.submit_minibatch_jobswrapper(best_params_resend, submission_parameters_dict)

            except Exception as e:
                print "Failure while submitting sub-task for best parameter. Continuing anyway."
                print parameters
                print e


result_callback_function_infos = dict(function=best_parameters_callback, parameters=best_parameters_seen)

# Callback after each iteration, let's save all candidates/fitness, and do a contour plot
data_io = dataio.DataIO(label='cmaes_alliter_tracking', output_folder=os.path.join(simul_out_dir, 'outputs'))

cma_iter_parameters = dict(ax=None, candidates=[], fitness=[], dataio=data_io)
def cma_iter_plot_scatter3d_candidates(all_variables, parameters=None):
    candidates = parameters['candidates']
    fitness = parameters['fitness']

    # Take candidates and fitness and append them
    candidates.extend(all_variables['parameters_candidates_array'])
    fitness.extend(all_variables['fitness_results'].tolist())

    candidates_arr = np.array(candidates)
    fitness_arr = np.array(fitness)
    parameter_names_sorted = all_variables['parameter_names_sorted']
    time_space = np.arange(candidates_arr.shape[0])

    # Save data
    if parameters['dataio'] is not None:
        parameters['dataio'].save_variables_default(locals(), ['candidates', 'fitness', 'parameter_names_sorted'])
        parameters['dataio'].make_link_output_to_dropbox(dropbox_current_experiment_folder='fitexperiment_allt_cmaes_11_2015')

    # Do plot
    # if parameters['ax'] is None:
    #   _, parameters['ax'] = plt.subplots(2, 1)

    # parameters['ax'][0].plot(time_space, candidates_arr)
    # parameters['ax'][0].set_xlabel('Time')
    # parameters['ax'][0].set_ylabel('Parameters')
    # parameters['ax'][0].legend(parameter_names_sorted)

    # parameters['ax'][1].plot(time_space, fitness_arr, label='NLL90')
    # parameters['ax'][1].set_xlabel('Time')
    # parameters['ax'][1].set_ylabel('NLL90')

    # if parameters['dataio'] is not None:
    #     parameters['dataio'].save_current_figure('cmaes_optim_timeevolution_{label}_{unique_id}.pdf')


cma_iter_callback_function_infos = dict(function=cma_iter_plot_scatter3d_candidates, parameters=cma_iter_parameters)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

