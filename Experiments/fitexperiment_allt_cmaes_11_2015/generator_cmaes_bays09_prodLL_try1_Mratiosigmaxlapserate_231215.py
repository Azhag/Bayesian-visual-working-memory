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
# submit_cmd = 'sh'

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# For this computation, we need to have LL > 0 \forall t.
# so we keep track of the smallest LL we've seen and if it goes below 0, than we subtract it (i.e. add its absolute value)
cma_iter_parameters = dict(minLL=-30000, candidates=[], fitness=[], steps=0)

num_repetitions = 5
experiment_id = 'bays09'

run_label = 'cmaes_bays09_prodLL_try1_Mratiosigmaxlapserate_repetitions{num_repetitions}_231215'
dropbox_current_experiment_folder='fitexperiment_allt_cmaes_11_2015'

simul_out_dir = os.path.join(os.getcwd(), run_label.format(**locals()))

parameter_generation = 'cma-es'
cma_population_size = 20
cma_sigma0 = 1.0
cma_logger_do_plot = True
cma_use_auto_scaling = True
cma_use_bounds = True

sleeping_period = dict(min=10, max=20)

pbs_submission_infos = dict(description='Fit experiments (bays09), using distfit_bays09_prodll ResultComputation), using the CMA-ES code. Looks at all t<=T here. Changes M, ratio_conj and sigmax. Combines LL scores between nitems with geometric mean.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fitexperiment_allmetrics',
                                               code_type='mixed',
                                               output_directory='.',
                                               experiment_id=experiment_id,
                                               bic_K=4,
                                               ratio_conj=0.5,
                                               session_id='cmaes_Mratiosigmaxlapserate_bays09',
                                               result_computation='dist_prodll_allt',
                                               shiftMinLL=cma_iter_parameters['minLL'],
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
                            submit_label='cmaes_ll90_bays',
                            resource=resource,
                            partition=partition,
                            qos='auto')


sigmax_range      =   dict(sampling_type='uniform', low=0.1, high=0.8, dtype=float)
ratioconj_range   =   dict(sampling_type='uniform', low=0.01, high=1.0, dtype=float)
lapserate_range   =   dict(sampling_type='uniform', low=0.0, high=0.15, dtype=float)
M_range           =   dict(sampling_type='randint', low=6, high=625, dtype=int)


dict_parameters_range =   dict(M=M_range, lapse_rate=lapserate_range, ratio_conj=ratioconj_range, sigmax=sigmax_range)

# result_callback_function to track best parameter
best_parameters_seen = dict(result=np.nan, job_name='', parameters=None, submit_best=True, pbs_submission_infos_copy=copy.deepcopy(pbs_submission_infos))
def best_parameters_callback(job, parameters=None):

    if not np.any(np.isnan(job.get_result())) and (np.any(np.isnan(parameters['result'])) or (job.get_result()[0] <= parameters['result'])):
        # New best parameter!
        parameters['result'] = job.get_result()[0]
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
                    session_id='cmaes_bays09_prodLL_summarystats_rerun_231215',
                    result_computation='filenameoutput',
                    label='%d_M%dratio%.2fsx%.2flapse%.2f_cmaes_bays09_prodLL_summarystats_rerun_231215' % (cma_iter_parameters['steps'], parameters['parameters']['M'], parameters['parameters']['ratio_conj'], parameters['parameters']['sigmax'], parameters['parameters']['lapse_rate'])))
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
cma_iter_parameters['dataio'] = dataio.DataIO(label='cmaes_alliter_tracking', output_folder=os.path.join(simul_out_dir, 'outputs'))

## Callback after CMA/ES iteration
# need to track the min LL to have LL always positive throughout. This is then enforced by the ResultComputation
def cma_iter_track_min_ll(all_variables, parameters=None):
    # Second column of fitness_results contain minLL.
    currMin = np.min(all_variables['fitness_results'][:, 1])
    if currMin < parameters['minLL']:
        parameters['minLL'] = currMin
        # if different runs had different minLL, try to compensate for it.
        # (this is wrong for geometric mean... but let's try anyway)
        all_variables['fitness_results'][:, 0] -= (all_variables['fitness_results'][:, 1] - parameters['minLL'])
        print '=====>>> New minLL: ', parameters['minLL']

    # Only keep the first column
    fitness_results=all_variables['fitness_results'][:, 0].copy()

    # Save everything to disk
    parameters['candidates'].extend(all_variables['parameters_candidates_array'])
    parameters['fitness'].extend(fitness_results.tolist())
    parameters['parameter_names_sorted'] = all_variables['parameter_names_sorted']
    parameters['steps'] += 1

    if parameters['dataio'] is not None:
        parameters['dataio'].save_variables_default(parameters, ['candidates', 'fitness', 'parameter_names_sorted'])
        parameters['dataio'].make_link_output_to_dropbox(dropbox_current_experiment_folder=dropbox_current_experiment_folder)

    return dict(fitness_results=fitness_results)


cma_iter_callback_function_infos = dict(function=cma_iter_track_min_ll, parameters=cma_iter_parameters)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

