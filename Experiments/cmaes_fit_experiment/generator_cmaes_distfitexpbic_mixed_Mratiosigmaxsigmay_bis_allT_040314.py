"""
    ExperimentDescriptor to use CMA-ES to fit Memory curves using a Mixed population code

"""

import os
import numpy as np
import experimentlauncher
import inspect
import utils
import dataio
import copy
import submitpbs

# Commit @8c49507 +

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 5
M  = 200
T = 6

run_label = 'cmaes_distfitexpbic_mixed_Mratiosigmaxsigmay_allT_repetitions{num_repetitions}_040314'
simul_out_dir = os.path.join(os.getcwd(), run_label.format(**locals()))

parameter_generation = 'cma-es'
cma_population_size = 30
cma_sigma0 = 1.0
cma_logger_do_plot = True
cma_use_auto_scaling = True
cma_use_bounds = True

sleeping_period = dict(min=10, max=20)

pbs_submission_infos = dict(description='Fit experiments (bays09, gorgo11, dualrecall), using distfitexpbic ResultComputation), using the CMA-ES code. Looks at all t<=T here. Changes M, ratio_conj, sigmax and sigmay. Only looks at BIC score.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fitexperiment_allT',
                                               code_type='mixed',
                                               output_directory='.',
                                               ratio_conj=0.5,
                                               subaction='collect_responses',
                                               session_id='cmaes_ratiosigmax',
                                               result_computation='distfitexpbic',
                                               M=M,
                                               sigmax=0.1,
                                               N=200,
                                               T=T,
                                               sigmay=0.0001,
                                               inference_method='none',
                                               num_samples=300,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               slice_width=0.07,
                                               burn_samples=300,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='1:00:00',
                            memory='2gb',
                            simul_out_dir=simul_out_dir,
                            pbs_submit_cmd=submit_cmd,
                            submit_label='cmaes_4d_bic')


sigmax_range      =   dict(low=0.0000001, high=1.0, dtype=float)
sigmay_range      =   dict(low=0.0000001, high=1.0, dtype=float)
ratioconj_range   =   dict(low=0.0, high=1.0, dtype=float)
M_range           =   dict(low=2, high=500, dtype=int)

dict_parameters_range =   dict(ratio_conj=ratioconj_range, sigmax=sigmax_range, M=M_range, sigmay=sigmay_range)


# result_callback_function to track best parameter
best_parameters_seen = dict(result=np.nan, job_name='', parameters=None, submit_best=True, pbs_submission_infos_copy=copy.deepcopy(pbs_submission_infos))
def best_parameters_callback(job, parameters=None):

    if not np.any(np.isnan(job.get_result())) and (np.any(np.isnan(parameters['result'])) or (job.get_result() <= parameters['result'])):
        # New best parameter!
        parameters['result'] = job.get_result()
        parameters['job_name'] = job.job_name
        parameters['parameters'] = job.experiment_parameters

        print "\n\n>>>>>> Found new best parameters: \n%s\n\n" % parameters

        np.save('./outputs/best_params', dict(parameters=parameters))

        # If desired, automatically create additional plots.
        if parameters.get('submit_best', False):

            pbs_submission_infos_copy = parameters['pbs_submission_infos_copy']
            try:
                # Will check the best fitting parameters, and relaunch simulations for them, in order to get new cool plots.

                pbs_submission_infos_copy['other_options'].update(dict(
                    action_to_do='launcher_do_memory_curve_marginal_fi_withplots',
                    subaction='collect_responses',
                    inference_method='sample',
                    N=300,
                    num_samples=500,
                    output_directory=os.path.join(simul_out_dir, 'outputs'),
                    selection_method='last',
                    num_repetitions=5,
                    burn_samples=300,
                    stimuli_generation='random',
                    stimuli_generation_recall='random',
                    session_id='cmaes_fitting_experiments_relaunchs',
                    result_computation='filenameoutput',
                    label='cmaes_Mratiosigmaxsigmay_fitting_experiment_rerun_040314'))
                pbs_submission_infos_copy['walltime'] = '10:00:00'

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

    # Save data
    if parameters['dataio'] is not None:
        parameters['dataio'].save_variables_default(locals(), ['candidates', 'fitness', 'parameter_names_sorted'])
        parameters['dataio'].make_link_output_to_dropbox(dropbox_current_experiment_folder='cmaes_fit_experiment')

    # Do plots
    # Param 1 vs 2 vs 3
    parameters['ax'] = utils.scatter3d(candidates_arr[:, 0], candidates_arr[:, 1], candidates_arr[:, 2], c=np.log(fitness_arr), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[2], ax_handle=parameters['ax'])
    if parameters['dataio'] is not None:
        parameters['dataio'].save_current_figure('cmaes_optim_landscape_123_scatter3d_{label}_{unique_id}.pdf')

    # Param 1 vs 2 vs 4
    parameters['ax'] = utils.scatter3d(candidates_arr[:, 0], candidates_arr[:, 1], candidates_arr[:, 3], c=np.log(fitness_arr), xlabel=parameter_names_sorted[0], ylabel=parameter_names_sorted[1], zlabel=parameter_names_sorted[3], ax_handle=parameters['ax'])
    if parameters['dataio'] is not None:
        parameters['dataio'].save_current_figure('cmaes_optim_landscape_124_scatter3d_{label}_{unique_id}.pdf')

    # Param 2 vs 3 vs 4
    parameters['ax'] = utils.scatter3d(candidates_arr[:, 1], candidates_arr[:, 2], candidates_arr[:, 3], c=np.log(fitness_arr), xlabel=parameter_names_sorted[1], ylabel=parameter_names_sorted[2], zlabel=parameter_names_sorted[3], ax_handle=parameters['ax'])
    if parameters['dataio'] is not None:
        parameters['dataio'].save_current_figure('cmaes_optim_landscape_234_scatter3d_{label}_{unique_id}.pdf')

cma_iter_callback_function_infos = dict(function=cma_iter_plot_scatter3d_candidates, parameters=cma_iter_parameters)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

