"""
    CMA/ES on new FitExperimentAllT
"""

import os
import numpy as np
import experimentlauncher
import inspect
import utils
import dataio

# Commit > @d0d5ff8f2

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'


num_repetitions = 3
experiment_id = 'bays09'

run_label = 'cmaes_bays09_ll92_8try_Mratiosigmax_repetitions{num_repetitions}_220816'
simul_out_dir = os.path.join(os.getcwd(), run_label.format(**locals()))

parameter_generation = 'cma-es'
cma_sigma0 = 1.0
cma_logger_do_plot = True
cma_use_bounds = True
cma_use_auto_scaling = True
cma_use_transforms = True
cma_tolfun = 1e-3
cma_population_size = 'auto_10x'
cma_boundary_handling = 'BoundPenalty'

sleeping_period = dict(min=5, max=15)

pbs_submission_infos = dict(description='''Fit experiments (bays09), using dist_ll92_allt ResultComputation), using the CMA-ES code. Now with sigma_baseline instead of sigma_output. Using new fixed Covariance matrix for Sampler, should change N=1 case most.

  Changes M, ratio_conj, sigmax, sigma baseline.
  Looks at all t<=T. Use full LL score, all datapoints.

  Small try at reproducing fits from paper, removing all non-original parameters.
  ''',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fitexperiment_allmetrics',
                                               code_type='mixed',
                                               output_directory='.',
                                               experiment_id=experiment_id,
                                               bic_K=5,
                                               ratio_conj=0.5,
                                               session_id='cmaes_8try_Mratiosigmax_bays09',
                                               result_computation='dist_ll92_allt',
                                               M=100,
                                               sigmax=0.1,
                                               renormalize_sigma=None,
                                               N=200,
                                               T=1,
                                               sigmay=0.00001,
                                               sigma_baseline=0.00001,
                                               sigma_output=0.0,
                                               lapse_rate=0.0,
                                               inference_method='none',
                                               num_samples=100,
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
                            submit_label='cmaes_8try_bays',
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


# ============================================================================
sigmax_range = dict(low=0.05,
                    high=0.6,
                    x0=0.3,
                    scaling=cma_sigma0/3.,
                    dtype=float,
                    transform_fct=utils.tsfr_square,
                    transform_inv_fct=utils.tsfr_square_inv
                    )
ratioconj_range = dict(low=0.0,
                       high=1.0,
                       x0=0.5,
                       scaling=cma_sigma0/3.,
                       dtype=float,
                       )
M_range = dict(low=6,
               high=400,
               dtype=int
               )


dict_parameters_range = dict(M=M_range,
                             ratio_conj=ratioconj_range,
                             sigmax=sigmax_range
                             )
# ============================================================================




# result_callback_function to track best parameter
best_parameters_seen = dict(result=np.nan, job_name='', parameters=None)
def best_parameters_callback(job, parameters=None):

    if not np.any(np.isnan(job.get_result())) and (np.any(np.isnan(parameters['result'])) or (job.get_result() <= parameters['result'])):
        # New best parameter!
        parameters['result'] = job.get_result()
        parameters['job_name'] = job.job_name
        parameters['parameters'] = job.experiment_parameters
        parameters['best_parameters'] = utils.subdict(job.experiment_parameters, dict_parameters_range.keys())

        print "\n\n>>>>>> Found new best parameters: \n%s %s %s\n\n" % (parameters['best_parameters'], parameters['result'], parameters['job_name'])
        parameters['best_parameters']

        np.save('./outputs/best_params', dict(parameters=parameters))


result_callback_function_infos = dict(function=best_parameters_callback, parameters=best_parameters_seen)

# Callback after each iteration, let's save all candidates/fitness, and do a contour plot
data_io = dataio.DataIO(label='cmaes_alliter_tracking', output_folder=os.path.join(simul_out_dir, 'outputs'))

cma_iter_parameters = dict(ax=None, candidates=[], fitness=[], dataio=data_io)
def cma_iter_processing(all_variables, parameters=None):
    print "\n\n !!! CMA/ES CALLBACK  !!! \n\n"
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
        parameters['dataio'].make_link_output_to_dropbox(dropbox_current_experiment_folder='fitexperiment_sigmabaseline_cmaes_08_2016')


cma_iter_callback_function_infos = dict(function=cma_iter_processing, parameters=cma_iter_parameters)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

