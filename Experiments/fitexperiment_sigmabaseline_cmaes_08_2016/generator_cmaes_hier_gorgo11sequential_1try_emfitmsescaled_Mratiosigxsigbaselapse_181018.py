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
# from functools import partial

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

run_label = 'cmaes_hier_gorgo11sequential_1try_emfitmsescaled_Mratiosigxsigbaselapsealpha_rep{num_repetitions}_181018'
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

sleeping_period = dict(min=1, max=5)

pbs_submission_infos = dict(
    description=
    '''Fit sequential experiment (gorgo11 sequential), using dist_collapsedemfit_gorgo11seq ResultComputation), using the CMA-ES code. Now with sigma_baseline instead of sigma_output. Using new fixed Covariance matrix for Sampler, should change N=1 case most.

    HIERARCHICAL CODE. Trying with only layer 2 now, to see what happens.

    Changes M, ratio_conj, sigmax, sigma baseline, lapse rate.
    Looks at all subjects, T and trecall.

    Combine all data across subjects here.

    This uses the new MSE of EM Fit, where we cheat a bit by normalising the Kappa by the max Kappa from the data.
    This directly put Kappa back over the same dynamic range as the mixture proportions. Hopefully works...
    ''',
    command='python $WORKDIR/experimentlauncher.py',
    other_options=dict(
        action_to_do='launcher_do_fitexperiment_sequential_allmetrics',
        code_type='hierarchical',
        output_directory='.',
        experiment_id='gorgo11_sequential',
        bic_K=5,
        type_layer_one='feature',
        output_both_layers=None,
        normalise_weights=1,
        ratio_hierarchical=0.5,
        normalise_gain=None,
        threshold=1.0,
        session_id='cmaes_hier_1try_Mratiosigxlrsigbase_gorgo11seq',
        result_computation='dist_collapsedemfit_gorgo11seq',
        M=100,
        sigmax=0.1,
        renormalize_sigma=None,
        N=500,
        T=1,
        alpha=1,
        sigmay=0.00001,
        sigma_baseline=0.001,
        sigma_output=0.0,
        lapse_rate=0.0,
        inference_method='sample',
        num_samples=100,
        selection_num_samples=1,
        selection_method='last',
        slice_width=0.07,
        burn_samples=20,
        num_repetitions=num_repetitions,
        enforce_min_distance=0.17,
        specific_stimuli_random_centers=None,
        stimuli_generation='random',
        stimuli_generation_recall='random',
        autoset_parameters=None,
        label=run_label,
        experiment_data_dir=os.path.normpath(
            os.path.join(os.environ['WORKDIR_DROP'],
                         '../../experimental_data')),
    ),
    walltime='1:00:00',
    memory='2gb',
    simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
    pbs_submit_cmd=submit_cmd,
    source_dir=os.environ['WORKDIR_DROP'],
    submit_label='cma_h_gorgoseq_1',
    resource=resource,
    partition=partition,
    qos='auto')


## Define our filtering function
def filtering_function(new_parameters,
                       dict_parameters_range,
                       function_parameters=None):
  '''
    Given M and ratio_conj, will adapt them so that M_conj is always correct and integer.

    or if should_clamp is False, will not change them
    '''
  M_layer_two = int(
      np.round(new_parameters['ratio_hierarchical'] * new_parameters['M']))
  M_layer_one = new_parameters['M'] - M_layer_two

  M_layer_one_fixed = int(M_layer_one / 2) * 2
  M_layer_two_fixed = M_layer_one - M_layer_one_fixed + M_layer_two
  ratio_correct = M_layer_two_fixed / float(new_parameters['M'])

  if function_parameters['should_clamp']:
    # Clamp and return true
    new_parameters['ratio_hierarchical'] = ratio_correct

  return True


filtering_function_parameters = {'should_clamp': True}


# ============================================================================
sigmax_range = dict(
    low=0.005,
    high=1.,
    x0=0.2,
    scaling=cma_sigma0 / 3.,
    dtype=float,
    transform_fct=utils.tsfr_square,
    transform_inv_fct=utils.tsfr_square_inv)
sigmabaseline_range = dict(
    low=0.0,
    high=1.,
    x0=0.2,
    scaling=cma_sigma0 / 3.,
    dtype=float,
    transform_fct=utils.tsfr_square,
    transform_inv_fct=utils.tsfr_square_inv)
ratiohier_range = dict(
    low=0.0,
    high=1.0,
    x0=0.5,
    scaling=cma_sigma0 / 3.,
    dtype=float,
)
lapserate_range = dict(
    low=0.0,
    high=0.1,
    x0=0.05,
    scaling=cma_sigma0 / 10.,
    dtype=float,
    transform_fct=utils.tsfr_square,
    transform_inv_fct=utils.tsfr_square_inv)
M_range = dict(low=20, high=400, dtype=int)
alpha_range = dict(
    low=0.5, high=1.0, x0=0.8, scaling=cma_sigma0 / 6., dtype=float)

dict_parameters_range = dict(
    M=M_range,
    lapse_rate=lapserate_range,
    ratio_hierarchical=ratiohier_range,
    sigmax=sigmax_range,
    sigma_baseline=sigmabaseline_range,
    alpha=alpha_range)

# ============================================================================


# result_callback_function to track best parameter
best_parameters_seen = dict(result=np.nan, job_name='', parameters=None, submit_best=False, pbs_submission_infos_copy=copy.deepcopy(pbs_submission_infos))
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
def cma_iter_store(all_variables, parameters=None):
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

  # Do plot
  # if parameters['ax'] is None:
  #   _, parameters['ax'] = plt.subplots(2, 1)

  # parameters['ax'][0].plot(time_space, candidates_arr)
  # parameters['ax'][0].set_xlabel('Time')
  # parameters['ax'][0].set_ylabel('Parameters')
  # parameters['ax'][0].legend(parameter_names_sorted)

  # parameters['ax'][1].plot(time_space, fitness_arr, label='NLL92')
  # parameters['ax'][1].set_xlabel('Time')
  # parameters['ax'][1].set_ylabel('NLL92')

  # if parameters['dataio'] is not None:
  #     parameters['dataio'].save_current_figure('cmaes_optim_timeevolution_{label}_{unique_id}.pdf')


cma_iter_callback_function_infos = dict(
  function=cma_iter_store, parameters=cma_iter_parameters)


if __name__ == '__main__':

  this_file = inspect.getfile(inspect.currentframe())
  print "Running ", this_file

  arguments_dict = dict(parameters_filename=this_file)
  arguments_dict.update(parameters_entryscript)

  experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)
