"""
    ExperimentDescriptor to fit the experimental data, using LL and checking if the mixture model
    has similar outputs.
"""

import os
import numpy as np
import experimentlauncher
import inspect
import getpass

# Commit @d8c9acb

# Read from other scripts
parameters_entryscript = dict(
    action_to_do='launcher_do_generate_submit_pbs_from_param_files',
    output_directory='.')
submit_jobs = True

parameter_generation = 'random'  ## !!!!!! RANDOM HERE   !!!!!
num_random_samples = 5000
limit_max_queued_jobs = 70

resource = ''

# partition = 'wrkstn'
# partition = 'test'
# partition = 'intel-ivy'
partition = 'all'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 3
experiment_id = 'bays09'

run_label = 'LL_normcstfixed_random_fitexperimentallt_{experiment_id}_sigmaxMratiobaseline_repetitions{num_repetitions}_080619'

pbs_submission_infos = dict(
    description='''
    Fixed normalisation constant -inf. Check if landscapes are better.

    Loads an experiment, and uses the data for the model, for all T in the dataset. Computes Loglikelihood, BIC and fits the mixture model, and compare the mixture model fits to the experimental data.
    ''',
    command='python $WORKDIR/experimentlauncher.py',
    other_options=dict(
        action_to_do='launcher_do_fitexperiment_allmetrics',
        experiment_id=experiment_id,
        filter_datapoints_size=500,
        filter_datapoints_selection='sequential',
        bic_K=5,
        M=100,
        code_type='mixed',
        ratio_conj=0.5,
        renormalize_sigma=None,
        N=200,
        R=2,
        T=6,
        sigmax=0.1,
        sigmay=0.00001,
        sigma_baseline=0.001,
        sigma_output=0.0,
        lapse_rate=0.0,
        inference_method='sample',
        num_samples=50,
        selection_num_samples=1,
        selection_method='last',
        slice_width=0.07,
        burn_samples=100,
        num_repetitions=num_repetitions,
        enforce_min_distance=0.17,
        specific_stimuli_random_centers=None,
        stimuli_generation='random',
        stimuli_generation_recall='random',
        autoset_parameters=None,
        collect_responses=None,
        label=run_label,
        output_directory='.',
        experiment_data_dir=os.path.normpath(
            os.path.join(os.environ['WORKDIR_DROP'],
                         '../../experimental_data')),
    ),
    walltime='10:00:00',
    memory='2gb',
    simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
    pbs_submit_cmd=submit_cmd,
    limit_max_queued_jobs=limit_max_queued_jobs,
    source_dir=os.environ['WORKDIR_DROP'],
    submit_label='LLnorm_rnd_0806',
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
  M_conj_prior = int(new_parameters['M'] * new_parameters['ratio_conj'])
  M_conj_true = int(np.floor(M_conj_prior**0.5)**2)
  M_feat_true = int(np.floor((new_parameters['M'] - M_conj_prior) / 2.) * 2.)
  M_true = M_conj_true + M_feat_true
  ratio_true = M_conj_true / float(M_true)

  if function_parameters['should_clamp']:
    # Clamp them and return true
    new_parameters['M'] = M_true
    new_parameters['ratio_conj'] = ratio_true

    return True
  else:
    return np.allclose(M_true, new_parameters['M'])


filtering_function_parameters = {'should_clamp': True}

sigmax_range = dict(
    sampling_type='uniform',
    low=0.005,
    high=0.6,
    dtype=float,
)
ratioconj_range = dict(
    sampling_type='uniform',
    low=0.0,
    high=1.0,
    dtype=float,
)
sigmabaseline_range = dict(
    low=0.0,
    high=1.0,
    sampling_type='uniform',
    dtype=float,
)
M_range = dict(
    sampling_type='randint',
    low=6,
    high=625,
    dtype=int,
)

dict_parameters_range = dict(
    M=M_range,
    sigma_baseline=sigmabaseline_range,
    ratio_conj=ratioconj_range,
    sigmax=sigmax_range)

if __name__ == '__main__':

  this_file = inspect.getfile(inspect.currentframe())
  print "Running ", this_file

  arguments_dict = dict(parameters_filename=this_file)
  arguments_dict.update(parameters_entryscript)

  experiment_launcher = experimentlauncher.ExperimentLauncher(
      run=True, arguments_dict=arguments_dict)
