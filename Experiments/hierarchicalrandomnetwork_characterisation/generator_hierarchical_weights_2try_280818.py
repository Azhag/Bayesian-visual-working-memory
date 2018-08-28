"""
    Random sweeps for hierarchical characterisation
"""

import os
import numpy as np
import experimentlauncher
import inspect
import utils

# Read from other scripts
parameters_entryscript = dict(
    action_to_do='launcher_do_generate_submit_pbs_from_param_files',
    output_directory='.')
submit_jobs = True

parameter_generation = 'random'
num_random_samples = 10000
limit_max_queued_jobs = 150

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 10

run_label = 'hierarchical_weights_ptheta_2try_repetitions{num_repetitions}_280818'
sleeping_period = dict(min=10, max=30)

pbs_submission_infos = dict(
    description=
    '''Random sweeps for Thesis, chapter about Single item characterisation.
    HIERARCHICAL CODE.

    low-sparsity regime, where high threshold seems to make super efficient
    populations?

    sigma_weights is useless, vary sparsity and threshold.
    ''',
    command='python $WORKDIR/experimentlauncher.py',
    other_options=dict(
        action_to_do='launcher_check_fisher_fit_1obj_2016',
        code_type='hierarchical',
        output_directory='.',
        type_layer_one='feature',
        normalise_weights=1,
        output_both_layers=None,
        ratio_hierarchical=0.5,
        threshold=1.0,
        bic_K=5,
        session_id='hier_2_ptheta',
        M=100,
        sigmax=0.3,
        renormalize_sigma=None,
        N=200,
        T=1,
        sigmay=0.000001,
        sigma_baseline=0.0,
        sigma_output=0.0,
        lapse_rate=0.0,
        inference_method='sample',
        num_samples=100,
        selection_num_samples=1,
        selection_method='last',
        slice_width=0.07,
        burn_samples=50,
        num_repetitions=num_repetitions,
        enforce_min_distance=0.000000001,
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
    limit_max_queued_jobs=limit_max_queued_jobs,
    source_dir=os.environ['WORKDIR_DROP'],
    submit_label='hier_weights_2try',
    resource=resource,
    partition=partition,
    qos='auto')


# ## Define our filtering function
# def filtering_function(new_parameters,
#                        dict_parameters_range,
#                        function_parameters=None):
#   '''
#     Given M and ratio_conj, will adapt them so that M_conj is always correct and integer.

#     or if should_clamp is False, will not change them
#     '''
#   M_layer_two = int(
#       np.round(new_parameters['ratio_hierarchical'] * new_parameters['M']))
#   M_layer_one = new_parameters['M'] - M_layer_two

#   M_layer_one_fixed = int(M_layer_one / 2) * 2
#   M_layer_two_fixed = M_layer_one - M_layer_one_fixed + M_layer_two
#   ratio_correct = M_layer_two_fixed / float(new_parameters['M'])

#   if function_parameters['should_clamp']:
#     # Clamp and return true
#     new_parameters['ratio_hierarchical'] = ratio_correct

#   return True

# filtering_function_parameters = {'should_clamp': True}

dict_parameters_range = {
    'threshold': dict(sampling_type='uniform', low=0., high=10., dtype=float),
    'sparsity': dict(sampling_type='uniform', low=0., high=0.15, dtype=float),
}

if __name__ == '__main__':

  this_file = inspect.getfile(inspect.currentframe())
  print "Running ", this_file

  arguments_dict = dict(parameters_filename=this_file)
  arguments_dict.update(parameters_entryscript)

  experiment_launcher = experimentlauncher.ExperimentLauncher(
      run=True, arguments_dict=arguments_dict)
