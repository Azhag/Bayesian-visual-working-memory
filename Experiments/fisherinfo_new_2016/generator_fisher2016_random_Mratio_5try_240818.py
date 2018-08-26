import os
import numpy as np
import experimentlauncher
import inspect
import utils

# Commit @d8c9acb

# Read from other scripts
parameters_entryscript = dict(
    action_to_do='launcher_do_generate_submit_pbs_from_param_files',
    output_directory='.')
submit_jobs = True

parameter_generation = 'random'
num_random_samples = 5000
limit_max_queued_jobs = 150

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 10

run_label = 'fisher2016_random_Mratio_5try_fixed_repetitions{num_repetitions}_240818'
sleeping_period = dict(min=10, max=30)

pbs_submission_infos = dict(
    description=
    'New latest 2018 sweep, to get Figure 4.1 and 4.2 for Thesis, showing Fisher info and other metrics comparisons + the 2D plot of FI/kappa ratio for varying M and Ratio_conj.',
    command='python $WORKDIR/experimentlauncher.py',
    other_options=dict(
        action_to_do='launcher_check_fisher_fit_1obj_2016',
        code_type='mixed',
        output_directory='.',
        ratio_conj=1,
        autoset_parameters=None,
        M=14 * 14,
        sigmax=0.25,
        renormalize_sigma=None,
        N=300,
        R=2,
        T=1,
        sigmay=0.000001,
        sigma_output=0.0,
        sigma_baseline=0.0,
        inference_method='sample',
        num_samples=50,
        selection_num_samples=1,
        selection_method='last',
        slice_width=0.07,
        burn_samples=50,
        num_repetitions=num_repetitions,
        # enforce_min_distance=0.17,
        enforce_min_distance=0.000000001,
        specific_stimuli_random_centers=None,
        stimuli_generation='random_fixed_first',
        stimuli_generation_recall='random',
        rc_scale=1.0,
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
    submit_label='fisher_5try',
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

  M_true, ratio_true = utils.fix_M_ratioconj(new_parameters['M'],
                                             new_parameters['ratio_conj'])

  if function_parameters['should_clamp']:
    # Clamp them and return true
    new_parameters['M'] = M_true
    new_parameters['ratio_conj'] = ratio_true

    return True
  else:
    return np.allclose(M_true, new_parameters['M'])


filtering_function_parameters = {'should_clamp': True}

dict_parameters_range = {
    'M': dict(sampling_type='randint', low=6, high=625, dtype=int),
    'ratio_conj': dict(sampling_type='uniform', low=0.01, high=1., dtype=float),
}

if __name__ == '__main__':

  this_file = inspect.getfile(inspect.currentframe())
  print "Running ", this_file

  arguments_dict = dict(parameters_filename=this_file)
  arguments_dict.update(parameters_entryscript)

  experiment_launcher = experimentlauncher.ExperimentLauncher(
      run=True, arguments_dict=arguments_dict)
