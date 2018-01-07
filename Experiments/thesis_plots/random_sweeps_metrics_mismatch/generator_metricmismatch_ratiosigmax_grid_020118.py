"""
    ExperimentDescriptor to fit the experimental data, using the Mixture models only.

    Spawns random samples for now, let's see.
"""

import os
import numpy as np
import experimentlauncher
import inspect

# Commit @8c49507 +

# Read from other scripts
parameters_entryscript = dict(
    action_to_do='launcher_do_generate_submit_pbs_from_param_files',
    output_directory='.')
submit_jobs = True

parameter_generation = 'grid'
limit_max_queued_jobs = 90

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

M = 100.
num_repetitions = 3
experiment_id = 'bays09'

run_label = 'metricmismatch_ratiosigmax_grid_1try_repetitions{num_repetitions}_020118'
sleeping_period = dict(min=1, max=5)

pbs_submission_infos = dict(
    description=
    'Runs the model for 1..T items. Computes precision, Fisher information, fits the mixture model, and compare the mixture model fits to the experimental data (Bays09 and Gorgo11 here). Also stores all responses. Meant to run random sampling for a long while! Hierarchical population code.',
    command='python $WORKDIR/experimentlauncher.py',
    other_options=dict(
        action_to_do='launcher_do_fitexperiment_allmetrics',
        code_type='mixed',
        output_directory='.',
        experiment_id=experiment_id,
        bic_K=4,
        ratio_conj=0.5,
        session_id='metricmismatch_ratiosigmax',
        M=M,
        sigmax=0.1,
        renormalize_sigma=None,
        N=200,
        T=1,
        filter_datapoints_size=500,
        filter_datapoints_selection='sequential',
        sigmay=0.00001,
        sigma_baseline=0.1,
        sigma_output=0.0,
        lapse_rate=0.0,
        inference_method='sample',
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
    submit_label='metric_0118',
    resource=resource,
    partition=partition,
    qos='auto')

dict_parameters_range = dict(
    ratio_conj=dict(range=np.linspace(0.0001, 1.0, 50), dtype=float),
    sigmax=dict(range=np.linspace(0.0001, 0.3, 50), dtype=float))


## Define our filtering function
def filtering_function(new_parameters,
                       dict_parameters_range,
                       function_parameters=None):
    '''
    Given M and ratio_conj, will adapt them so that M_conj is always correct and integer.

    or if should_clamp is False, will not change them
    '''
    new_parameters['M'] = M
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

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(
        run=True, arguments_dict=arguments_dict)
