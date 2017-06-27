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
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True

parameter_generation = 'random'
num_random_samples = 10000
limit_max_queued_jobs = 90

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 3

run_label = 'fit_mixturemodels_hier_sigmaxMratio_random_repetitions{num_repetitions}_270617'
sleeping_period = dict(min=20, max=50)

pbs_submission_infos = dict(description='Runs the model for 1..T items. Computes precision, Fisher information, fits the mixture model, and compare the mixture model fits to the experimental data (Bays09 and Gorgo11 here). Also stores all responses. Meant to run random sampling for a long while! Hierarchical population code.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fit_mixturemodels',
                                               code_type='hierarchical',
                                               type_layer_one='feature',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               output_directory='.',
                                               ratio_hierarchical=0.5,
                                               M=100,
                                               N=200,
                                               T=1,
                                               sigmax=0.1,
                                               sigmay=0.0001,
                                               sigma_output=0.0,
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
                                               renormalize_sigma=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            source_dir=os.environ['WORKDIR_DROP'],
                            submit_label='fitmixt_hier_2706',
                            resource=resource,
                            partition=partition,
                            qos='auto')


dict_parameters_range = dict(
    M=dict(sampling_type='randint', dtype=int,
           low=10, high=625),
    ratio_hierarchical=dict(sampling_type='uniform', dtype=float,
                            low=0.01, high=1.0),
    sigmax=dict(sampling_type='uniform', dtype=float,
                low=0.01, high=1.0)
)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

