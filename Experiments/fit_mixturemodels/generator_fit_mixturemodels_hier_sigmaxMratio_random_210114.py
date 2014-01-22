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

parameter_generation = 'random'  ## !!!!!! RANDOM HERE   !!!!!
num_random_samples = 1000
limit_max_queued_jobs = 300

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 3

run_label = 'fit_mixturemodels_hier_sigmaxMratio_random_repetitions{num_repetitions}_210114'

pbs_submission_infos = dict(description='Runs the model for 1..T items. Computes precision, Fisher information, fits the mixture model, and compare the mixture model fits to the experimental data (Bays09 and Gorgo11 here). Also stores all responses. Meant to run random sampling for a long while! Hierarchical population code.',
                            command='python /nfs/home2/lmatthey/Documents/work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fit_mixturemodels',
                                               code_type='hierarchical',
                                               output_directory='.',
                                               ratio_hierarchical=0.5,
                                               M=100,
                                               sigmax=0.1,
                                               N=200,
                                               T=6,
                                               sigmay=0.0001,
                                               inference_method='sample',
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
                                               collect_responses=None,
                                               label=run_label,
                                               experiment_data_dir='/nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data',
                                               ),
                            walltime='40:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            submit_label='fitmixturemodel_rnd')


sigmax_range      =   dict(sampling_type='uniform', low=0.01, high=1.0, dtype=float)
ratiohier_range   =   dict(sampling_type='uniform', low=0.01, high=1.0, dtype=float)
M_range           =   dict(sampling_type='randint', low=10, high=625, dtype=int)

dict_parameters_range =   dict(M=M_range, ratio_hierarchical=ratiohier_range, sigmax=sigmax_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

