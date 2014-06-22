"""
    ExperimentDescriptor to fit Memory curves using a Conjunctive population code

    Uses the new Marginal Inverse Fisher Information, and some new code altogether.
"""

import os
import numpy as np
import experimentlauncher
import inspect

# Commit @4ffae5c

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

# sigmax = 0.333
# M = 225
# ratio_hierarchical = 1.0  # Cheat and use this to separate runs, hackkkyyyy

sigmax = 0.26136364
M  = 196
ratio_hierarchical = 0.9  # Cheat and use this to separate runs, hackkkyyyy


num_repetitions = 1
# num_repetitions = 10
num_workers = 30


run_label = 'memory_curve_conj_ratiosigmax_separated_repetitions{num_repetitions}multworkers_100314'

pbs_submission_infos = dict(description='Fit Memory curves using the new code (october 2013). Compute marginal inverse fisher information, which is slightly better at capturing items interactions effects. Also fit Mixture models directly. Uses conjunctive population code, automatically set. New space of search. Uses 6 items.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_memory_curve_marginal_fi',
                                               code_type='conj',
                                               output_directory='.',
                                               type_layer_one='feature',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               ratio_hierarchical=1.0,
                                               M=M,
                                               sigmax=sigmax,
                                               N=300,
                                               T=6,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               num_samples=500,
                                               selection_method='last',
                                               selection_num_samples=1,
                                               slice_width=0.07,
                                               burn_samples=300,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='separated',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            submit_label='memorycurves_conj')


sigmax_range      =   dict(range=np.array(num_workers*[sigmax]), dtype=float)
ratiohier_range   =   dict(range=np.array([ratio_hierarchical]), dtype=float)

dict_parameters_range =   dict(ratio_hierarchical=ratiohier_range, sigmax=sigmax_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

