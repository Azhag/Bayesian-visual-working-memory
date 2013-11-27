"""
    ExperimentDescriptor to fit Memory curves using a Conjunctive population code

    Uses the new Marginal Inverse Fisher Information, and some new code altogether.
    Precisions do fit nicely, given a factor of 2.
"""

import os
import numpy as np
import experimentlauncher
import inspect

# Commit @8191aa2 +

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'
submit_cmd = 'qsub'
# submit_cmd = 'sbatch'

num_repetitions = 10

run_label = 'memory_curve_conj_Msigmax_autoset_repetitions{num_repetitions}_pbs_201013'

pbs_submission_infos = dict(description='Fit Memory curves using the new code (october 2013). Compute marginal inverse fisher information, which is slightly better at capturing items interactions effects. Also fit Mixture models directly.',
                            command='python /nfs/home2/lmatthey/Documents/work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_memory_curve_marginal_fi',
                                               code_type='conj',
                                               output_directory='.',
                                               ratio_conj=0.5,
                                               M=100,
                                               sigmax=0.1,
                                               N=200,
                                               T=5,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               num_samples=100,
                                               selection_num_samples=1,
                                               burn_samples=100,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='separated',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir='/nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data',
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            submit_label='memorycurves_conj')


sigmax_range  =   dict(range=np.linspace(0.01, 0.5, 25.), dtype=float)
M_range       =   dict(range=(np.arange(1, 21)**2.).astype(int), dtype=int)

dict_parameters_range =   dict(M=M_range, sigmax=sigmax_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

