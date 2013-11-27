"""
    ExperimentDescriptor for Misbinding effect for Mixed population code.
"""

import os
import numpy as np
from experimentlauncher import *
import inspect

# Commit @2042319 +

# Read from other scripts
parameters_entryscript=dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'

M = 200
sigmax = 0.2


run_label = 'misbinding_mixed_varyratio_avgposterior_sigmax{sigmax}_M{M}_ratioconj{ratio_conj}_210613'

pbs_submission_infos = dict(description='Study misbindings, by computing an average posterior for fixed stimuli. Check the distribution of errors as well. Uses a Mixed population, vary ratio_conj to see what happens. Limit to squared ratio_subpop_nb only',
                            command='python /nfs/home2/lmatthey/Documents/work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_average_posterior',
                                               code_type='mixed',
                                               output_directory='.',
                                               ratio_conj=0.5,
                                               M=M,
                                               sigmax=sigmax,
                                               N=1000,
                                               T=2,
                                               sigmay=0.0001,
                                               inference_method='none',
                                               num_repetitions=1,
                                               stimuli_generation='separated',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(sigmax=sigmax, M=M, ratio_conj='')),
                            pbs_submit_cmd='qsub',
                            submit_label='avgpost_mixed_MMl')


ratio_range           =   dict(range=np.arange(2, int(M**0.5)+1.)**2.0/M, dtype=float)

dict_parameters_range = dict(ratio_conj=ratio_range)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

