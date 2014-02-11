"""
    ExperimentDescriptor for Mixed parameter sweep.
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

## Define our filtering function
def filtering_function(new_parameters, dict_parameters_range, function_parameters=None):
    '''
        Receive M and ratio_conj, make sure that M_conj = M*ratio_conj is a squared number (or close to it)
    '''

    return ((function_parameters['M']*new_parameters['ratio_conj'])**0.5 % 1.0) < 1e-3

filtering_function_parameters = {'M': M}

run_label = 'mixed_varyratio_sigmax{sigmax}_200613'

pbs_submission_infos = dict(description='Mixed population. Check how the precision changes for varying ratio_conj and T.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_mixed_varyratio_precision_pbs',
                                               code_type='mixed',
                                               output_directory='.',
                                               M=M,
                                               sigmax=0.2,
                                               N=500,
                                               T=6,
                                               ratio_conj=0.5,
                                               sigmay=0.0001,
                                               inference_method='max_lik',
                                               num_repetitions=5,
                                               num_samples=500,
                                               selection_method='last',
                                               autoset_parameters=1,
                                               label=run_label),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label),
                            pbs_submit_cmd='qsub',
                            submit_label='mixed_ratio')

ratio_range           =   dict(range=np.linspace(0.01, 1.0, 2000.), dtype=float)

dict_parameters_range = dict(ratio_conj=ratio_range)


if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

