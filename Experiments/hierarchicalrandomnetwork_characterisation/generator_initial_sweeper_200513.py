"""
    ExperimentDescriptor for hierarchicalrandomnetwork parameter sweep.
"""

import os
import numpy as np
from experimentlauncher import *
import inspect

# Commit @06693ee +

parameters_entryscript=dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'

pbs_submission_infos = dict(description='Hierarchical network, testing effect of M and M_layer_one. No filtering.', command='python $WORKDIR_DROP/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_hierarchical_precision_M_Mlower_pbs', code_type='hierarchical', output_directory='.', M=100, M_layer_one=100, type_layer_one='conjunctive', sigmax=0.1, N=500, T=6, sigmay=0.0001, inference_method='max_lik', num_repetitions=3, label='hierarchical_M_Mlower_volume_conjunctivelayer'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'hierarchical_M_Mlower_volume_conjunctivelayer_210513'))

M_range           =   dict(range=np.linspace(5, 505, 26), dtype=int)
M_lower_range     =   dict(range=np.arange(5, 31, 2)**2., dtype=int)
# M_range           =   dict(range=np.linspace(5, 505, 1), dtype=int)
# M_lower_range     =   dict(range=np.arange(5, 6, 2)**2., dtype=int)

dict_parameters_range = dict(M=M_range, M_layer_one=M_lower_range)


## BE CAREFUL ABOUT ALL_PARAMETERS FOR FILTERING FUNCTION...
# # Create the filtering all parameters dict, make sure parameters are appropriately shared...
# filtering_all_parameters = all_parameters.copy()
# for key, val in pbs_submission_infos['other_options'].items():
#     filtering_all_parameters[key] = val

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

