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

pbs_submission_infos = dict(description='Hierarchical network, testing effect of M and M_layer_one. No filtering.', command='python /nfs/home2/lmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', other_options=dict(action_to_do='launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature_pbs', code_type='hierarchical', output_directory='.', M=100, M_layer_one=100, type_layer_one='feature', sigmax=0.1, N=500, T=6, sigmay=0.0001, inference_method='max_lik', num_repetitions=3, label='hierarchical_M_sparsity_sigmaweight_volume_featurelayer'), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(os.getcwd(), 'hierarchical_M_sparsity_sigmaweight_volume_featurelayer_210513'))

M_range             =   dict(range=np.linspace(5, 505, 26), dtype=int)
sparsity_range      =   dict(range=np.linspace(0.01, 1.0, 10.), dtype=float)
sigma_weights_range =   dict(range=np.linspace(0.1, 2.0, 10), dtype=float)

dict_parameters_range = dict(M=M_range, sparsity=sparsity_range, sigma_weights=sigma_weights_range)


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

