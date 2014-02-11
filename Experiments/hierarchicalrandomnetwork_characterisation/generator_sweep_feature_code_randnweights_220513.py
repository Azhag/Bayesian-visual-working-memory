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

pbs_submission_infos = dict(description='Hierarchical network, feature layer one. Checking effect of M, sparsity and sigma_weights. Done with randn weights here',
                            command='python $WORKDIR_DROP/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature_pbs',
                                               code_type='hierarchical',
                                               output_directory='.',
                                               M=100,
                                               M_layer_one=100,
                                               type_layer_one='feature',
                                               distribution_weights='randn',
                                               normalise_weights=2,
                                               sigmax=0.1,
                                               N=500,
                                               T=6,
                                               sigmay=0.0001,
                                               inference_method='max_lik',
                                               num_repetitions=3,
                                               label='hierarchical_M_sparsity_sigmaweight_volume_featurelayer_randnweights'),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), 'hierarchical_M_sparsity_sigmaweight_volume_featurelayer_randnweights_220513'),
                            wait_submitting=True,
                            submit_label='hier_feat_randn')


M_range             =   dict(range=np.linspace(5, 505, 11), dtype=int)
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

