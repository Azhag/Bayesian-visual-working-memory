"""
    ExperimentDescriptor for hierarchicalrandomnetwork parameter sweep.
"""

import os
import numpy as np
from experimentlauncher import *
import inspect

# Commit @53dce72 +
parameters_entryscript=dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'

normalise_weights = 2

if normalise_weights == 1:
    run_label = 'hierarchical_M_sparsity_sigmaweight_volume_featurelayer_varythreshold_230513'
elif normalise_weights == 2:
    run_label = 'hierarchical_M_sparsity_sigmaweight_volume_featurelayer_varythreshold_othernorm_300513'

pbs_submission_infos = dict(description='Hierarchical network, feature layer one, testing effect of M, sparsity, sigma weights and now threshold as well. No filtering.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_hierarchical_precision_M_sparsity_sigmaweight_feature_pbs',
                                               code_type='hierarchical',
                                               output_directory='.',
                                               M=100,
                                               M_layer_one=100,
                                               type_layer_one='feature',
                                               sigmax=0.1,
                                               N=500,
                                               T=6,
                                               sigmay=0.0001,
                                               inference_method='max_lik',
                                               num_repetitions=3,
                                               normalise_weights=normalise_weights,
                                               label=run_label),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label),
                            submit_label='hier_feat_thr')


# M_range             =   dict(range=np.linspace(5, 505, 26), dtype=int)
M_range             =   dict(range=np.array([25, 50, 100, 200]), dtype=int)
sparsity_range      =   dict(range=np.linspace(0.01, 1.0, 10.), dtype=float)
sigma_weights_range =   dict(range=np.linspace(0.1, 5.0, 10), dtype=float)
threshold_range     =   dict(range=np.linspace(0.0, 2., 10.), dtype=float)


dict_parameters_range = dict(M=M_range, sparsity=sparsity_range, sigma_weights=sigma_weights_range, threshold=threshold_range)


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

