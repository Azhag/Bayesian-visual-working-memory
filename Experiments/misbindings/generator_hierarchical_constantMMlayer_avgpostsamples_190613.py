"""
    ExperimentDescriptor for hierarchicalrandomnetwork parameter sweep.
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

## Define our filtering function 
def filtering_function(new_parameters, dict_parameters_range, function_parameters=None):
    '''
        Receive M and M_layer_one, should make sure their sum is equal to some constant value (set in function_parameters)
    '''
    if function_parameters['comparison_type'] == 'equal':
        return np.allclose((new_parameters['M'] + new_parameters['M_layer_one']), function_parameters['target_M_total'])
    elif function_parameters['comparison_type'] == 'smaller_equal':
        return (float(new_parameters['M'] + new_parameters['M_layer_one']) <= function_parameters['target_M_total']) and (new_parameters['M'] < new_parameters['M_layer_one'])

filtering_function_parameters = {'target_M_total': 200., 'comparison_type': 'equal'}

run_label = 'misbinding_hierarchical_constantMMlower_avgposterior_sampling_190613'

pbs_submission_infos = dict(description='Study misbindings, by computing an average posterior for fixed stimuli. Check the distribution of errors as well. Uses a Hierarchical Network, and vary the ratio between M higher / M lower to see if a reversal occurs.', 
                            command='python /nfs/home2/lmatthey/Documents/work/Visual_working_memory/code/git-bayesian-visual-working-memory/experimentlauncher.py', 
                            other_options=dict(action_to_do='launcher_do_average_posterior', 
                                               code_type='hierarchical', 
                                               output_directory='.', 
                                               M=100, 
                                               M_layer_one=100, 
                                               type_layer_one='feature', 
                                               sigmax=0.2, 
                                               N=1000, 
                                               T=2, 
                                               sigmay=0.0001, 
                                               output_both_layers=1,
                                               inference_method='none', 
                                               num_repetitions=1,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               stimuli_generation='separated',
                                               stimuli_generation_recall='random',
                                               label=run_label), 
                            walltime='10:00:00', 
                            memory='2gb', 
                            simul_out_dir=os.path.join(os.getcwd(), run_label),
                            pbs_submit_cmd='qsub',
                            submit_label='avgpost_hier_MMl')

M_range           =   dict(range=np.arange(1, 201), dtype=int)
M_lower_range     =   dict(range=np.arange(2, 200, 2), dtype=int)        
# M_range           =   dict(range=np.linspace(5, 505, 1), dtype=int)
# M_lower_range     =   dict(range=np.arange(5, 6, 2)**2., dtype=int)

dict_parameters_range = dict(M=M_range, M_layer_one=M_lower_range)


if __name__ == '__main__':
    
    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

