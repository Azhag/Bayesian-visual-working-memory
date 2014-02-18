"""
    ExperimentDescriptor to fit Hierarchical population code on Specific stimuli
"""

import os
import numpy as np
from experimentlauncher import *
import inspect

from utils import cross

# Commit @8255dc8

# Read from other scripts
parameters_entryscript=dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'
submit_cmd = 'qsub'
# submit_cmd = 'sbatch'

M = 200
num_repetitions = 5

run_label = 'specific_stimuli_corrected_hierarchical_sigmaxmindistance_autoset_repetitions{num_repetitions}mult_collectall_041213'

pbs_submission_infos = dict(description='See patterns of errors on Specific Stimuli, with Mixed population code. Internally vary ratio_conj. Vary sigmax and enforce_min_distance here. Collect everything, with corrected varying stimuli',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_hierarchical_special_stimuli_varyMMlower',
                                               output_directory='.',
                                               subaction='collect_responses',
                                               M=M,
                                               sigmax=0.1,
                                               code_type='hierarchical',
                                               type_layer_one='feature',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               N=100,
                                               T=3,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               slice_width=0.07,
                                               num_samples=100,
                                               burn_samples=200,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='specific_stimuli',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            submit_label='spec_stim_hier')

sigmax_range          =   dict(range=np.array([0.10, 0.153, 0.20, 0.255, 0.3, 0.4]), dtype=float)
min_distance_range    =   dict(range=np.linspace(0.01, np.pi/2., 23.), dtype=float)

dict_parameters_range =   dict(enforce_min_distance=min_distance_range, sigmax=sigmax_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

