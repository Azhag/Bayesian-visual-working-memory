"""
    ExperimentDescriptor to collect responses, used then to create histograms of errors and bias to nontarget responses

    Mixed population code.

    Based on Bays 2009.
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
submit_cmd = 'qsub'
# submit_cmd = 'sbatch'

num_repetitions = 1
# num_repetitions = 10
M  = 200
T  = 6
ratio_hierarchical = 0.84
sigmax = 0.68

run_label = 'error_distribution_hier_paperplots_ratiohier{ratio_hierarchical}sigmax{sigmax}T{T}repetitions{num_repetitions}_220114'

pbs_submission_infos = dict(description='Runs and collect responses to generate histograms. Should do it for multiple T, possibly with the parameters optimal for memory curve fits (but not that needed...)',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_error_distributions',
                                               code_type='hierarchical',
                                               output_directory='.',
                                               type_layer_one='feature',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               ratio_hierarchical=ratio_hierarchical,
                                               ratio_conj=0.845,
                                               M=M,
                                               sigmax=sigmax,
                                               N=1000,
                                               T=T,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               num_samples=500,
                                               selection_num_samples=1,
                                               slice_width=0.07,
                                               burn_samples=500,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='40:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            submit_label='errordist_hier')

# sigmax_range =    dict(range=np.linspace(0.05, 0.65, 13.), dtype=float)
T_range      =    dict(range=np.arange(1, T+1), dtype=int)

dict_parameters_range =   dict(T=T_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

