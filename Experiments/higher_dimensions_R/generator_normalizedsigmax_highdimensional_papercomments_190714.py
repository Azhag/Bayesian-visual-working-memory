"""
    ExperimentDescriptor that collects precision and model fits while varying
    M and ratio_conj for a mixed population code.

    See if for a given precision, ratio_conj varies linearly with N.

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
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

# limit_max_queued_jobs = 50

num_repetitions = 3
# num_repetitions = 10
# T = 1
# T = 2
T = 3
sigmax = 0.1
cued_feature_type = 'all'

run_label = 'highdimensional_normalizedsigmax_mixed_papercomments_cuedtype{cued_feature_type}T{T}repetitions{num_repetitions}_170714'

pbs_submission_infos = dict(description='Runs and collect precision, mixture model fits and fisher info for varying M, ratio_conj and R. Should then look at how R affects the precision and the ratio. New with max output Response implementation.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_check_scaling_ratio_with_M_single',
                                               code_type='mixed',
                                               output_directory='.',
                                               type_layer_one='feature',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               ratio_hierarchical=0.5,
                                               ratio_conj=1.0,
                                               M=100,
                                               sigmax=sigmax,
                                               renormalize_sigmax=None,
                                               N=200,
                                               T=T,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               cued_feature_type=cued_feature_type,
                                               num_samples=300,
                                               selection_num_samples=1,
                                               slice_width=0.07,
                                               burn_samples=200,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               enforce_distance_cued_feature_only=None
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            submit_label='highdim_respmax%d' % T)

nb_M = 11.

M_range      =   dict(range=np.arange(100, 800, round((800-100)/nb_M)), dtype=int)
ratio_range  =   dict(range=np.linspace(0.0001, 1.0, 15.), dtype=float)
R_range      =   dict(range=np.arange(2, 6), dtype=int)

dict_parameters_range =   dict(M=M_range, ratio_conj=ratio_range, R=R_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

