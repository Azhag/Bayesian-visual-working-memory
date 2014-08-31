"""
    ExperimentDescriptor to fit the experimental data, using the Mixture models only.

    Spawns random samples for now, let's see.
"""

import os
# import numpy as np
import experimentlauncher
import inspect
import getpass

# Commit @d8c9acb

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True

parameter_generation = 'random'  ## !!!!!! RANDOM HERE   !!!!!
num_random_samples = 1000
limit_max_queued_jobs = 30

resource = ''

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

# FOR DIRAC
if getpass.getuser() == 'dc-matt1':
  resource = 'DIRAC-DX001'
  submit_cmd = 'sbatch'
  pbs_unfilled_script = open(os.path.join(os.environ['WORKDIR_DROP'], 'dirac_submission_slurm_unfilled.sh'), 'r').read()

num_repetitions = 3
M = 144

run_label = 'rcscaleeffect_random_fitmixturemodels_rcscale_M{M}_repetitions{num_repetitions}_290814'

pbs_submission_infos = dict(description='Runs the model for 1..T items. Computes precision, Fisher information, fits the mixture model, and compare the mixture model fits to the experimental data (Bays09 and Gorgo11 here). Also stores all responses. Vary Rcscale, goal is to see where the optimal one lies.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fit_mixturemodels',
                                               code_type='mixed',
                                               output_directory='.',
                                               ratio_conj=1.0,
                                               M=M,
                                               sigmax=0.1,
                                               renormalize_sigmax=None,
                                               N=300,
                                               T=6,
                                               sigmay=0.0001,
                                               sigma_output=0.0,
                                               inference_method='sample',
                                               num_samples=300,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               slice_width=0.07,
                                               burn_samples=300,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               collect_responses=None,
                                               rc_scale=0.5,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='10:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            submit_label='rcscale_effect',
                            resource=resource,
                            qos='auto')

if getpass.getuser() == 'dc-matt1':
  pbs_submission_infos['pbs_unfilled_script'] = pbs_unfilled_script
  pbs_submission_infos['walltime'] = '12:00:00'


rc_scale_range     =   dict(sampling_type='uniform', low=0.01, high=30.0, dtype=float)

dict_parameters_range =   dict(rc_scale = rc_scale_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

