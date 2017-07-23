"""
    ExperimentDescriptor to get bootstrap samples of the full item mixture model.

    Mixed population code.

    Based on Bays 2009.
"""

import os
import numpy as np
import experimentlauncher
import inspect

# Commit @bb60b17+

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'
limit_max_queued_jobs = 90
resource = ''
partition = 'wrkstn'
# partition = 'test'
# partition = 'intel-ivy'
# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 1
num_workers = 500

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

run_label = 'bootstrap_experimental_gorgo11_bootstrapsamples{num_repetitions}mult_200717'

pbs_submission_infos = dict(description='Collect bootstrap samples, using experimental data responses. Uses mixture model with single kappa. Gorgo11 dataset here.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_bootstrap_experimental',
                                               experiment_id='gorgo11',
                                               output_directory='.',
                                               output_both_layers=None,
                                               normalise_weights=1,
                                               threshold=1.0,
                                               ratio_hierarchical=0.5,
                                               ratio_conj=0.845,
                                               M=200,
                                               sigmax=0.4,
                                               N=1000,
                                               T=1,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               num_samples=500,
                                               selection_num_samples=1,
                                               slice_width=0.07,
                                               burn_samples=500,
                                               num_repetitions=1,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               autoset_parameters=None,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='1:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            source_dir=os.environ['WORKDIR_DROP'],
                            submit_label='bootstrap_gorgo11',
                            resource=resource,
                            partition=partition,
                            qos='auto')

numrepet_range = dict(range=np.array(num_workers*[num_repetitions]), dtype=int)

dict_parameters_range = dict(num_repetitions=numrepet_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

