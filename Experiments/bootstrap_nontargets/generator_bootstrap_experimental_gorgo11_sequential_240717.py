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
num_workers = 1000

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

run_label = 'bootstrap_experimental_gorgo11_sequential_bootstrapsamples{num_repetitions}_240717'

pbs_submission_infos = dict(description='Collect bootstrap samples, using experimental data responses. Uses mixture model with single kappa. Gorgo11 Sequential dataset here.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_bootstrap_experimental_sequential',
                                               output_directory='.',
                                               num_repetitions=1,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='1:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            source_dir=os.environ['WORKDIR_DROP'],
                            submit_label='bootstrap_gorgo11_seq',
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

