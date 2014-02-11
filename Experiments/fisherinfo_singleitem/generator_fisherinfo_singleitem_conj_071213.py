"""
    ExperimentDescriptor to get different metric for the Fisher information / Precision of samples
"""

import os
import numpy as np
import experimentlauncher
import inspect

# Commit @ee93064

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True
parameter_generation = 'grid'

num_repetitions = 1
num_workers = 500

submit_cmd = 'qsub'
# submit_cmd = 'sbatch'

run_label = 'fisherinfo_singleitem_conj_correctlargen_repetitions{num_repetitions}mult_121213'

pbs_submission_infos = dict(description='Runs multiple metric to get estimate of fisher information for a single item, using a conjunctive code here. Should then put everything back nicely. Corrected larg N FI now.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_do_fisher_information_param_search_pbs',
                                               code_type='conj',
                                               output_directory='.',
                                               M=100,
                                               sigmax=0.15,
                                               N=500,
                                               T=1,
                                               sigmay=0.0001,
                                               inference_method='sample',
                                               num_samples=200,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               slice_width=0.07,
                                               burn_samples=300,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
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
                            submit_label='fi_singleitem')

numrepet_range =    dict(range=np.array(num_workers*[num_repetitions]), dtype=int)

dict_parameters_range =   dict(num_repetitions=numrepet_range)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

