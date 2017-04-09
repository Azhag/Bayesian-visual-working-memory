import os
import numpy as np
import experimentlauncher
import inspect
import utils

# Commit @d8c9acb

# Read from other scripts
parameters_entryscript = dict(action_to_do='launcher_do_generate_submit_pbs_from_param_files', output_directory='.')
submit_jobs = True

parameter_generation = 'random'
num_random_samples = 10000
limit_max_queued_jobs = 60

resource = ''

# partition = 'wrkstn'
# partition = 'test'
partition = 'intel-ivy'

# submit_cmd = 'qsub'
submit_cmd = 'sbatch'

num_repetitions = 3

run_label = 'fisher2016_random_large_3try_repetitions{num_repetitions}_010916'
sleeping_period = dict(min=10, max=30)

pbs_submission_infos = dict(description='Small sweep to get Receptive width effect.',
                            command='python $WORKDIR/experimentlauncher.py',
                            other_options=dict(action_to_do='launcher_check_fisher_fit_1obj_2016',
                                               code_type='conj',
                                               output_directory='.',
                                               ratio_conj=1,
                                               M=14 * 14,
                                               sigmax=0.25,
                                               renormalize_sigma=None,
                                               N=200,
                                               R=2,
                                               T=1,
                                               sigmay=0.000001,
                                               sigma_output=0.0,
                                               sigma_baseline=0.0,
                                               inference_method='sample',
                                               num_samples=50,
                                               selection_num_samples=1,
                                               selection_method='last',
                                               slice_width=0.07,
                                               burn_samples=50,
                                               num_repetitions=num_repetitions,
                                               enforce_min_distance=0.17,
                                               specific_stimuli_random_centers=None,
                                               stimuli_generation='random',
                                               stimuli_generation_recall='random',
                                               rc_scale=1.0,
                                               label=run_label,
                                               experiment_data_dir=os.path.normpath(os.path.join(os.environ['WORKDIR_DROP'], '../../experimental_data')),
                                               ),
                            walltime='1:00:00',
                            memory='2gb',
                            simul_out_dir=os.path.join(os.getcwd(), run_label.format(**locals())),
                            pbs_submit_cmd=submit_cmd,
                            limit_max_queued_jobs=limit_max_queued_jobs,
                            source_dir=os.environ['WORKDIR_DROP'],
                            submit_label='fisher_3try_0109',
                            resource=resource,
                            partition=partition,
                            qos='auto')


# ## Define our filtering function
# def filtering_function(new_parameters,
#                        dict_parameters_range,
#                        function_parameters=None):
#     '''
#     Given M and ratio_conj, will adapt them so that M_conj is always correct and integer.

#     or if should_clamp is False, will not change them
#     '''

#     M_true, ratio_true = utils.fix_M_ratioconj(
#         new_parameters['M'], new_parameters['ratio_conj'])

#     if function_parameters['should_clamp']:
#         # Clamp them and return true
#         new_parameters['M'] = M_true
#         new_parameters['ratio_conj'] = ratio_true

#         return True
#     else:
#         return np.allclose(M_true, new_parameters['M'])

# filtering_function_parameters = {'should_clamp': True}

dict_parameters_range = dict(
    T=dict(sampling_type='randint',
           low=1,
           high=2,
           dtype=int
           ),
    rc_scale=dict(sampling_type='uniform',
                  low=0.01,
                  high=60.0,
                  dtype=float
                  ),
)

if __name__ == '__main__':

    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)

    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)
