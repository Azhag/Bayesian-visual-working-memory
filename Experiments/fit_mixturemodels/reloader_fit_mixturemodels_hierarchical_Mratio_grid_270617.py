"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os

from experimentlauncher import ExperimentLauncher
import re
import inspect
import imp


this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript = dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Sweep for Hierarchical network characterisation',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     construct_multidimension_npyarr=False,
                     # limit_max_files=1,
                     parameters=['M',
                                 'ratio_hierarchical',
                                 ],
                     variables_to_load=['result_all_precisions',
                                        'result_em_fits',
                                        'result_dist_bays09',
                                        'result_dist_bays09_emmixt_KL',
                                        'result_dist_gorgo11',
                                        'result_dist_gorgo11_emmixt_KL'
                                        ],
                     post_processing=None,
                     save_output_filename='fitmixt_hierarchical_grid'
                     )


if __name__ == '__main__':

    print "Running ", this_file

    arguments_dict = dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]
