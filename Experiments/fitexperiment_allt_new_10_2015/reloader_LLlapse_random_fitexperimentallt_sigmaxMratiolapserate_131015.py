"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os
import numpy as np

import matplotlib as mpl
if 'DISPLAY' in os.environ and mpl.get_backend() == 'pdf':
    # Most likely Gatsby machine interactively, change backend
    mpl.use('TkAgg')

import matplotlib.pyplot as plt

from experimentlauncher import ExperimentLauncher
from dataio import DataIO
import launchers_memorycurves_marginal_fi
import plots_experimental_data

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data


this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='New Likelihood, Lapse rate now! Loads an experiment, and uses the data for the model, for all T in the dataset. Computes Loglikelihood, BIC and fits the mixture model, and compare the mixture model fits to the experimental data. Meant to run random sampling for a long while! Uses the new output noise process, varying sigma_output accordingly. Also use the new normalised sigma_x, which should behave better.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     construct_multidimension_npyarr=False,
                     # limit_max_files=1,
                     parameters=['M', 'ratio_conj', 'sigmax', 'lapse_rate'],
                     variables_to_load=['result_ll_sum', 'result_ll_n', 'result_bic', 'result_ll90_sum', 'result_precision', 'result_em_fits', 'result_emfit_mse', 'result_emfit_mixt_kl', 'result_fi_theo', 'result_fi_theocov' ],
                     post_processing=None,
                     save_output_filename='plots_fitexperiment_%s_Mratiosigmaxlapserate' % generator_module.experiment_id
                     )


if __name__ == '__main__':

    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
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
