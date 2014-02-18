"""
    ExperimentDescriptor to get different metric for the Fisher information / Precision of samples
"""

import os
import numpy as np
from experimentlauncher import *
from dataio import *
import re

import matplotlib.pyplot as plt

import inspect

import utils
import cPickle as pickle

import statsmodels.distributions as stmodsdist

import em_circularmixture_allitems_uniquekappa

# Commit @7f4e2c6


def plots_fisherinfo_singleitem(data_pbs, generator_module=None):
    '''
        Reload bootstrap samples, plot its histogram, fit empirical CDF and save it for quicker later use.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_metric_comparison = True

    # caching_bootstrap_filename = None
    # caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_misbinding_mixed.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_FI_rc_curv_mult = np.squeeze(data_pbs.dict_arrays['result_FI_rc_curv_mult']['results'])
    result_FI_rc_curv_all = np.squeeze(data_pbs.dict_arrays['result_FI_rc_curv_all']['results'])
    result_FI_rc_precision_mult = np.squeeze(data_pbs.dict_arrays['result_FI_rc_precision_mult']['results'])
    result_FI_rc_theo_mult = np.squeeze(data_pbs.dict_arrays['result_FI_rc_theo_mult']['results'])
    result_FI_rc_truevar_mult = np.squeeze(data_pbs.dict_arrays['result_FI_rc_truevar_mult']['results'])
    result_FI_rc_samples_mult = np.squeeze(data_pbs.dict_arrays['result_FI_rc_samples_mult']['results'])
    result_FI_rc_samples_all = np.squeeze(data_pbs.dict_arrays['result_FI_rc_samples_all']['results'])
    result_em_fits_all = np.squeeze(data_pbs.dict_arrays['result_em_fits']['results'])

    print result_FI_rc_curv_mult.shape
    print result_FI_rc_curv_all.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    if plot_metric_comparison:
        values_bars = np.array([np.mean(result_em_fits_all[0]), np.mean(result_FI_rc_theo_mult[0]), np.mean(result_FI_rc_theo_mult[1]), np.mean(result_FI_rc_curv_all)])
        values_bars_std = np.array([np.std(result_em_fits_all[0]), np.std(result_FI_rc_theo_mult[0]), np.std(result_FI_rc_theo_mult[1]), np.std(result_FI_rc_curv_all)])

        set_colormap = plt.cm.cubehelix
        color_gen = [set_colormap((i+0.1)/(float(np.max((5, len(values_bars)))+0.1))) for i in xrange(np.max((5, len(values_bars))))][::-1]

        bars_indices = np.arange(values_bars.size)
        width = 0.7

        ## Plot all as bars
        f, ax = plt.subplots(figsize=(10,6))

        for bar_i in xrange(values_bars.size):
            plt.bar(bars_indices[bar_i], values_bars[bar_i], width=width, color=color_gen[bar_i], zorder=2)
            plt.errorbar(bars_indices[bar_i] + width/2., values_bars[bar_i], yerr=values_bars_std[bar_i], ecolor='k', capsize=20, capthick=2, linewidth=2, zorder=3)

        # Add the precision bar times 2
        plt.bar(bars_indices[0], 2*values_bars[0], width=width, color=color_gen[0], alpha=0.5, hatch='/', linestyle='dashed', zorder=1)

        plt.xticks(bars_indices + width/2., ['Precision', 'Fisher Information', 'Fisher Information\n Large N', 'Curvature'], rotation=0)
        plt.xlim((-0.2, bars_indices.size))
        f.canvas.draw()
        plt.tight_layout()


        if savefigs:
            dataio.save_current_figure('bar_fisherinfo_singleitem_metric_comparison_{label}_{unique_id}.pdf')


    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['nb_repetitions']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='fisherinfo_singleitem')


    plt.show()


    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]
# generator_script = 'generator_specific_stimuli_mixed_fixedemfit_otherrange_201113.py'

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs multiple metric to get estimate of fisher information for a single item, using a conjunctive code here. Should then put everything back nicely.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['num_repetitions'],
                     variables_to_load=['result_FI_rc_curv_mult', 'result_FI_rc_curv_all', 'result_FI_rc_precision_mult', 'result_FI_rc_theo_mult', 'result_FI_rc_truevar_mult', 'result_FI_rc_samples_mult', 'result_FI_rc_samples_all', 'result_em_fits'],
                     variables_description=['FI from curvature mean/std', 'FI from curvature all samples', 'FI from precision', 'FI from theory, both finite size and large N', 'FI from posterior samples mean/std', 'FI from posterior samples all samples'],
                     post_processing=plots_fisherinfo_singleitem,
                     save_output_filename='plots_fisherinfo_singleitem',
                     concatenate_multiple_datasets=True
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

