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

# import matplotlib.animation as plt_anim
from mpl_toolkits.mplot3d import Axes3D

import re
import inspect
import imp

import utils
# import submitpbs
import load_experimental_data


def plots_fit_mixturemodels_random(data_pbs, generator_module=None):
    '''
        Reload runs from PBS
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", data_pbs.dataset_infos['parameters']
    # parameters: M, ratio_conj, sigmax

    # Extract data
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']


    result_responses_flat = np.array(data_pbs.dict_arrays['result_responses']['results_flat'])
    result_targets_flat = np.array(data_pbs.dict_arrays['result_target']['results_flat'])
    result_nontargets_flat = np.array(data_pbs.dict_arrays['result_nontargets']['results_flat'])

    result_parameters_flat = np.array(data_pbs.dict_arrays['result_responses']['parameters_flat'])
    all_repeats_completed = data_pbs.dict_arrays['result_responses']['repeats_completed']

    all_args_arr = np.array(data_pbs.loaded_data['args_list'])

    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    ratio_conj_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    alpha_space = data_pbs.loaded_data['parameters_uniques']['alpha']
    trecall_space = data_pbs.loaded_data['parameters_uniques']['fixed_cued_feature_time']

    num_repetitions = generator_module.num_repetitions
    parameter_names_sorted = data_pbs.dataset_infos['parameters']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    ##### Because of lazyness, the responses are weird.
    # Each run is for a given trecall. But we run N items= 1 .. Nmax anyway
    # so if trecall > N, you have np.nan
    # => Need to reconstruct the thing properly, to have lower triangle of Nitem x Trecall filled
    # Also, trecall is the actual Time. Hence we need to change its meaning to be Tmax- (trecall + 1) or whatever.

    # Load ground truth
    data_gorgo11_sequ = load_experimental_data.load_data_gorgo11_sequential(fit_mixture_model=True)

    ## Filter everything with repeats_completed == num_repet and trecall=last
    filter_data = (result_parameters_flat[:, 0] == (T_space.max() - 1) ) & (all_repeats_completed == num_repetitions - 1)
    result_parameters_flat = result_parameters_flat[filter_data]
    result_responses_flat = result_responses_flat[filter_data]
    result_targets_flat = result_targets_flat[filter_data]
    result_nontargets_flat = result_nontargets_flat[filter_data]
    all_args_arr = all_args_arr[filter_data]
    all_repeats_completed = all_repeats_completed[filter_data]

    print "Size post-filter: ", result_parameters_flat.shape[0]

    def str_best_params(best_i, result_dist_to_use):
        return ' '.join(["%s %.4f" % (parameter_names_sorted[param_i], result_parameters_flat[best_i, param_i]) for param_i in xrange(len(parameter_names_sorted))]) + ' >> %f' % result_dist_to_use[best_i]


    # all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['parameter_names_sorted', 'all_args_arr', 'all_repeats_completed', 'filter_data']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='gorgo11_sequential_fitmixturemodel')


    plt.show()

    return locals()


def mask_outliers(result_dist_to_use, sigma_outlier=3):
    '''
        Mask outlier datapoints.
        Compute the mean of the results and assume that points with:
          result > mean + sigma_outlier*std
        are outliers.

        As we want the minimum values, do not mask small values
    '''
    return np.ma.masked_greater(result_dist_to_use, np.mean(result_dist_to_use) + sigma_outlier*np.std(result_dist_to_use))



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. All experiments. Random sampling of parameter space. Perhaps too big, be careful...',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     construct_multidimension_npyarr=False,
                     # limit_max_files=1,
                     parameters=['fixed_cued_feature_time', 'M', 'ratio_conj', 'sigmax', 'alpha'],
                     variables_to_load=['result_responses', 'result_target', 'result_nontargets'],
                     variables_description=['responses' 'target', 'nontargets'],
                     post_processing=plots_fit_mixturemodels_random,
                     save_output_filename='plots_sequentialgorgo11_fitmixturemodel_Mratiosigmaxalphatrecall'
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
