"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os
import numpy as np
from experimentlauncher import *
from dataio import *

import re
import inspect

import utils
import submitpbs

# Commit @2042319 +


def plots_fitting_experiments_random(data_pbs, generator_module=None):
    '''
        Reload 2D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_per_T = False
    plots_interpolate = False

    # do_relaunch_bestparams_pbs = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()
    # parameters: ratio_conj, sigmax, T

    # Extract data
    result_fitexperiments_flat = np.array(data_pbs.dict_arrays['result_fitexperiments']['results_flat'])
    result_fitexperiments_all_flat = np.array(data_pbs.dict_arrays['result_fitexperiments_all']['results_flat'])
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_fitexperiments']['parameters_flat'])

    # Extract order of datasets
    experiment_ids = data_pbs.loaded_data['datasets_list'][0]['fitexperiment_parameters']['experiment_ids']

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    # Compute some stuff
    result_fitexperiments_bic_avg = utils.nanmean(result_fitexperiments_flat[:, 0, 0], axis=-1)
    T_space = np.unique(result_parameters_flat[:, 2])


    if plots_per_T:
        for T in T_space:
            currT_indices = result_parameters_flat[:, 2] == T

            utils.contourf_interpolate_data_interactive_maxvalue(result_parameters_flat[currT_indices][..., :2], result_fitexperiments_bic_avg[currT_indices], xlabel='Ratio_conj', ylabel='sigma x', title='BIC, T %d' % T, interpolation_numpoints=200, interpolation_method='nearest', log_scale=False)

    # Interpolate
    if plots_interpolate:

        rcscale_target = 5.0

        rbf_interpolator = spint.Rbf(result_parameters_flat[:,0], result_parameters_flat[:,1], result_parameters_flat[:,2], result_parameters_flat[:,3], result_fitexperiments_bic_avg, smooth = 0.0)

        param1_space = np.linspace(0.01, 1.0, 50)
        param2_space = np.linspace(0.01, 1.0, 50)
        param3_space = np.array([rcscale_target])
        param4_space = np.array([rcscale_target])
        params_crossspace = np.array(utils.cross(param1_space, param2_space, param3_space, param4_space))

        interpolated_data = rbf_interpolator(params_crossspace[:, 0], params_crossspace[:, 1], params_crossspace[:, 2], params_crossspace[:, 3]).reshape((param1_space.size, param2_space.size))

        # utils.pcolor_2d_data(interpolated_data, param1_space, param2_space, 'ratio', 'sigmax', 'interpolated, fixing rcscales= %.2f' % rcscale_target)

        points_closeby = ((result_parameters_flat[:, 2] - rcscale_target)**2 + (result_parameters_flat[:, 3] - rcscale_target)**2) < 0.5
        plt.figure()
        plt.imshow(interpolated_data, extent=(param1_space.min(), param1_space.max(), param2_space.min(), param2_space.max()))
        plt.scatter(result_parameters_flat[points_closeby, 0], result_parameters_flat[points_closeby, 1], s=100, c=result_fitexperiments_bic_avg[points_closeby], marker='o')


    # if plot_per_ratio:
    #     # Plot the evolution of loglike as a function of sigmax, with std shown
    #     for ratio_conj_i, ratio_conj in enumerate(ratio_space):
    #         ax = utils.plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

    #         ax.get_figure().canvas.draw()

    #         if savefigs:
    #             dataio.save_current_figure('results_fitexp_%s_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % (exp_dataset, ratio_conj))



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['experiment_ids']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)
        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='fitting_experiments')


    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. All experiments. Random sampling of parameter space. Perhaps too big, be careful...',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj', 'sigmax', 'rc_scale', 'rc_scale2'],
                     variables_to_load=['result_fitexperiments', 'result_fitexperiments_all'],
                     variables_description=['Fit experiments sum', 'Fit experiments per experiment'],
                     post_processing=plots_fitting_experiments_random,
                     save_output_filename='plots_fitexp_random_mixed'
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

