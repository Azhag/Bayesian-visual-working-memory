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


def plots_logposterior_mixed_autoset(data_pbs, generator_module=None):
    '''
        Reload 2D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    plot_per_ratio = False
    plot_2d_pcolor = False

    do_relaunch_bestparams_pbs = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_log_posterior_mean = np.squeeze(data_pbs.dict_arrays['result_log_posterior_mean']['results'])
    result_log_posterior_std = np.squeeze(data_pbs.dict_arrays['result_log_posterior_std']['results'])

    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']

    exp_dataset = data_pbs.loaded_data['args_list'][0]['experiment_id']

    print ratio_space
    print sigmax_space
    print result_log_posterior_mean.shape, result_log_posterior_std.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])


    if plot_per_ratio:
        # Plot the evolution of loglike as a function of sigmax, with std shown
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            ax = utils.plot_mean_std_area(sigmax_space, result_log_posterior_mean[ratio_conj_i], result_log_posterior_std[ratio_conj_i])

            ax.get_figure().canvas.draw()

            if savefigs:
                dataio.save_current_figure('results_fitexp_%s_loglike_ratioconj%.2f_{label}_global_{unique_id}.pdf' % (exp_dataset, ratio_conj))

    if plot_2d_pcolor:
        # Plot the mean loglikelihood as a 2d surface
        utils.pcolor_2d_data(result_log_posterior_mean, x=ratio_space, y=sigmax_space, xlabel="Ratio conj", ylabel="Sigma x", title="Loglikelihood of experimental data, \n3 items dualrecall, rcscale automatically set", ticks_interpolate=5, cmap=colormap)
        # plt.tight_layout()

        if savefigs:
            dataio.save_current_figure('results_fitexp_%s_loglike_2d_ratiosigmax_{label}_global_{unique_id}.pdf' % exp_dataset)

    if do_relaunch_bestparams_pbs:
        # Will check the best fitting parameters, and relaunch simulations for them, in order to get new cool plots.

        # We can actually use the original dictionary for submission informations, and just change the launcher to one produces the appropriate results
        generator_parameters_dict = generator_module.__dict__
        generator_parameters_dict['pbs_submission_infos']['other_options'].update(dict(
            action_to_do='launcher_do_memory_curve_marginal_fi',
            subaction='collect_responses',
            inference_method='sample',
            N=300,
            T=6,
            num_samples=500,
            selection_method='last',
            num_repetitions=3,
            burn_samples=500,
            stimuli_generation='random',
            stimuli_generation_recall='random',
            session_id='fitting_experiments_relaunchs',
            result_computation='filenameoutput'))
        generator_parameters_dict['pbs_submission_infos']['other_options']['label'] = 'fitting_experiment_rerun_nitems{T}M{M}_090414'.format(T = generator_parameters_dict['n_items_to_fit'], M = generator_parameters_dict['M'])
        generator_parameters_dict['sleeping_period'] = dict(min=5, max=10)

        submit_pbs = submitpbs.SubmitPBS(pbs_submission_infos=generator_parameters_dict['pbs_submission_infos'], debug=True)

        # Now run a series of Jobs to obtain the required data
        # the "Result" of them are the filename of their output. Should then save those in a dictionary.
        # For later processing, if the exact same parameters are used, then everything will be reloaded automatically, MAAAGIC.

        # Extract the parameters to try
        best_params_to_try = []
        best_axis2_i_all = np.argmax(result_log_posterior_mean, axis=1)
        for axis1_i, best_axis2_i in enumerate(best_axis2_i_all):
            parameter_dict = dict(ratio_conj=ratio_space[axis1_i], sigmax=sigmax_space[best_axis2_i])

            best_params_to_try.append(parameter_dict)

        # Submit them, waiting on them in the process. Should obtain a list of filenames back
        utils.chdir_safe(generator_parameters_dict['pbs_submission_infos']['simul_out_dir'])
        result_minibatch_filenames = submit_pbs.submit_minibatch_jobswrapper(best_params_to_try, generator_parameters_dict)

        result_reloaded_datasets = []
        for filename_i, result_filename in enumerate(result_minibatch_filenames):
            curr_reloaded_dataset = utils.load_npy(result_filename + '.npy')
            result_reloaded_datasets.append(curr_reloaded_dataset)

        utils.chdir_safe('../')






    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['exp_dataset']

    if savefigs:
        dataio.save_variables_default(locals(), variables_to_save)


    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. Dualrecall dataset, fitting for 3 items. We submit varying ratio_conj and sigmax, and vary rcscale and rcscale2 on each node. Should plot as a 2D surface',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj', 'sigmax'],
                     variables_to_load=['result_log_posterior_mean', 'result_log_posterior_std'],
                     variables_description=['Log posterior mean of experimental data', 'Log posterior std dev'],
                     post_processing=plots_logposterior_mixed_autoset,
                     save_output_filename='plots_fitexp_mixed_autoset'
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

