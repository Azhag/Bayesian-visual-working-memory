"""
    ExperimentDescriptor to collect responses, used then to create histograms of errors and bias to nontarget responses

    Based on Bays 2009.
"""

import os
import numpy as np
import experimentlauncher
import dataio as DataIO
import utils
import re
import imp
import matplotlib.pyplot as plt

import inspect


# Commit @4ffae5c +


def plots_errors_distribution(data_pbs, generator_module=None):
    '''
        Reload responses

        Plot errors distributions.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plot_persigmax = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16

    angle_space = np.linspace(-np.pi, np.pi, 51)
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_responses_all = data_pbs.dict_arrays['result_responses']['results']
    result_target_all = data_pbs.dict_arrays['result_target']['results']
    result_nontargets_all =  data_pbs.dict_arrays['result_nontargets']['results']
    result_em_fits_all = data_pbs.dict_arrays['result_em_fits']['results']

    T_space = data_pbs.loaded_data['parameters_uniques']['T']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']
    nb_repetitions = result_responses_all.shape[-1]
    N = result_responses_all.shape[-2]

    result_pval_vtest_nontargets = np.empty((sigmax_space.size, T_space.size))*np.nan


    print sigmax_space
    print T_space
    print result_responses_all.shape, result_target_all.shape, result_nontargets_all.shape, result_em_fits_all.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    if plot_persigmax:
        for sigmax_i, sigmax in enumerate(sigmax_space):
            print "sigmax: ", sigmax

            # Compute the error between the response and the target
            errors_targets = utils.wrap_angles(result_responses_all[sigmax_i] - result_target_all[sigmax_i])

            errors_nontargets = np.empty(result_nontargets_all[sigmax_i].shape)
            errors_best_nontarget = np.empty(errors_targets.shape)
            for T_i in xrange(1, T_space.size):
                for repet_i in xrange(nb_repetitions):
                    # Could do a nicer indexing but fuck it

                    # Compute the error between the responses and nontargets.
                    errors_nontargets[T_i, :, :, repet_i] = utils.wrap_angles((result_responses_all[sigmax_i, T_i, :, repet_i, np.newaxis] - result_nontargets_all[sigmax_i, T_i, :, :, repet_i]))

                    # Errors between the response the best nontarget.
                    errors_best_nontarget[T_i, :, repet_i] = errors_nontargets[T_i, np.arange(errors_nontargets.shape[1]), np.nanargmin(np.abs(errors_nontargets[T_i, ..., repet_i]), axis=1), repet_i]


            for T_i, T in enumerate(T_space):
                print "T: ", T
                # Now, per T items, show the distribution of errors and of errors to non target

                # Error to target
                utils.hist_samples_density_estimation(errors_targets[T_i].reshape(nb_repetitions*N), bins=angle_space, title='Errors between response and target, N=%d' % (T))

                if savefigs:
                    dataio.save_current_figure('error_target_hist_sigmax%.2f_T%d_{label}_{unique_id}.pdf' % (sigmax, T))

                if T > 1:
                    # Error to nontarget
                    ax_handle = utils.hist_samples_density_estimation(errors_nontargets[T_i, :, :T_i].reshape(nb_repetitions*N*T_i), bins=angle_space, title='Errors between response and non targets, N=%d' % (T))

                    result_pval_vtest_nontargets[sigmax, T_i] = utils.V_test(errors_nontargets[T_i, :, :T_i].reshape(nb_repetitions*N*T_i))['pvalue']

                    print result_pval_vtest_nontargets[sigmax, T_i]

                    ax_handle.text(0.02, 0.97, "Vtest pval: %.2f" % (result_pval_vtest_nontargets[sigmax, T_i]), transform=ax_handle.transAxes, horizontalalignment='left', fontsize=12)

                    if savefigs:
                        dataio.save_current_figure('error_nontargets_hist_sigmax%.2f_T%d_{label}_{unique_id}.pdf' % (sigmax, T))

                    # Error to best non target
                    utils.hist_samples_density_estimation(errors_best_nontarget[T_i].reshape(nb_repetitions*N), bins=angle_space, title='Errors between response and best non target, N=%d' % (T))

                    if savefigs:
                        dataio.save_current_figure('error_bestnontarget_hist_sigmax%.2f_T%d_{label}_{unique_id}.pdf' % (sigmax, T))



    all_args = data_pbs.loaded_data['args_list']

    if savedata:
        dataio.save_variables_default(locals())

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='error_distribution')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs and collect responses to generate histograms. Should do it for multiple T, possibly with the parameters optimal for memory curve fits (but not that needed...)',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['sigmax', 'T'],
                     variables_to_load=['result_responses', 'result_target', 'result_nontargets', 'result_em_fits'],
                     variables_description=['Responses', 'Targets', 'Nontargets', 'Fits mixture model'],
                     post_processing=plots_errors_distribution,
                     save_output_filename='plots_errors_distribution',
                     concatenate_multiple_datasets=True
                     )




if __name__ == '__main__':

    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]

