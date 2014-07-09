"""
    ExperimentDescriptor to fit Memory curves using a Mixed population code

    Uses the new Marginal Inverse Fisher Information, and some new code altogether.
"""

import os
import numpy as np
import experimentlauncher
import dataio as DataIO
import utils
import re
import imp
import matplotlib.pyplot as plt
import launchers
import cPickle as pickle

import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import em_circularmixture_allitems_kappafi

import inspect

import load_experimental_data

import scipy.interpolate as spint

# Commit @8c49507 +


def plots_ratioMscaling(data_pbs, generator_module=None):
    '''
        Reload and plot precision/fits of a Mixed code.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_pcolor_all = False
    plots_effect_M_target_precision = False
    plots_multiple_precisions = False

    plots_effect_M_target_kappa = False

    plots_subpopulations_effects = False

    plots_subpopulations_effects_kappa_fi = True
    compute_fisher_info_perratioconj = True
    caching_fisherinfo_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'cache_fisherinfo.pickle')

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = (utils.nanmean(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_all_precisions_std = (utils.nanstd(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_em_fits_mean = (utils.nanmean(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_em_fits_std = (utils.nanstd(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))

    all_args = data_pbs.loaded_data['args_list']

    result_em_fits_kappa = result_em_fits_mean[..., 0]

    M_space = data_pbs.loaded_data['parameters_uniques']['M'].astype(int)
    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    num_repetitions = generator_module.num_repetitions

    print M_space
    print ratio_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    target_precision = 100.
    dist_to_target_precision = (result_all_precisions_mean - target_precision)**2.
    best_dist_to_target_precision = np.argmin(dist_to_target_precision, axis=1)
    MAX_DISTANCE = 100.

    ratio_target_precision_given_M = np.ma.masked_where(dist_to_target_precision[np.arange(dist_to_target_precision.shape[0]), best_dist_to_target_precision] > MAX_DISTANCE, ratio_space[best_dist_to_target_precision])

    if plots_pcolor_all:
        # Check evolution of precision given M and ratio
        utils.pcolor_2d_data(result_all_precisions_mean, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='precision wrt M / ratio')
        if savefigs:
            dataio.save_current_figure('precision_log_pcolor_{label}_{unique_id}.pdf')

        # See distance to target precision evolution
        utils.pcolor_2d_data(dist_to_target_precision, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='Dist to target precision %d' % target_precision)
        if savefigs:
            dataio.save_current_figure('dist_targetprecision_log_pcolor_{label}_{unique_id}.pdf')


        # Show kappa
        utils.pcolor_2d_data(result_em_fits_kappa, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='kappa wrt M / ratio')
        if savefigs:
            dataio.save_current_figure('kappa_log_pcolor_{label}_{unique_id}.pdf')

        utils.pcolor_2d_data((result_em_fits_kappa - 200)**2., log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='dist to kappa')
        if savefigs:
            dataio.save_current_figure('dist_kappa_log_pcolor_{label}_{unique_id}.pdf')


    if plots_effect_M_target_precision:
        def plot_ratio_target_precision(ratio_target_precision_given_M, target_precision):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_precision_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for precison %d' % target_precision)

            if savefigs:
                dataio.save_current_figure('effect_ratio_M_targetprecision%d_{label}_{unique_id}.pdf' % target_precision)

        plot_ratio_target_precision(ratio_target_precision_given_M, target_precision)

        if plots_multiple_precisions:
            target_precisions = np.array([100, 200, 300, 500, 1000])
            for target_precision in target_precisions:
                dist_to_target_precision = (result_all_precisions_mean - target_precision)**2.
                best_dist_to_target_precision = np.argmin(dist_to_target_precision, axis=1)
                ratio_target_precision_given_M = np.ma.masked_where(dist_to_target_precision[np.arange(dist_to_target_precision.shape[0]), best_dist_to_target_precision] > MAX_DISTANCE, ratio_space[best_dist_to_target_precision])

                # replot
                plot_ratio_target_precision(ratio_target_precision_given_M, target_precision)

    if plots_effect_M_target_kappa:
        def plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_kappa_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for precison %d' % target_kappa)

            if savefigs:
                dataio.save_current_figure('effect_ratio_M_targetkappa%d_{label}_{unique_id}.pdf' % target_kappa)

        target_kappa = np.array([100, 200, 300, 500, 1000, 3000])
        for target_kappa in target_kappa:
            dist_to_target_kappa = (result_em_fits_kappa - target_kappa)**2.
            best_dist_to_target_kappa = np.argmin(dist_to_target_kappa, axis=1)
            ratio_target_kappa_given_M = np.ma.masked_where(dist_to_target_kappa[np.arange(dist_to_target_kappa.shape[0]), best_dist_to_target_kappa] > MAX_DISTANCE, ratio_space[best_dist_to_target_kappa])

            # replot
            plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa)

    if plots_subpopulations_effects:
        # result_all_precisions_mean
        for M_tot_selected_i, M_tot_selected in enumerate(M_space[::2]):

            M_conj_space = ((1.-ratio_space)*M_tot_selected).astype(int)
            M_feat_space = M_tot_selected - M_conj_space

            f, axes = plt.subplots(2, 2)
            axes[0, 0].plot(ratio_space, result_all_precisions_mean[2*M_tot_selected_i])
            axes[0, 0].set_xlabel('ratio')
            axes[0, 0].set_title('Measured precision')

            axes[1, 0].plot(ratio_space, M_conj_space**2*M_feat_space)
            axes[1, 0].set_xlabel('M_feat_size')
            axes[1, 0].set_title('M_c**2*M_f')

            axes[0, 1].plot(ratio_space, M_conj_space**2.)
            axes[0, 1].set_xlabel('M')
            axes[0, 1].set_title('M_c**2')

            axes[1, 1].plot(ratio_space, M_feat_space)
            axes[1, 1].set_xlabel('M')
            axes[1, 1].set_title('M_f')

            f.suptitle('M_tot %d' % M_tot_selected, fontsize=15)
            f.set_tight_layout(True)

            if savefigs:
                dataio.save_current_figure('scaling_precision_subpop_Mtot%d_{label}_{unique_id}.pdf' % M_tot_selected)

            plt.close(f)

    if plots_subpopulations_effects_kappa_fi:
        # From cache
        if caching_fisherinfo_filename is not None:
            if os.path.exists(caching_fisherinfo_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_fisherinfo_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        result_fisherinfo_Mratio = cached_data['result_fisherinfo_Mratio']
                        compute_fisher_info_perratioconj = False

                except IOError:
                    print "Error while loading ", caching_fisherinfo_filename, "falling back to computing the Fisher Info"

        if compute_fisher_info_perratioconj:
            # We did not save the Fisher info, but need it if we want to fit the mixture model with fixed kappa. So recompute them using the args_dicts

            result_fisherinfo_Mratio = np.empty((M_space.size, ratio_space.size))

            # Invert the all_args_i -> M, ratio_conj direction
            parameters_indirections = data_pbs.loaded_data['parameters_dataset_index']

            for M_i, M in enumerate(M_space):
                for ratio_conj_i, ratio_conj in enumerate(ratio_space):
                    # Get index of first dataset with the current ratio_conj (no need for the others, I think)
                    try:
                        arg_index = parameters_indirections[(M, ratio_conj)][0]

                        # Now using this dataset, reconstruct a RandomFactorialNetwork and compute the fisher info
                        curr_args = all_args[arg_index]

                        # curr_args['stimuli_generation'] = lambda T: np.linspace(-np.pi*0.6, np.pi*0.6, T)

                        (_, _, _, sampler) = launchers.init_everything(curr_args)

                        # Theo Fisher info
                        result_fisherinfo_Mratio[M_i, ratio_conj_i] = sampler.estimate_fisher_info_theocov()

                        # del curr_args['stimuli_generation']
                    except KeyError:
                        result_fisherinfo_Mratio[M_i, ratio_conj_i] = np.nan


            # Save everything to a file, for faster later plotting
            if caching_fisherinfo_filename is not None:
                try:
                    with open(caching_fisherinfo_filename, 'w') as filecache_out:
                        data_cache = dict(result_fisherinfo_Mratio=result_fisherinfo_Mratio)
                        pickle.dump(data_cache, filecache_out, protocol=2)
                except IOError:
                    print "Error writing out to caching file ", caching_fisherinfo_filename

        # result_em_fits_kappa
        if False:

            for M_tot_selected_i, M_tot_selected in enumerate(M_space[::2]):

                M_conj_space = ((1.-ratio_space)*M_tot_selected).astype(int)
                M_feat_space = M_tot_selected - M_conj_space

                f, axes = plt.subplots(2, 2)
                axes[0, 0].plot(ratio_space, result_em_fits_kappa[2*M_tot_selected_i])
                axes[0, 0].set_xlabel('ratio')
                axes[0, 0].set_title('Fitted kappa')

                axes[1, 0].plot(ratio_space, utils.stddev_to_kappa(1./result_fisherinfo_Mratio[2*M_tot_selected_i]**0.5))
                axes[1, 0].set_xlabel('M_feat_size')
                axes[1, 0].set_title('kappa_FI_mixed')

                f.suptitle('M_tot %d' % M_tot_selected, fontsize=15)
                f.set_tight_layout(True)

                if savefigs:
                    dataio.save_current_figure('scaling_kappa_subpop_Mtot%d_{label}_{unique_id}.pdf' % M_tot_selected)

                plt.close(f)

        utils.pcolor_2d_data((result_fisherinfo_Mratio- 2000)**2., log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='Fisher info')
        if savefigs:
            dataio.save_current_figure('dist2000_fi_log_pcolor_{label}_{unique_id}.pdf')




    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='ratio_scaling_M')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs and collect precision and mixture model fits for varying M and ratio_conj. Should then look for a specific precision/kappa and see how ratio_conj evolves with M',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj'],
                     variables_to_load=['result_all_precisions', 'result_em_fits'],
                     variables_description=['Precision of recall', 'Fits mixture model'],
                     post_processing=plots_ratioMscaling,
                     save_output_filename='plots_ratioMscaling',
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

