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

    plots_pcolor_all = True
    plots_effect_M_target_kappa = False

    plots_kappa_fi_comparison = False
    plots_multiple_fisherinfo = False
    specific_plot_effect_R = False

    convert_M_realsizes = True

    plots_pcolor_realsizes_Msubs = True
    plots_pcolor_realsizes_Mtot = True

    colormap = None  # or 'cubehelix'
    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_all_precisions_mean = (utils.nanmean(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_all_precisions_std = (utils.nanstd(data_pbs.dict_arrays['result_all_precisions']['results'], axis=-1))
    result_em_fits_mean = (utils.nanmean(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_em_fits_std = (utils.nanstd(data_pbs.dict_arrays['result_em_fits']['results'], axis=-1))
    result_fisherinfo_mean = (utils.nanmean(data_pbs.dict_arrays['result_fisher_info']['results'], axis=-1))
    result_fisherinfo_std = (utils.nanstd(data_pbs.dict_arrays['result_fisher_info']['results'], axis=-1))

    result_em_fits_kappa = result_em_fits_mean[..., 0]
    result_em_fits_target = result_em_fits_mean[..., 1]
    result_em_fits_kappa_valid = np.ma.masked_where(result_em_fits_target < 0.8, result_em_fits_kappa)


    # flat versions
    result_parameters_flat = np.array(data_pbs.dict_arrays['result_all_precisions']['parameters_flat'])
    result_all_precisions_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_all_precisions']['results_flat']), axis=-1)
    result_em_fits_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_em_fits']['results_flat']), axis=-1)
    result_fisherinfor_mean_flat = np.mean(np.array(data_pbs.dict_arrays['result_fisher_info']['results_flat']), axis=-1)
    result_em_fits_kappa_flat = result_em_fits_mean_flat[..., 0]
    result_em_fits_target_flat = result_em_fits_mean_flat[..., 1]
    result_em_fits_kappa_valid_flat = np.ma.masked_where(result_em_fits_target_flat < 0.8, result_em_fits_kappa_flat)

    all_args = data_pbs.loaded_data['args_list']


    M_space = data_pbs.loaded_data['parameters_uniques']['M'].astype(int)
    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    R_space = data_pbs.loaded_data['parameters_uniques']['R'].astype(int)
    num_repetitions = generator_module.num_repetitions

    print M_space
    print ratio_space
    print R_space
    print result_all_precisions_mean.shape, result_em_fits_mean.shape

    dataio = DataIO.DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    MAX_DISTANCE = 100.

    if convert_M_realsizes:
        # alright, currently M*ratio_conj gives the conjunctive subpopulation,
        # but only floor(M_conj**1/R) neurons are really used. So we should
        # convert to M_conj_real and M_feat_real instead of M and ratio
        result_parameters_flat_subM_converted = []
        result_parameters_flat_Mtot_converted = []

        for params in result_parameters_flat:
            M = params[0]; ratio_conj = params[1]; R = int(params[2])

            M_conj_prior = int(M*ratio_conj)
            M_conj_true = int(np.floor(M_conj_prior**(1./R))**R)
            M_feat_true = int(np.floor((M-M_conj_prior)/R)*R)

            # result_parameters_flat_subM_converted contains (M_conj, M_feat, R)
            result_parameters_flat_subM_converted.append(np.array([M_conj_true, M_feat_true, R]))
            # result_parameters_flat_Mtot_converted contains (M_tot, ratio_conj, R)
            result_parameters_flat_Mtot_converted.append(np.array([float(M_conj_true+M_feat_true), M_conj_true/float(M_conj_true+M_feat_true), R]))

        result_parameters_flat_subM_converted = np.array(result_parameters_flat_subM_converted)
        result_parameters_flat_Mtot_converted = np.array(result_parameters_flat_Mtot_converted)

    if plots_pcolor_all:
        if convert_M_realsizes:
            def plot_interp(points, data, currR_indices, title='', points_label='', xlabel='', ylabel=''):
                utils.contourf_interpolate_data_interactive_maxvalue(points[currR_indices][..., :2], data[currR_indices], xlabel=xlabel, ylabel=ylabel, title='%s, R=%d' % (title, R), interpolation_numpoints=200, interpolation_method='nearest', log_scale=False)

                if savefigs:
                    dataio.save_current_figure('pcolortrueM%s_%s_R%d_log_{label}_{unique_id}.pdf' % (points_label, title, R))

            all_datas = [dict(name='precision', data=result_all_precisions_mean_flat), dict(name='kappa', data=result_em_fits_kappa_flat), dict(name='kappavalid', data=result_em_fits_kappa_valid_flat), dict(name='target', data=result_em_fits_target_flat), dict(name='fisherinfo', data=result_fisherinfor_mean_flat)]
            all_points = []
            if plots_pcolor_realsizes_Msubs:
                all_points.append(dict(name='sub', data=result_parameters_flat_subM_converted, xlabel='M_conj', ylabel='M_feat'))
            if plots_pcolor_realsizes_Mtot:
                all_points.append(dict(name='tot', data=result_parameters_flat_Mtot_converted, xlabel='Mtot', ylabel='ratio_conj'))

            for curr_points in all_points:
                for curr_data in all_datas:
                    for R_i, R in enumerate(R_space):
                        currR_indices = curr_points['data'][:, 2] == R

                        plot_interp(curr_points['data'], curr_data['data'], currR_indices, title=curr_data['name'], points_label=curr_points['name'], xlabel=curr_points['xlabel'], ylabel=curr_points['ylabel'])

                # # show precision
                # plot_interp(result_parameters_flat_subM_converted, result_all_precisions_mean_flat, currR_indices, title='precision')

                # # show kappa
                # plot_interp(result_parameters_flat_subM_converted, result_em_fits_kappa_flat, currR_indices, title='kappa')

                # plot_interp(result_parameters_flat_subM_converted, result_em_fits_kappa_valid_flat, currR_indices, title='kappavalid')

                # # show probability on target
                # plot_interp(result_parameters_flat_subM_converted, result_em_fits_target_flat, currR_indices, title='target')

                # # show fisher info
                # plot_interp(result_parameters_flat_subM_converted, result_fisherinfor_mean_flat, currR_indices, title='fisherinfo')

        else:
            # Do one pcolor for M and ratio per R
            for R_i, R in enumerate(R_space):
                # Check evolution of precision given M and ratio
                utils.pcolor_2d_data(result_all_precisions_mean[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='precision, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_precision_R%d_log_{label}_{unique_id}.pdf' % R)

                # Show kappa
                try:
                    utils.pcolor_2d_data(result_em_fits_kappa_valid[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='kappa, R=%d' % R)
                    if savefigs:
                        dataio.save_current_figure('pcolor_kappa_R%d_log_{label}_{unique_id}.pdf' % R)
                except ValueError:
                    pass

                # Show probability on target
                utils.pcolor_2d_data(result_em_fits_target[..., R_i], log_scale=False, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='target, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_target_R%d_{label}_{unique_id}.pdf' % R)

                # # Show Fisher info
                utils.pcolor_2d_data(result_fisherinfo_mean[..., R_i], log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='fisher info, R=%d' % R)
                if savefigs:
                    dataio.save_current_figure('pcolor_fisherinfo_R%d_log_{label}_{unique_id}.pdf' % R)

                plt.close('all')

    if plots_effect_M_target_kappa:
        def plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa, R):
            f, ax = plt.subplots()
            ax.plot(M_space, ratio_target_kappa_given_M)
            ax.set_xlabel('M')
            ax.set_ylabel('Optimal ratio')
            ax.set_title('Optimal Ratio for kappa %d, R=%d' % (target_kappa, R))

            if savefigs:
                dataio.save_current_figure('optratio_M_targetkappa%d_R%d_{label}_{unique_id}.pdf' % (target_kappa, R))

        target_kappas = np.array([100, 200, 300, 500, 1000, 3000])
        for R_i, R in enumerate(R_space):
            for target_kappa in target_kappas:
                dist_to_target_kappa = (result_em_fits_kappa[..., R_i] - target_kappa)**2.
                best_dist_to_target_kappa = np.argmin(dist_to_target_kappa, axis=1)
                ratio_target_kappa_given_M = np.ma.masked_where(dist_to_target_kappa[np.arange(dist_to_target_kappa.shape[0]), best_dist_to_target_kappa] > MAX_DISTANCE, ratio_space[best_dist_to_target_kappa])

                # replot
                plot_ratio_target_kappa(ratio_target_kappa_given_M, target_kappa, R)

            plt.close('all')


    if plots_kappa_fi_comparison:

        # result_em_fits_kappa and fisher info
        if True:
            for R_i, R in enumerate(R_space):
                for M_tot_selected_i, M_tot_selected in enumerate(M_space[::2]):

                    # M_conj_space = ((1.-ratio_space)*M_tot_selected).astype(int)
                    # M_feat_space = M_tot_selected - M_conj_space

                    f, axes = plt.subplots(2, 1)
                    axes[0].plot(ratio_space, result_em_fits_kappa[2*M_tot_selected_i, ..., R_i])
                    axes[0].set_xlabel('ratio')
                    axes[0].set_title('Fitted kappa')

                    axes[1].plot(ratio_space, utils.stddev_to_kappa(1./result_fisherinfo_mean[2*M_tot_selected_i, ..., R_i]**0.5))
                    axes[1].set_xlabel('ratio')
                    axes[1].set_title('kappa_FI')

                    f.suptitle('M_tot %d' % M_tot_selected, fontsize=15)
                    f.set_tight_layout(True)

                    if savefigs:
                        dataio.save_current_figure('comparison_kappa_fisher_R%d_M%d_{label}_{unique_id}.pdf' % (R, M_tot_selected))

                    plt.close(f)

        if plots_multiple_fisherinfo:
            target_fisherinfos = np.array([100, 200, 300, 500, 1000])
            for R_i, R in enumerate(R_space):
                for target_fisherinfo in target_fisherinfos:
                    dist_to_target_fisherinfo = (result_fisherinfo_mean[..., R_i] - target_fisherinfo)**2.

                    utils.pcolor_2d_data(dist_to_target_fisherinfo, log_scale=True, x=M_space, y=ratio_space, xlabel='M', ylabel='ratio', xlabel_format="%d", title='Fisher info, R=%d' % R)
                    if savefigs:
                        dataio.save_current_figure('pcolor_distfi%d_R%d_log_{label}_{unique_id}.pdf' % (target_fisherinfo, R))

                plt.close('all')

    if specific_plot_effect_R:
        # Choose a M, find which ratio gives best fit to a given kappa
        M_target = 356
        M_target_i = np.argmin(np.abs(M_space - M_target))

        utils.pcolor_2d_data(result_em_fits_kappa_valid[M_target_i], log_scale=True, x=ratio_space, y=R_space, xlabel='ratio', ylabel='R', ylabel_format="%d", title='Kappa, M %d' % (M_target))
        if savefigs:
            dataio.save_current_figure('specific_pcolor_kappa_M%d_log_{label}_{unique_id}.pdf' % (M_target))
        # target_kappa = np.ma.mean(result_em_fits_kappa_valid[M_target_i])
        # target_kappa = 5*1e3
        target_kappa = 1.2e3

        dist_target_kappa = np.abs(result_em_fits_kappa_valid[M_target_i] - target_kappa)

        utils.pcolor_2d_data(dist_target_kappa, log_scale=True, x=ratio_space, y=R_space, xlabel='ratio', ylabel='R', ylabel_format="%d", title='Kappa dist %.2f, M %d' % (target_kappa, M_target))
        if savefigs:
            dataio.save_current_figure('specific_pcolor_distkappa%d_M%d_log_{label}_{unique_id}.pdf' % (target_kappa, M_target))




    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = []

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='higher_dimensions_R')

    plt.show()

    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Runs and collect precision and mixture model fits for varying M, ratio_conj and R. Should check the effect of R',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'ratio_conj', 'R'],
                     variables_to_load=['result_all_precisions', 'result_em_fits', 'result_fisher_info'],
                     variables_description=['Precision of recall', 'Fits mixture model', 'Fisher infor'],
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

