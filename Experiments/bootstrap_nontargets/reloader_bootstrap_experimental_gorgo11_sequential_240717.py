"""
    ExperimentDescriptor to get bootstrap samples of the full item mixture model.

    Mixed population code.

    Based on Bays 2009.
"""

import os
import numpy as np
from dataio import *
import re

import matplotlib.pyplot as plt

import inspect

import utils
import cPickle as pickle

import statsmodels.distributions as stmodsdist

import load_experimental_data

import em_circularmixture
import collections
import os

from experimentlauncher import ExperimentLauncher
import re
import inspect
import imp


this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript = dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]


# Commit @04754b3


def plots_boostrap(data_pbs, generator_module=None):
    '''
        Reload bootstrap samples, plot its histogram, fit empirical CDF and save it for quicker later use.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    load_fit_bootstrap = True
    plots_hist_cdf = True
    estimate_bootstrap = True

    should_fit_bootstrap = True
    # caching_bootstrap_filename = None
    caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_gorgo11_seq.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_nontarget_bootstrap_nitems_trecall = np.squeeze(data_pbs.dict_arrays['result_nontarget_bootstrap_nitems_trecall']['results'])
    result_nontarget_bootstrap_subject_nitems_trecall = np.squeeze(data_pbs.dict_arrays['result_nontarget_bootstrap_subject_nitems_trecall']['results'])

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    dataset = load_experimental_data.load_data(
        experiment_id='gorgo11_sequential',
        fit_mixture_model=True)

    item_space = np.unique(dataset['n_items'])
    subject_space = np.unique(dataset['subject'])

    if load_fit_bootstrap:
        if caching_bootstrap_filename is not None:
            if os.path.exists(caching_bootstrap_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_bootstrap_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        nontarget_bootstrap_nitems_trecall = cached_data['nontarget_bootstrap_nitems_trecall']
                        nontarget_bootstrap_subject_nitems_trecall = cached_data['nontarget_bootstrap_subject_nitems_trecall']
                        should_fit_bootstrap = False

                except IOError:
                    print "Error while loading ", caching_bootstrap_filename, "falling back to computing the EM fits"

        if should_fit_bootstrap:
            print("\n\n>>> FITTING BOOTSTRAP <<<")
            nontarget_bootstrap_nitems_trecall = dict()
            nontarget_bootstrap_subject_nitems_trecall = dict()

            # Fit ECDF
            for n_items_i, n_items in enumerate(item_space):
                if n_items > 1:
                    for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                        print "Nitems %d, trecall %d, all subjects" % (
                            n_items, trecall)

                        current_ecdf = stmodsdist.empirical_distribution.ECDF(utils.dropnan(
                                result_nontarget_bootstrap_nitems_trecall[
                                    n_items_i, trecall_i]
                            ))

                        # Store in a dict(n_items_i) -> {ECDF object, n_items}
                        nontarget_bootstrap_nitems_trecall[
                            (n_items_i, trecall_i)] = dict(ecdf=current_ecdf,
                                              n_items=n_items,
                                              trecall=trecall)

                        for subject_i, subject in enumerate(subject_space):
                            print "Nitems %d, trecall %d, subject %d" % (
                                n_items, trecall, subject)

                            current_ecdf = stmodsdist.empirical_distribution.ECDF(utils.dropnan(
                                    result_nontarget_bootstrap_subject_nitems_trecall[
                                        subject_i, n_items_i, trecall_i]))

                            nontarget_bootstrap_subject_nitems_trecall[
                                (n_items_i, trecall_i, subject_i)] = dict(
                                    ecdf=current_ecdf,
                                    n_items=n_items,
                                    trecall=trecall,
                                    subject=subject)

            # Save everything to a file, for faster later plotting
            if caching_bootstrap_filename is not None:
                try:
                    with open(caching_bootstrap_filename, 'w') as filecache_out:
                        data_bootstrap = dict(
                            nontarget_bootstrap_nitems_trecall=nontarget_bootstrap_nitems_trecall,
                            nontarget_bootstrap_subject_nitems_trecall=nontarget_bootstrap_subject_nitems_trecall)
                        pickle.dump(data_bootstrap, filecache_out, protocol=2)
                except IOError:
                    print "Error writing out to caching file ", caching_bootstrap_filename


    if plots_hist_cdf:
        print("\n\n>>> PLOTTING HISTOGRAMS <<<")
        ## Plots now
        for n_items_i, n_items in enumerate(item_space):
            if n_items > 1:
                for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                    print "Nitems %d, trecall %d, all subjects" % (
                            n_items, trecall)
                    # Same for collapsed data accross subjects
                    # Histogram of samples, for subject/nitems
                    _, axes = plt.subplots(ncols=2, figsize=(12, 6))
                    axes[0].hist(utils.dropnan(result_nontarget_bootstrap_nitems_trecall[
                            n_items_i, trecall_i]),
                        bins=100,
                        normed='density')
                    axes[0].set_xlim([0.0, 1.0])
                    # ECDF now
                    axes[1].plot(
                        nontarget_bootstrap_nitems_trecall[(n_items_i, trecall_i)]['ecdf'].x,
                        nontarget_bootstrap_nitems_trecall[(n_items_i, trecall_i)]['ecdf'].y,
                        linewidth=2)
                    axes[1].set_xlim([0.0, 1.0])

                    if savefigs:
                        dataio.save_current_figure('histecdf_bootstrap_nitems%d_trecall%d_{label}_{unique_id}.pdf' % (
                                n_items, trecall))

                    for subject_i, subject in enumerate(subject_space):
                        # Histogram of samples, for subject/nitems
                        _, axes = plt.subplots(ncols=2, figsize=(12, 6))
                        axes[0].hist(utils.dropnan(result_nontarget_bootstrap_subject_nitems_trecall[
                                subject_i, n_items_i, trecall_i]),
                            bins=100,
                            normed='density')
                        axes[0].set_xlim([0.0, 1.0])
                        # ECDF now
                        axes[1].plot(
                            nontarget_bootstrap_subject_nitems_trecall[
                                (n_items_i, trecall_i, subject_i)]['ecdf'].x,
                            nontarget_bootstrap_subject_nitems_trecall[
                                (n_items_i, trecall_i, subject_i)]['ecdf'].y,
                            linewidth=2)
                        axes[1].set_xlim([0.0, 1.0])

                        if savefigs:
                            dataio.save_current_figure('histecdf_bootstrap_nitems%d_trecall%d_subject%d_{label}_{unique_id}.pdf' % (
                                    n_items, trecall, subject))

    if estimate_bootstrap:
        print("\n\n>>> ESTIMATING P-VALUES <<<")
        # Compute bootstrap p-value
        result_pvalue_bootstrap_nitems_trecall = np.empty((dataset['n_items_size'], dataset['n_items_size']))*np.nan
        result_pvalue_bootstrap_subject_nitems_trecall = np.empty((
            dataset['subject_size'], dataset['n_items_size'], dataset['n_items_size']))*np.nan


        for n_items_i, n_items in enumerate(item_space):
            if n_items > 1:
                for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                    print "Nitems %d, trecall %d, all subjects" % (n_items, trecall)
                    # Data collapsed accross subjects
                    ids_filtered = (
                        (dataset['n_items'] == n_items) &
                        (dataset['probe'] == trecall) &
                        (~dataset['masked'])).flatten()

                    bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                        dataset['response'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 1:n_items],
                        nontarget_bootstrap_ecdf=nontarget_bootstrap_nitems_trecall[
                                (n_items_i, trecall_i)]['ecdf'])

                    result_pvalue_bootstrap_nitems_trecall[
                        n_items_i, trecall_i] = bootstrap['p_value']
                    print "p_val:", result_pvalue_bootstrap_nitems_trecall

                    for subject_i, subject in enumerate(subject_space):
                        print "Nitems %d, trecall %d, subject %d" % (
                            n_items, trecall, subject)

                        # Bootstrap per subject and nitems
                        ids_filtered = (
                            (dataset['n_items'] == n_items) &
                            (dataset['probe'] == trecall) &
                            (dataset['subject'] == subject) &
                            (~dataset['masked'])).flatten()

                        # Compute bootstrap if required
                        bootstrap = em_circularmixture.bootstrap_nontarget_stat(
                            dataset['response'][ids_filtered, 0],
                            dataset['item_angle'][ids_filtered, 0],
                            dataset['item_angle'][ids_filtered, 1:n_items],
                            nontarget_bootstrap_ecdf=nontarget_bootstrap_subject_nitems_trecall[
                                (n_items_i, trecall_i, subject_i)]['ecdf'])

                        result_pvalue_bootstrap_subject_nitems_trecall[
                            subject_i, n_items_i, trecall_i] = bootstrap['p_value']

                        print "p_val:", result_pvalue_bootstrap_subject_nitems_trecall[
                                subject_i, n_items_i, trecall_i]


        signif_level = 0.05
        result_signif_nitems_trecall = result_pvalue_bootstrap_nitems_trecall < signif_level
        result_num_signif_subject_nitems_trecall = np.sum(result_pvalue_bootstrap_subject_nitems_trecall < signif_level, axis=0)
        print "Summary:"
        print "Collapsed subjects:", result_signif_nitems_trecall
        print "Per subjects (%d total): %s" % (dataset['subject_size'], result_num_signif_subject_nitems_trecall)


    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['nb_repetitions', 'signif_level']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='bootstrap_nontargets')


    plt.show()


    return locals()



print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Collect bootstrap samples, using past responses from the model as target/responses (make sure its correct in the launcher itself). Hack a bit to run multiple jobs of the same parameter using the array functionality of PBS/SLURM. Experimental data Bays09. Mixture model with single kappa.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['num_repetitions'],
                     variables_to_load=['result_nontarget_bootstrap_nitems_trecall', 'result_nontarget_bootstrap_subject_nitems_trecall'],
                     variables_description=['Bootstrap samples collapsed accross subject', 'Bootstrap samples, per subject and nitems'],
                     post_processing=plots_boostrap,
                     save_output_filename='plots_boostrap',
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

