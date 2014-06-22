"""
    ExperimentDescriptor to get bootstrap samples of the full item mixture model.

    Mixed population code.

    Based on Bays 2009.
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

import load_experimental_data

import em_circularmixture_allitems_uniquekappa

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
    caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_bays09.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_bootstrap_nitems_samples = np.squeeze(data_pbs.dict_arrays['result_bootstrap_nitems_samples']['results'])
    result_bootstrap_subject_nitems_samples = np.squeeze(data_pbs.dict_arrays['result_bootstrap_subject_nitems_samples']['results'])


    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    dataset = load_experimental_data.load_data_bays09(fit_mixture_model=True)

    if load_fit_bootstrap:
        if caching_bootstrap_filename is not None:

            if os.path.exists(caching_bootstrap_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_bootstrap_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        bootstrap_nitems_samples = cached_data['bootstrap_nitems_samples']
                        bootstrap_subject_nitems_samples = cached_data['bootstrap_subject_nitems_samples']
                        should_fit_bootstrap = False

                except IOError:
                    print "Error while loading ", caching_bootstrap_filename, "falling back to computing the EM fits"

        if should_fit_bootstrap:

            bootstrap_nitems_samples = dict()
            bootstrap_subject_nitems_samples = dict()

            # Fit ECDF
            for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
                if n_items > 1:
                    print "Nitems %d, all subjects" % (n_items)
                    current_ecdf_allitems = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_nitems_samples[n_items_i]))

                    # Store in a dict(n_items_i) -> {ECDF object, n_items}
                    bootstrap_nitems_samples[n_items_i] = dict(ecdf=current_ecdf_allitems, n_items=n_items)

                    for subject_i, subject in enumerate(np.unique(dataset['subject'])):
                        print "Nitems %d, subject %d" % (n_items, subject)

                        current_ecdf_subj_items = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_subject_nitems_samples[subject_i, n_items_i]))

                        if n_items_i not in bootstrap_subject_nitems_samples:
                            bootstrap_subject_nitems_samples[n_items_i] = dict()
                        bootstrap_subject_nitems_samples[n_items_i][subject_i] = dict(ecdf=current_ecdf_subj_items, n_items=n_items, subject=subject)

            # Save everything to a file, for faster later plotting
            if caching_bootstrap_filename is not None:
                try:
                    with open(caching_bootstrap_filename, 'w') as filecache_out:
                        data_bootstrap = dict(bootstrap_nitems_samples=bootstrap_nitems_samples, bootstrap_subject_nitems_samples=bootstrap_subject_nitems_samples)
                        pickle.dump(data_bootstrap, filecache_out, protocol=2)
                except IOError:
                    print "Error writing out to caching file ", caching_bootstrap_filename


    if plots_hist_cdf:
        ## Plots now
        for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
            if n_items > 1:
                for subject_i, subject in enumerate(np.unique(dataset['subject'])):

                    # Histogram of samples, for subject/nitems
                    _, axes = plt.subplots(ncols=2, figsize=(12, 6))
                    axes[0].hist(utils.dropnan(result_bootstrap_subject_nitems_samples[subject_i, n_items_i]), bins=100, normed='density')
                    axes[0].set_xlim([0.0, 1.0])
                    # ECDF now
                    axes[1].plot(bootstrap_subject_nitems_samples[n_items_i][subject_i]['ecdf'].x, bootstrap_subject_nitems_samples[n_items_i][subject_i]['ecdf'].y, linewidth=2)
                    axes[1].set_xlim([0.0, 1.0])

                    if savefigs:
                        dataio.save_current_figure('histecdf_bootstrap_nitems%d_subject%d_{label}_{unique_id}.pdf' % (n_items, subject))

                # Same for collapsed data accross subjects
                # Histogram of samples, for subject/nitems
                _, axes = plt.subplots(ncols=2, figsize=(12, 6))
                axes[0].hist(utils.dropnan(result_bootstrap_nitems_samples[n_items_i]), bins=100, normed='density')
                axes[0].set_xlim([0.0, 1.0])
                # ECDF now
                axes[1].plot(bootstrap_nitems_samples[n_items_i]['ecdf'].x, bootstrap_nitems_samples[n_items_i]['ecdf'].y, linewidth=2)
                axes[1].set_xlim([0.0, 1.0])

                if savefigs:
                    dataio.save_current_figure('histecdf_bootstrap_nitems%d_{label}_{unique_id}.pdf' % (n_items))

    if estimate_bootstrap:
        # Compute bootstrap p-value
        result_pvalue_bootstrap_nitems = np.empty(dataset['n_items_size'])*np.nan
        result_pvalue_bootstrap_subject_nitems_samples = np.empty((dataset['n_items_size'], dataset['subject_size']))*np.nan


        for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
            if n_items > 1:
                print "Nitems %d, all subjects" % (n_items)
                # Data collapsed accross subjects
                ids_filtered = (dataset['n_items'] == n_items).flatten()

                bootstrap = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(
                    dataset['response'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 0],
                    dataset['item_angle'][ids_filtered, 1:n_items],
                    sumnontargets_bootstrap_ecdf=bootstrap_nitems_samples[n_items_i]['ecdf']
                    )

                result_pvalue_bootstrap_nitems[n_items_i] = bootstrap['p_value']
                print "p_val:", result_pvalue_bootstrap_nitems

                for subject_i, subject in enumerate(np.unique(dataset['subject'])):
                    print "Nitems %d, subject %d" % (n_items, subject)

                    # Bootstrap per subject and nitems
                    ids_filtered = (dataset['subject']==subject).flatten() & (dataset['n_items'] == n_items).flatten()

                    # Get pvalue
                    bootstrap = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(
                        dataset['response'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 0],
                        dataset['item_angle'][ids_filtered, 1:n_items],
                        sumnontargets_bootstrap_ecdf=bootstrap_subject_nitems_samples[n_items_i][subject_i]['ecdf'])
                    result_pvalue_bootstrap_subject_nitems_samples[n_items_i, subject_i] = bootstrap['p_value']

                    print "p_val:", result_pvalue_bootstrap_subject_nitems_samples[n_items_i, subject_i]

        signif_level = 0.05
        result_signif_nitems = result_pvalue_bootstrap_nitems < signif_level
        result_num_signif_subject_nitems = np.sum(result_pvalue_bootstrap_subject_nitems_samples < signif_level, axis=1)
        print "Summary:"
        print "Collapsed subjects:",result_signif_nitems
        print "Per subjects (%d total): %s" % (dataset['subject_size'], result_num_signif_subject_nitems)


    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['nb_repetitions', 'signif_level']

    if savedata:
        dataio.save_variables_default(locals(), variables_to_save)

        dataio.make_link_output_to_dropbox(dropbox_current_experiment_folder='bootstrap_nontargets')


    plt.show()


    return locals()



this_file = inspect.getfile(inspect.currentframe())

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", os.path.split(this_file)[-1])[-1]
# generator_script = 'generator_specific_stimuli_mixed_fixedemfit_otherrange_201113.py'

print "Reloader data generated from ", generator_script

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Collect bootstrap samples, using past responses from the model as target/responses (make sure its correct in the launcher itself). Hack a bit to run multiple jobs of the same parameter using the array functionality of PBS/SLURM. Experimental data Bays09. Mixture model with single kappa.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['num_repetitions'],
                     variables_to_load=['result_bootstrap_nitems_samples', 'result_bootstrap_subject_nitems_samples'],
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

