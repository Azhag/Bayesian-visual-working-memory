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
    estimate_bootstrap = False

    should_fit_bootstrap = True
    # caching_bootstrap_filename = None
    caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_misbinding_mixed.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_bootstrap_samples_allitems_uniquekappa_sumnontarget = np.squeeze(data_pbs.dict_arrays['result_bootstrap_samples_allitems_uniquekappa_sumnontarget']['results'])
    result_bootstrap_samples = np.squeeze(data_pbs.dict_arrays['result_bootstrap_samples']['results'])
    result_bootstrap_samples_allitems_uniquekappa_allnontarget = np.squeeze(data_pbs.dict_arrays['result_bootstrap_samples_allitems_uniquekappa_allnontarget']['results'])


    ratio_space = data_pbs.loaded_data['datasets_list'][0]['ratio_space']

    print "ratio_space", ratio_space
    print result_bootstrap_samples.shape
    print result_bootstrap_samples_allitems_uniquekappa_sumnontarget.shape
    print result_bootstrap_samples_allitems_uniquekappa_allnontarget.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    if load_fit_bootstrap:
        if caching_bootstrap_filename is not None:

            if os.path.exists(caching_bootstrap_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_bootstrap_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        bootstrap_ecdf_bays_ratioconj = cached_data['bootstrap_ecdf_bays_ratioconj']
                        bootstrap_ecdf_allitems_sum_ratioconj = cached_data['bootstrap_ecdf_allitems_sum_ratioconj']
                        bootstrap_ecdf_allitems_all_ratioconj = cached_data['bootstrap_ecdf_allitems_all_ratioconj']
                        should_fit_bootstrap = False

                except IOError:
                    print "Error while loading ", caching_bootstrap_filename, "falling back to computing the EM fits"

        if should_fit_bootstrap:

            bootstrap_ecdf_bays_ratioconj = dict()
            bootstrap_ecdf_allitems_sum_ratioconj = dict()
            bootstrap_ecdf_allitems_all_ratioconj = dict()

            # Fit bootstrap
            for ratio_conj_i, ratio_conj in enumerate(ratio_space):
                # One bootstrap CDF per condition
                bootstrap_ecdf_bays = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_samples[ratio_conj_i]))
                bootstrap_ecdf_allitems_sum = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_samples_allitems_uniquekappa_sumnontarget[ratio_conj_i]))
                bootstrap_ecdf_allitems_all = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_samples_allitems_uniquekappa_allnontarget[ratio_conj_i]))

                # Store in a dict(sigmax) -> dict(T) -> ECDF object
                bootstrap_ecdf_bays_ratioconj[ratio_conj_i] = dict(ecdf=bootstrap_ecdf_bays, T=2, ratio_conj=ratio_conj)
                bootstrap_ecdf_allitems_sum_ratioconj[ratio_conj_i] = dict(ecdf=bootstrap_ecdf_allitems_sum, T=2, ratio_conj=ratio_conj)
                bootstrap_ecdf_allitems_all_ratioconj[ratio_conj_i] = dict(ecdf=bootstrap_ecdf_allitems_all, T=2, ratio_conj=ratio_conj)

            # Save everything to a file, for faster later plotting
            if caching_bootstrap_filename is not None:
                try:
                    with open(caching_bootstrap_filename, 'w') as filecache_out:
                        data_bootstrap = dict(bootstrap_ecdf_allitems_sum_ratioconj=bootstrap_ecdf_allitems_sum_ratioconj, bootstrap_ecdf_allitems_all_ratioconj=bootstrap_ecdf_allitems_all_ratioconj, bootstrap_ecdf_bays_ratioconj=bootstrap_ecdf_bays_ratioconj)
                        pickle.dump(data_bootstrap, filecache_out, protocol=2)
                except IOError:
                    print "Error writing out to caching file ", caching_bootstrap_filename


    if plots_hist_cdf:
        ## Plots now
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            # Histogram of samples
            _, axes = plt.subplots(ncols=3, figsize=(18, 6))
            axes[0].hist(utils.dropnan(result_bootstrap_samples[ratio_conj_i]), bins=100, normed='density')
            axes[0].set_xlim([0.0, 1.0])
            axes[1].hist(utils.dropnan(result_bootstrap_samples_allitems_uniquekappa_sumnontarget[ratio_conj_i]), bins=100, normed='density')
            axes[1].set_xlim([0.0, 1.0])
            axes[2].hist(utils.dropnan(result_bootstrap_samples_allitems_uniquekappa_allnontarget[ratio_conj_i]), bins=100, normed='density')
            axes[2].set_xlim([0.0, 1.0])

            if savefigs:
                dataio.save_current_figure('hist_bootstrap_ratioconj%.2f_{label}_{unique_id}.pdf' % (ratio_conj))

            # ECDF now
            _, axes = plt.subplots(ncols=3, sharey=True, figsize=(18, 6))
            axes[0].plot(bootstrap_ecdf_bays_ratioconj[ratio_conj_i]['ecdf'].x, bootstrap_ecdf_bays_ratioconj[ratio_conj_i]['ecdf'].y, linewidth=2)
            axes[0].set_xlim([0.0, 1.0])
            axes[1].plot(bootstrap_ecdf_allitems_sum_ratioconj[ratio_conj_i]['ecdf'].x, bootstrap_ecdf_allitems_sum_ratioconj[ratio_conj_i]['ecdf'].y, linewidth=2)
            axes[1].set_xlim([0.0, 1.0])
            axes[2].plot(bootstrap_ecdf_allitems_all_ratioconj[ratio_conj_i]['ecdf'].x, bootstrap_ecdf_allitems_all_ratioconj[ratio_conj_i]['ecdf'].y, linewidth=2)
            axes[2].set_xlim([0.0, 1.0])

            if savefigs:
                dataio.save_current_figure('ecdf_bootstrap_ratioconj%.2f_{label}_{unique_id}.pdf' % (ratio_conj))

    if estimate_bootstrap:
        raise NotImplementedError()
        # # model_outputs = utils.load_npy( os.path.join(os.getenv("WORKDIR_DROP", None), 'Experiments', 'bootstrap_nontargets', 'global_plots_errors_distribution-plots_errors_distribution-d977e237-cfce-473b-a292-00695e725259.npy'))

        # data_responses_all = model_outputs['result_responses_all'][..., 0]
        # data_target_all = model_outputs['result_target_all'][..., 0]
        # data_nontargets_all = model_outputs['result_nontargets_all'][..., 0]

        # # Compute bootstrap p-value
        # result_pvalue_bootstrap_sum = np.empty((ratio_space.size))*np.nan
        # result_pvalue_bootstrap_all = np.empty((ratio_space.size))*np.nan
        # for ratio_conj_i, ratio_conj in enumerate(ratio_space):
        #     bootstrap_allitems_nontargets_allitems_uniquekappa = em_circularmixture_allitems_uniquekappa.bootstrap_nontarget_stat(data_responses_all[sigmax_i, (T-1)], data_target_all[sigmax_i, (T-1)], data_nontargets_all[sigmax_i, (T-1), :, :(T-1)], sumnontargets_bootstrap_ecdf=bootstrap_ecdf_allitems_sum_ratioconj[sigmax_i][T-1]['ecdf'], allnontargets_bootstrap_ecdf=bootstrap_ecdf_allitems_all_ratioconj[sigmax_i][T-1]['ecdf'])

        #     result_pvalue_bootstrap_sum[sigmax_i, T-2] = bootstrap_allitems_nontargets_allitems_uniquekappa['p_value']
        #     result_pvalue_bootstrap_all[sigmax_i, T-2, :(T-1)] = bootstrap_allitems_nontargets_allitems_uniquekappa['allnontarget_p_value']

        #     print sigmax, T, result_pvalue_bootstrap_sum[sigmax_i, T-2], result_pvalue_bootstrap_all[sigmax_i, T-2, :(T-1)], np.sum(result_pvalue_bootstrap_all[sigmax_i, T-2, :(T-1)] < 0.05)

    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['nb_repetitions']

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
dataset_infos = dict(label='Collect bootstrap samples, using past responses from the model as target/responses (make sure its correct in the launcher itself). Hack a bit to run multiple jobs of the same parameter using the array functionality of PBS/SLURM.',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['num_repetitions'],
                     variables_to_load=['result_bootstrap_samples', 'result_bootstrap_samples_allitems_uniquekappa_sumnontarget', 'result_bootstrap_samples_allitems_uniquekappa_allnontarget'],
                     variables_description=['Bootstrap samples for Bays model', 'Bootstrap samples for model with all items, sum of mixt nontargets', 'Bootstrap samples for model with all items, all nontargets mixt'],
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

