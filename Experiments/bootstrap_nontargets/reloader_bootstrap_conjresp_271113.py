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

# Commit @2042319 +


def plots_boostrap(data_pbs, generator_module=None):
    '''
        Reload bootstrap samples, plot its histogram, fit empirical CDF and save it for quicker later use.
    '''

    #### SETUP
    #
    savefigs = True
    savedata = True

    plots_hist_cdf = True

    should_fit_bootstrap = True
    # caching_bootstrap_filename = None
    caching_bootstrap_filename = os.path.join(generator_module.pbs_submission_infos['simul_out_dir'], 'outputs', 'cache_bootstrap_conjresp.pickle')

    plt.rcParams['font.size'] = 16
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_bootstrap_samples_allitems = np.squeeze(data_pbs.dict_arrays['result_bootstrap_samples_allitems']['results'])
    result_bootstrap_samples = np.squeeze(data_pbs.dict_arrays['result_bootstrap_samples']['results'])


    sigmax_space = data_pbs.loaded_data['datasets_list'][0]['sigmax_space']
    T_space = data_pbs.loaded_data['datasets_list'][0]['T_space']

    print result_bootstrap_samples_allitems.shape
    print result_bootstrap_samples.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])


    if plots_hist_cdf:
        # Do one plot per min distance.

        # # Plot Log-likelihood of Mixture model, sanity check
        # utils.pcolor_2d_data(result_em_fits_mean[min_dist_i, ..., -1].T, x=ratio_space, y=sigmax_space, xlabel='ratio', ylabel='sigma_x', title='EM loglik, min_dist=%.3f' % min_dist, log_scale=False)
        # if savefigs:
        #     dataio.save_current_figure('em_loglik_permindist_mindist%.2f_ratiosigmax_{label}_{unique_id}.pdf' % min_dist)


        if caching_bootstrap_filename is not None:

            if os.path.exists(caching_bootstrap_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_bootstrap_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        bootstrap_ecdf_allitems_sigmax_T = cached_data['bootstrap_ecdf_allitems_sigmax_T']
                        bootstrap_ecdf_bays_sigmax_T = cached_data['bootstrap_ecdf_bays_sigmax_T']
                        should_fit_bootstrap = False

                except IOError:
                    print "Error while loading ", caching_bootstrap_filename, "falling back to computing the EM fits"

        if should_fit_bootstrap:

            bootstrap_ecdf_allitems_sigmax_T = dict()
            bootstrap_ecdf_bays_sigmax_T = dict()

            # Fit bootstrap
            for sigmax_i, sigmax in enumerate(sigmax_space):
                for T_i, T in enumerate(T_space):
                    if T>1:
                        # One bootstrap CDF per condition
                        bootstrap_ecdf_allitems = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_samples_allitems[sigmax_i, T_i]))
                        bootstrap_ecdf_bays = stmodsdist.empirical_distribution.ECDF(utils.dropnan(result_bootstrap_samples[sigmax_i, T_i]))

                        # Store in a dict(sigmax) -> dict(T) -> ECDF object
                        bootstrap_ecdf_allitems_sigmax_T.setdefault(sigmax_i, dict())[T_i] = dict(ecdf=bootstrap_ecdf_allitems, T=T, sigmax=sigmax)
                        bootstrap_ecdf_bays_sigmax_T.setdefault(sigmax_i, dict())[T_i] = dict(ecdf=bootstrap_ecdf_bays, T=T, sigmax=sigmax)


            # Save everything to a file, for faster later plotting
            if caching_bootstrap_filename is not None:
                try:
                    with open(caching_bootstrap_filename, 'w') as filecache_out:
                        data_bootstrap = dict(bootstrap_ecdf_allitems_sigmax_T=bootstrap_ecdf_allitems_sigmax_T, bootstrap_ecdf_bays_sigmax_T=bootstrap_ecdf_bays_sigmax_T)
                        pickle.dump(data_bootstrap, filecache_out, protocol=2)
                except IOError:
                    print "Error writing out to caching file ", caching_bootstrap_filename


        ## Plots now
        for sigmax_i, sigmax in enumerate(sigmax_space):
            for T_i, T in enumerate(T_space):
                if T > 1:
                    # Histogram of samples
                    _, axes = plt.subplots(ncols=2, figsize=(12, 6))
                    axes[0].hist(utils.dropnan(result_bootstrap_samples[sigmax_i, T_i]), bins=100, normed='density')
                    axes[0].set_xlim([0.0, 1.0])
                    axes[1].hist(utils.dropnan(result_bootstrap_samples_allitems[sigmax_i, T_i]), bins=100, normed='density')
                    axes[1].set_xlim([0.0, 1.0])

                    if savefigs:
                        dataio.save_current_figure('hist_bootstrap_sigmax%.2f_T%d_{label}_{unique_id}.pdf' % (sigmax, T))

                    # ECDF now
                    _, axes = plt.subplots(ncols=2, sharey=True, figsize=(12, 6))
                    axes[0].plot(bootstrap_ecdf_bays_sigmax_T[sigmax_i][T_i]['ecdf'].x, bootstrap_ecdf_bays_sigmax_T[sigmax_i][T_i]['ecdf'].y, linewidth=2)
                    axes[0].set_xlim([0.0, 1.0])
                    axes[1].plot(bootstrap_ecdf_allitems_sigmax_T[sigmax_i][T_i]['ecdf'].x, bootstrap_ecdf_allitems_sigmax_T[sigmax_i][T_i]['ecdf'].y, linewidth=2)
                    axes[1].set_xlim([0.0, 1.0])

                    if savefigs:
                        dataio.save_current_figure('ecdf_bootstrap_sigmax%.2f_T%d_{label}_{unique_id}.pdf' % (sigmax, T))


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
                     parameters=['enforce_min_distance', 'sigmax'],
                     variables_to_load=['result_bootstrap_samples_allitems', 'result_bootstrap_samples'],
                     variables_description=['Bootstrap samples for model with all items', 'Bootstrap samples for Bays model'],
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

