import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import matplotlib.patches as plt_patches
# import matplotlib.gridspec as plt_grid
import os
import os.path
import cPickle as pickle
# import bottleneck as bn
import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import pandas as pd

import dataio as DataIO

import utils


def plots_check_bias_nontarget(dataset, dataio=None):
    '''
        Get an histogram of the errors between the response and all non targets
            If biased towards 0-values, should indicate misbinding errors.

        (if you do this with respect to all targets, it's retarded and should always be biased)
    '''
    n_items_space = np.unique(dataset['n_items'])
    angle_space = np.linspace(-np.pi, np.pi, 20)

    # Get histograms of errors, per n_item
    for nitems_i in xrange(n_items_space.size):
        utils.hist_samples_density_estimation(dataset['errors_nitems'][nitems_i], bins=angle_space, title='%s N=%d' % (dataset['name'], n_items_space[nitems_i]), dataio=dataio, filename='hist_bias_targets_%ditems_{label}_{unique_id}.pdf' % (n_items_space[nitems_i]))

    # Get histograms of bias to nontargets. Do that by binning the errors to others nontargets of the array.
    utils.plot_hists_bias_nontargets(dataset['errors_all_nitems'][n_items_space>1], bins=20, dataio=dataio, label='allnontargets', remove_first_column=True)

    rayleigh_test = utils.rayleigh_test(dataset['errors_all_nitems'][n_items_space>1].flatten())
    v_test = utils.V_test(dataset['errors_all_nitems'][n_items_space>1].flatten())
    print rayleigh_test
    print v_test



def plots_check_bias_bestnontarget(dataset, dataio=None):
    '''
        Get an histogram of errors between response and best nontarget.
        Should be more biased towards 0 than the overall average
    '''
    n_items_space = np.unique(dataset['n_items'])

    # Compute the errors to the best non target
    errors_nontargets = dataset['errors_all_nitems'][n_items_space>1]
    errors_nontargets = np.array([errors_nontargets_nitem[~np.all(np.isnan(errors_nontargets_nitem), axis=1), :] for errors_nontargets_nitem in errors_nontargets])

    indices_bestnontarget = [np.nanargmin(np.abs(errors_nontargets[n_item_i][..., 1:]), axis=-1) for n_item_i in xrange(errors_nontargets.shape[0])]
    # indices_bestnontarget = np.nanargmin(np.abs(errors_nontargets), axis=2)

    # Index of the argmin of absolute error. Not too bad, easy to index into.
    errors_bestnontargets_nitems = np.array([ errors_nontargets[n_items_i][ xrange(errors_nontargets[n_items_i].shape[0]), indices_bestnontarget[n_items_i] + 1]   for n_items_i in xrange(errors_nontargets.shape[0]) ])

    # Show histograms per n_items, like in Bays2009 figure
    utils.plot_hists_bias_nontargets(errors_bestnontargets_nitems, bins=20, label='bestnontarget', dataio=dataio)



def plots_check_bias_nontarget_randomized(dataset, dataio=None):
    '''
        Plot the histogram of errors to nontargets, after replacing all nontargets by random angles.
        If show similar bias, would be indication of low predictive power of distribution of errors to nontargets.
    '''

    n_items_space = np.unique(dataset['n_items'])

    # Copy item_angles
    new_item_angles = dataset['item_angle'].copy()

    # Will resample multiple times
    errors_nitems_new_dict = dict()
    nb_resampling = 100

    for resampling_i in xrange(nb_resampling):

        # Replace nontargets randomly
        nontarget_indices = np.nonzero(~np.isnan(new_item_angles[:, 1:]))
        new_item_angles[nontarget_indices[0], nontarget_indices[1]+1] = 2*np.pi*np.random.random(nontarget_indices[0].size) - np.pi

        # Compute errors
        new_all_errors = utils.wrap_angles(new_item_angles - dataset['response'], bound=np.pi)

        for n_items in n_items_space:
            ids_filtered = (dataset['n_items'] == n_items).flatten()

            if n_items in errors_nitems_new_dict:
                errors_nitems_new_dict[n_items] = np.r_[errors_nitems_new_dict[n_items], new_all_errors[ids_filtered]]
            else:
                errors_nitems_new_dict[n_items] = new_all_errors[ids_filtered]

    errors_nitems_new = np.array([val for key, val in errors_nitems_new_dict.items()])

    utils.plot_hists_bias_nontargets(errors_nitems_new[n_items_space>1], bins=20, label='allnontarget_randomized_%dresamplings' % nb_resampling, dataio=dataio, remove_first_column=True)

    ### Do same for best non targets
    # TODO Convert this for data_dualrecall
    errors_nontargets = errors_nitems_new[1:, :, 1:]
    indices_bestnontarget = np.nanargmin(np.abs(errors_nontargets), axis=2)

    # Index of the argmin of absolute error. Not too bad, easy to index into.
    errors_bestnontargets_nitems = np.array([ errors_nontargets[n_items_i, xrange(errors_nontargets.shape[1]), indices_bestnontarget[n_items_i]]   for n_items_i in xrange(errors_nontargets.shape[0]) ])

    # Show histograms
    utils.plot_hists_bias_nontargets(errors_bestnontargets_nitems, bins=20, label='bestnontarget_randomized_%dresamplings' % nb_resampling, dataio=dataio)




def plots_check_oblique_effect(data, nb_bins=100):
    '''
        Humans are more precise for vertical and horizontal bars than diagonal orientations.

        Check if present.
    '''

    # Construct the list of (target angles, errors), see if there is some structure in that
    errors_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['error'][:, 0]))

    # response_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['response']))
    # response_per_colour = np.array(zip(data['item_colour'][np.arange(data['probe'].size), data['probe'][:, 0]], data['response']))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')

    plt.figure()
    plt.plot(errors_per_angle[:, 0], np.abs(errors_per_angle[:, 1]), 'x')

    discrete_x = np.linspace(-np.pi/2., np.pi/2., nb_bins)
    avg_error = np.zeros(discrete_x.shape)
    std_error = np.zeros(discrete_x.shape)

    for x_i in np.arange(discrete_x.size):
        if x_i < discrete_x.size - 1:
            # Check what data comes in the current interval x[x_i, x_i+1]
            avg_error[x_i] = utils.mean_angles(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])
            std_error[x_i] = utils.angle_circular_std_dev(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])

    plt.figure()
    plt.plot(discrete_x, avg_error)

    plt.figure()
    plt.plot(discrete_x, avg_error**2.)

    plt.figure()
    plt.plot(discrete_x, np.abs(avg_error))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')
    plt.plot(discrete_x, avg_error, 'ro')


def plots_histograms_errors_targets_nontargets_nitems(dataset, dataio=None):
    '''
        Create subplots showing histograms of errors to targets and nontargets

        Adds Vtest texts on the nontargets
    '''

    angle_space = np.linspace(-np.pi, np.pi, 51)
    bins_center = angle_space[:-1] + np.diff(angle_space)[0]/2

    # Histogram, collapsing across subjects
    f1, axes1 = plt.subplots(ncols=dataset['n_items_size'], figsize=(dataset['n_items_size']*6, 6), sharey=True)
    f2, axes2 = plt.subplots(ncols=dataset['n_items_size']-1, figsize=((dataset['n_items_size']-1)*6, 6), sharey=True)

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        utils.hist_angular_data(dataset['errors_nitems'][n_items_i], bins=angle_space, title='%s N=%d' % (dataset['name'], n_items), norm='density', ax_handle=axes1[n_items_i], pretty_xticks=True)
        axes1[n_items_i].set_ylim([0., 2.0])

        if n_items > 1:
            utils.hist_angular_data(utils.dropnan(dataset['errors_nontarget_nitems'][n_items_i]), bins=angle_space, title='%s N=%d' % (dataset['name'], n_items), norm='density', ax_handle=axes2[n_items_i-1], pretty_xticks=True)

            axes2[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (dataset['vtest_nitems'][n_items_i]), transform=axes2[n_items_i-1].transAxes, horizontalalignment='left', fontsize=13)

            axes2[n_items_i - 1].set_ylim([0., 0.3])



    f1.canvas.draw()
    f2.canvas.draw()

    if dataio is not None:
        plt.figure(f1.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_target_all_{label}_{unique_id}.pdf")
        plt.figure(f2.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_nontarget_all_{label}_{unique_id}.pdf")

    # Do per subject and nitems, using average histogram
    f3, axes3 = plt.subplots(ncols=dataset['n_items_size'], figsize=(dataset['n_items_size']*6, 6), sharey=True)
    f4, axes4 = plt.subplots(ncols=dataset['n_items_size']-1, figsize=((dataset['n_items_size']-1)*6, 6), sharey=True)

    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):

        axes3[n_items_i].bar(bins_center, dataset['hist_cnts_target_nitems_stats']['mean'][n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=dataset['hist_cnts_target_nitems_stats']['sem'][n_items_i])
        # axes3[n_items_i].set_title('N=%d' % n_items)
        axes3[n_items_i].set_xlim([bins_center[0]-np.pi/(angle_space.size-1), bins_center[-1]+np.pi/(angle_space.size-1)])
        axes3[n_items_i].set_ylim([0., 2.0])
        axes3[n_items_i].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        axes3[n_items_i].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=16)

        if n_items > 1:
            axes4[n_items_i-1].bar(bins_center, dataset['hist_cnts_nontarget_nitems_stats']['mean'][n_items_i], width=2.*np.pi/(angle_space.size-1), align='center', yerr=dataset['hist_cnts_nontarget_nitems_stats']['sem'][n_items_i])
            # axes4[n_items_i-1].set_title('N=%d' % n_items)
            axes4[n_items_i-1].set_xlim([bins_center[0]-np.pi/(angle_space.size-1), bins_center[-1]+np.pi/(angle_space.size-1)])

            # axes4[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (pvalue_nontarget_subject_nitems_mean[n_items_i]), transform=axes4[n_items_i-1].transAxes, horizontalalignment='left', fontsize=13)
            axes4[n_items_i-1].text(0.02, 0.96, "Vtest pval: %.4f" % (dataset['vtest_nitems'][n_items_i]), transform=axes4[n_items_i-1].transAxes, horizontalalignment='left', fontsize=14)

            # TODO Add bootstrap there instead.

            axes4[n_items_i-1].set_ylim([0., 0.3])
            axes4[n_items_i-1].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            axes4[n_items_i-1].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=16)

        utils.scatter_marginals(utils.dropnan(dataset['data_to_fit'][n_items]['item_features'][:, 0, 0]), utils.dropnan(dataset['data_to_fit'][n_items]['response']), xlabel ='Target angle', ylabel='Response angle', title='%s histogram responses, %d items' % (dataset['name'], n_items), figsize=(9, 9), factor_axis=1.1, bins=61)
        # utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['3_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)
        # utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['6_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)

    f3.canvas.draw()
    f4.canvas.draw()

    if dataio is not None:
        plt.figure(f3.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_target_persubj_{label}_{unique_id}.pdf")
        plt.figure(f4.number)
        # plt.tight_layout()
        dataio.save_current_figure("hist_error_nontarget_persubj_{label}_{unique_id}.pdf")


def plots_em_mixtures(dataset, dataio=None, use_sem=True):
    '''
        Do plots for the mixture models and kappa
    '''
    T_space_exp = np.unique(dataset['n_items'])

    f, ax = plt.subplots()

    if use_sem:
        errorbars = 'sem'
    else:
        errorbars = 'std'

    # Mixture probabilities
    utils.plot_mean_std_area(T_space_exp, dataset['em_fits_nitems_arrays']['mean'][1], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][1]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Target')
    utils.plot_mean_std_area(T_space_exp, np.ma.masked_invalid(dataset['em_fits_nitems_arrays']['mean'][2]).filled(0.0), np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][2]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Nontarget')
    utils.plot_mean_std_area(T_space_exp, dataset['em_fits_nitems_arrays']['mean'][3], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][3]).filled(0.0), xlabel='Number of items', ylabel="Mixture probabilities", ax_handle=ax, linewidth=3, fmt='o-', markersize=5, label='Random')

    ax.legend(prop={'size':15})

    ax.set_title('Mixture model for EM fit %s' % dataset['name'])
    ax.set_xlim([1.0, T_space_exp.max()])
    ax.set_ylim([0.0, 1.1])
    ax.set_xticks(range(1, T_space_exp.max()+1))
    ax.set_xticklabels(range(1, T_space_exp.max()+1))

    f.canvas.draw()

    if dataio is not None:
        dataio.save_current_figure('emfits_mixtures_{label}_{unique_id}.pdf')

    # Kappa
    f, ax = plt.subplots()

    ax = utils.plot_mean_std_area(T_space_exp,
    dataset['em_fits_nitems_arrays']['mean'][0], np.ma.masked_invalid(dataset['em_fits_nitems_arrays'][errorbars][0]).filled(0.0), linewidth=3, fmt='o-', markersize=8, ylabel='Experimental data', ax_handle=ax)

    ax.legend(prop={'size':15})
    ax.set_title('Kappa for EM fit %s' % dataset['name'])
    ax.set_xlim([0.9, T_space_exp.max()+0.1])
    ax.set_ylim([0.0, np.max(dataset['em_fits_nitems_arrays']['mean'][0])*1.1])
    ax.set_xticks(range(1, T_space_exp.max()+1))
    ax.set_xticklabels(range(1, T_space_exp.max()+1))
    ax.get_figure().canvas.draw()

    if dataio is not None:
        dataio.save_current_figure('emfits_kappa_{label}_{unique_id}.pdf')

def plots_precision(dataset, dataio=None, use_sem=True):
    '''
        Do plots for the mixture models and kappa
    '''
    T_space_exp = np.unique(dataset['n_items'])

    precisions_to_plot = [['precision_subject_nitems_theo', 'Precision Theo'],['precision_subject_nitems_bays_notreatment', 'Precision BaysNoTreat'],['precision_subject_nitems_bays', 'Precision Bays'],['precision_subject_nitems_theo_nochance', 'Precision TheoNoChance']]

    for precision_to_plot, precision_title in precisions_to_plot:
        f, ax = plt.subplots()

        # Compute the errorbars
        precision_mean = np.mean(dataset[precision_to_plot], axis=0)
        precision_errors = np.std(dataset[precision_to_plot], axis=0)
        if use_sem:
            precision_errors /= np.sqrt(dataset['subject_size'])

        # Now show the precision
        utils.plot_mean_std_area(T_space_exp, precision_mean, precision_errors, xlabel='Number of items', label="Precision", ax_handle=ax, linewidth=3, fmt='o-', markersize=5)

        ax.legend(prop={'size':15})

        ax.set_title('%s %s' % (precision_title, dataset['name']))
        ax.set_xlim([1.0, T_space_exp.max()])
        ax.set_ylim([0.0, np.max(precision_mean)+np.max(precision_errors)])
        ax.set_xticks(range(1, T_space_exp.max()+1))
        ax.set_xticklabels(range(1, T_space_exp.max()+1))

        f.canvas.draw()

        if dataio is not None:
            dataio.save_current_figure('%s_{label}_{unique_id}.pdf' % precision_title)


def plots_bays2009(dataset, dataio=None):
    '''

    Some plots for the Bays2009 data
    '''

    plots_histograms_errors_targets_nontargets_nitems(dataset, dataio)

    plots_precision(dataset, dataio)

    plots_em_mixtures(dataset, dataio)


def plots_gorgo11(dataset, dataio=None):
    '''
        Plots for Gorgo11, assuming sequential data
    '''
    plots_histograms_errors_targets_nontargets_nitems(dataset, dataio)

    plots_precision(dataset, dataio)

    plots_em_mixtures(dataset, dataio)


def plots_dualrecall(dataset):
    '''
        Create plots for the double recall dataset
    '''

    to_plot = {'resp_vs_targ':True, 'error_boxplot':True, 'resp_rating':True, 'em_fits':True, 'loglik':True, 'resp_distrib':True, 'resp_conds':True}

    dataset_pd = dataset['panda']

    dataset_pd['error_abs'] = dataset_pd.error.abs()

    # Show distributions of responses wrt target angles/colour
    if to_plot['resp_vs_targ']:

        # Plot scatter and marginals for the orientation trials
        utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['3_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)
        utils.scatter_marginals(utils.dropnan(dataset['item_angle'][dataset['angle_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_angle'][dataset['angle_trials'] & dataset['6_items_trials']]), xlabel ='Target angle', ylabel='Response angle', title='%s Angle trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61)

        # Plot scatter and marginals for the colour trials
        utils.scatter_marginals(utils.dropnan(dataset['item_colour'][dataset['colour_trials']& dataset['3_items_trials'], 0]), utils.dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['3_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='%s Colour trials, 3 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)
        utils.scatter_marginals(utils.dropnan(dataset['item_colour'][dataset['colour_trials'] & dataset['6_items_trials'], 0]), utils.dropnan(dataset['probe_colour'][dataset['colour_trials'] & dataset['6_items_trials']]), xlabel ='Target colour', ylabel='Response colour', title='%s Colour trials, 6 items' % (dataset['name']), figsize=(9, 9), factor_axis=1.1, bins=61, show_colours=True)


    if 'em_fits' in dataset:

        # dataset_pd[ids_filtered][ids_targets_responses].boxplot('error_angle_abs', by='rating')
        # dataset_pd[ids_filtered][ids_nontargets_responses].boxplot('error_angle_abs', by='rating')

        if to_plot['error_boxplot']:
            dataset_pd.boxplot(column=['error_abs'], by=['cond', 'n_items', 'rating'])

        # for i in dataset_pd.subject.unique():
        #     dataset_pd[dataset_pd.subject == i].boxplot(column=['error_angle'], by=['n_items', 'rating'])

        # Show distribution responsibility as a function of rating
        if to_plot['resp_rating']:
            # dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 3.0].dropna(subset=['error']).groupby(['rating'])
            dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 6.0][dataset_pd.cond == 1.].dropna(subset=['error']).groupby(['rating'])
            _, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
            i = 0
            bins = np.linspace(0., 1.0, 31)
            for name, group in dataset_grouped_nona_rating:
                print name

                # Compute histograms and normalize per rating condition
                counts_target, bins_edges = np.histogram(group.resp_target, bins=bins)
                counts_nontarget, bins_edges = np.histogram(group.resp_nontarget, bins=bins)
                counts_random, bins_edges = np.histogram(group.resp_random, bins=bins)
                dedges = np.diff(bins_edges)[0]

                sum_counts = float(np.sum(counts_target) + np.sum(counts_nontarget) + np.sum(counts_random))
                counts_target = counts_target/sum_counts
                counts_nontarget = counts_nontarget/sum_counts
                counts_random = counts_random/sum_counts

                # Print Responsibility target density estimation
                # group.resp_target.plot(kind='kde', ax=axes[i, 0])
                axes[i, 0].bar(bins_edges[:-1], counts_target, dedges, color='b')
                axes[i, 0].set_xlim((0.0, 1.0))
                axes[i, 0].set_ylim((0.0, 0.35))
                axes[i, 0].text(0.5, 0.8, "T " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)

                # Print Responsibility nontarget density estimation
                # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
                axes[i, 1].bar(bins_edges[:-1], counts_nontarget, dedges, color='g')
                axes[i, 1].set_xlim((0.0, 1.0))
                axes[i, 1].set_ylim((0.0, 0.35))
                axes[i, 1].text(0.5, 0.8, "NT " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)

                # Print Responsibility random density estimation
                # group.resp_random.plot(kind='kde', ax=axes[i, 1])
                axes[i, 2].bar(bins_edges[:-1], counts_random, dedges, color='r')
                axes[i, 2].set_xlim((0.0, 1.0))
                axes[i, 2].set_ylim((0.0, 0.35))
                axes[i, 2].text(0.5, 0.8, "R " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)

                i += 1

            plt.suptitle("Colour trials")

        dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 6.0][dataset_pd.cond == 2.].dropna(subset=['error']).groupby(['rating'])
        f, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
        i = 0
        bins = np.linspace(0., 1.0, 31)
        for name, group in dataset_grouped_nona_rating:
            print name

            # Compute histograms and normalize per rating condition
            counts_target, bins_edges = np.histogram(group.resp_target, bins=bins)
            counts_nontarget, bins_edges = np.histogram(group.resp_nontarget, bins=bins)
            counts_random, bins_edges = np.histogram(group.resp_random, bins=bins)
            dedges = np.diff(bins_edges)[0]

            sum_counts = float(np.sum(counts_target) + np.sum(counts_nontarget) + np.sum(counts_random))
            counts_target = counts_target/sum_counts
            counts_nontarget = counts_nontarget/sum_counts
            counts_random = counts_random/sum_counts

            # Print Responsibility target density estimation
            # group.resp_target.plot(kind='kde', ax=axes[i, 0])
            axes[i, 0].bar(bins_edges[:-1], counts_target, dedges, color='b')
            axes[i, 0].set_xlim((0.0, 1.0))
            axes[i, 0].set_ylim((0.0, 0.35))
            axes[i, 0].text(0.5, 0.8, "T " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)

            # Print Responsibility nontarget density estimation
            # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
            axes[i, 1].bar(bins_edges[:-1], counts_nontarget, dedges, color='g')
            axes[i, 1].set_xlim((0.0, 1.0))
            axes[i, 1].set_ylim((0.0, 0.35))
            axes[i, 1].text(0.5, 0.8, "NT " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)

            # Print Responsibility random density estimation
            # group.resp_random.plot(kind='kde', ax=axes[i, 1])
            axes[i, 2].bar(bins_edges[:-1], counts_random, dedges, color='r')
            axes[i, 2].set_xlim((0.0, 1.0))
            axes[i, 2].set_ylim((0.0, 0.35))
            axes[i, 2].text(0.5, 0.8, "R " + str(name), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)

            i += 1
        plt.suptitle("Angle trials")


        # Add condition names
        dataset_pd['cond_name'] = np.array(['Colour', 'Angle'])[np.array(dataset_pd['cond']-1, dtype=int)]

        # Regroup some data
        dataset_grouped_nona_conditems = dataset_pd.dropna(subset=['error']).groupby(['cond_name', 'n_items'])
        dataset_grouped_nona_conditems_mean = dataset_grouped_nona_conditems.mean()[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa', 'train_LL', 'test_LL']]

        # Show inferred mixture proportions and kappa
        if to_plot['em_fits']:
            # ax = dataset_grouped_nona_conditems_mean[['mixt_target', 'mixt_nontarget', 'mixt_random', 'kappa']].plot(secondary_y='kappa', kind='bar')
            ax = dataset_grouped_nona_conditems_mean[['mixt_target', 'mixt_nontarget', 'mixt_random']].plot(kind='bar')
            ax.set_ylabel('Mixture proportions')

            ax = dataset_grouped_nona_conditems_mean[['kappa']].plot(kind='bar')
            ax.set_ylabel('Kappa')

        # Show loglihood of fit
        if to_plot['loglik']:
            f, ax = plt.subplots(1, 1)
            dataset_grouped_nona_conditems_mean[['train_LL', 'test_LL']].plot(kind='bar', ax=ax, secondary_y='test_LL')

        # Show boxplot of responsibilities
        if to_plot['resp_distrib']:
            dataset_grouped_nona_conditems.boxplot(column=['resp_target', 'resp_nontarget', 'resp_random'])

        # Show distributions of responsibilities
        if to_plot['resp_conds']:
            f, axes = plt.subplots(dataset_pd.cond_name.nunique()*dataset_pd.n_items.nunique(), 3)
            i = 0
            bins = np.linspace(0., 1.0, 31)
            for name, group in dataset_grouped_nona_conditems:
                print name

                # Print Responsibility target density estimation
                # group.resp_target.plot(kind='kde', ax=axes[i, 0])
                group.resp_target.hist(ax=axes[i, 0], color='b', bins=bins)
                axes[i, 0].text(0.5, 0.85, "T " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 0].transAxes)
                axes[i, 0].set_xlim((0.0, 1.0))

                # Print Responsibility nontarget density estimation
                # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
                group.resp_nontarget.hist(ax=axes[i, 1], color='g', bins=bins)
                axes[i, 1].text(0.5, 0.85, "NT " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 1].transAxes)
                axes[i, 1].set_xlim((0.0, 1.0))

                # Print Responsibility random density estimation
                # group.resp_random.plot(kind='kde', ax=axes[i, 1])
                group.resp_random.hist(ax=axes[i, 2], color='r', bins=bins)
                axes[i, 2].text(0.5, 0.85, "R " + ' '.join([str(x) for x in name]), fontweight='bold', horizontalalignment='center', transform = axes[i, 2].transAxes)
                axes[i, 2].set_xlim((0.0, 1.0))

                i += 1

        # Extract some parameters
        fitted_parameters = dataset_grouped_nona_conditems_mean.iloc[0].loc[['kappa', 'mixt_target', 'mixt_nontarget', 'mixt_random']]
        print fitted_parameters


def plot_bias_close_feature(dataset, dataio=None):
    '''
        Check if there is a bias in the response towards closest item (either closest wrt cued feature, or wrt all features)
    '''
    number_items_considered = 2

    # Error to nontarget
    bias_to_nontarget = np.abs(dataset['errors_nontarget_nitems'][number_items_considered-1][:, :number_items_considered-1].flatten())
    bias_to_target = np.abs(dataset['errors_nitems'][number_items_considered-1].flatten())
    ratio_biases = bias_to_nontarget/ bias_to_target
    response = dataset['data_to_fit'][number_items_considered]['response']

    target = dataset['data_to_fit'][number_items_considered]['item_features'][:, 0]
    nontarget = dataset['data_to_fit'][number_items_considered]['item_features'][:, 1]

    # Distance between probe and closest nontarget, in full feature space
    dist_target_nontarget_torus = utils.dist_torus(target, nontarget)

    # Distance only looking at recalled feature
    dist_target_nontarget_recalled = np.abs(utils.wrap_angles((target[:, 0] - nontarget[:, 0])))

    # Distance only looking at cued feature.
    # Needs more work. They are only a few possible values, so we can group them and get a boxplot for each
    dist_target_nontarget_cue = np.round(np.abs(utils.wrap_angles((target[:, 1] - nontarget[:, 1]))), decimals=8)
    dist_distinct_values = np.unique(dist_target_nontarget_cue)
    bias_to_nontarget_grouped_dist_cue = []
    for dist_value in dist_distinct_values:
        bias_to_nontarget_grouped_dist_cue.append(bias_to_nontarget[dist_target_nontarget_cue == dist_value])

    # Check if the response is closer to the target or nontarget, in relative terms.
    # Need to compute a ratio linking bias_to_target and bias_to_nontarget.
    # Two possibilities: response was between target and nontarget, or response was "behind" the target.
    ratio_response_close_to_nontarget = bias_to_nontarget/dist_target_nontarget_recalled
    indices_filter_other_side = bias_to_nontarget > dist_target_nontarget_recalled
    ratio_response_close_to_nontarget[indices_filter_other_side] = bias_to_nontarget[indices_filter_other_side]/(dist_target_nontarget_recalled[indices_filter_other_side] + bias_to_target[indices_filter_other_side])

    f, ax = plt.subplots(2, 2)
    ax[0, 0].plot(dist_target_nontarget_torus, bias_to_nontarget, 'x')
    ax[0, 0].set_xlabel('Distance full feature space')
    ax[0, 0].set_ylabel('Error to nontarget')

    ax[0, 1].boxplot(bias_to_nontarget_grouped_dist_cue, positions=dist_distinct_values)
    ax[0, 1].set_ylabel('Error to nontarget')
    ax[0, 1].set_xlabel('Distance cued feature only')

    # ax[1, 0].plot(dist_target_nontarget_recalled, np.ma.masked_greater(ratio_biases, 100), 'x')
    ax[1, 0].plot(dist_target_nontarget_recalled, bias_to_nontarget, 'x')
    # ax[1, 0].plot(dist_target_nontarget_recalled, np.ma.masked_greater(bias_to_nontarget/dist_target_nontarget_recalled, 30), 'x')
    ax[1, 0].set_xlabel('Distance recalled feature only')
    ax[1, 0].set_ylabel('Error to nontarget')

    ax[1, 1].plot(dist_target_nontarget_recalled, ratio_response_close_to_nontarget, 'x')
    ax[1, 1].set_xlabel('Distance recalled feature only')
    ax[1, 1].set_ylabel('Normalised distance to nontarget')


    f.suptitle('Effect of distance between items on bias of response towards nontarget')

    if dataio:
        f.set_size_inches(16, 16, forward=True)
        dataio.save_current_figure('plot_bias_close_feature_{label}_{unique_id}.pdf')

