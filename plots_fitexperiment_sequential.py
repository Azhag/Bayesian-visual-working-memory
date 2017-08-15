#!/usr/bin/env python
# encoding: utf-8

import collections
import matplotlib.pyplot as plt
import numpy as np
import progress
import pycircstat
import scipy.stats as spst
import utils
import warnings

# plt.rcParams['font.size'] = 24


class PlotsFitExperimentSequential(object):
    """
        This class does plots akin to paper, but for the Sequential dataset.

    """
    def __init__(self, fit_experiment_sequential,
                 do_histograms_errors_triangle=True,
                 do_mixtcurves_lasttrecall_fig6=True,
                 do_mixtcurves_collapsedpowerlaw_fig7=True,
                 ):

        self.fit_exp = fit_experiment_sequential
        self.experiment_id = self.fit_exp.experiment_id

        self.result_em_fits_stats = None
        self.do_histograms_errors_triangle = do_histograms_errors_triangle
        self.do_mixtcurves_lasttrecall_fig6 = do_mixtcurves_lasttrecall_fig6
        self.do_mixtcurves_collapsedpowerlaw_fig7 = do_mixtcurves_collapsedpowerlaw_fig7

        print "Doing Sequential plots for %s. \nHist %d, Fig6 %d, Fig7 %d" % (
            self.experiment_id,
            self.do_histograms_errors_triangle,
            self.do_mixtcurves_lasttrecall_fig6,
            self.do_mixtcurves_collapsedpowerlaw_fig7
        )


    def do_plots(self):
        '''
            Do all plots for that FitExperimentAllT.

            These correspond to a particular experiment_id only, not multiple.
        '''
        if self.do_histograms_errors_triangle:
            self.plots_histograms_errors_triangle()
        if self.do_mixtcurves_lasttrecall_fig6:
            self.plots_mixtcurves_lasttrecall_fig6()
        if self.do_mixtcurves_collapsedpowerlaw_fig7:
            self.plots_mixtcurves_collapsedpowerlaw_fig7()


    def plots_histograms_errors_triangle(self, size=12):
        '''
            Histograms of errors, for all n_items/trecall conditions.
        '''

        # Do the plots
        f, axes = plt.subplots(
            ncols=self.fit_exp.T_space.size,
            nrows=2*self.fit_exp.T_space.size,
            figsize=(size, 2*size))

        angle_space = np.linspace(-np.pi, np.pi, 51)
        for n_items_i, n_items in enumerate(self.fit_exp.T_space):
            for trecall_i, trecall in enumerate(self.fit_exp.T_space):
                if trecall <= n_items:
                    print "\n=== N items: {}, trecall: {}".format(
                        n_items, trecall)

                    # Sample
                    self.fit_exp.setup_experimental_stimuli(n_items, trecall)

                    if 'samples' in self.fit_exp.get_names_stored_responses():
                        self.fit_exp.restore_responses('samples')
                    else:
                        self.fit_exp.sampler.force_sampling_round()
                        self.fit_exp.store_responses('samples')

                    responses, targets, nontargets = (
                        self.fit_exp.sampler.collect_responses())

                    # Targets
                    errors_targets = utils.wrap_angles(targets - responses)
                    utils.hist_angular_data(
                        errors_targets,
                        bins=angle_space,
                        # title='N=%d, trecall=%d' % (n_items, trecall),
                        norm='density',
                        ax_handle=axes[2*n_items_i, trecall_i],
                        pretty_xticks=False)
                    axes[2*n_items_i, trecall_i].set_ylim([0., 1.4])
                    axes[2*n_items_i, trecall_i].xaxis.set_major_locator(
                        plt.NullLocator())
                    axes[2*n_items_i, trecall_i].yaxis.set_major_locator(
                        plt.NullLocator())

                    # Nontargets
                    if n_items > 1:
                        errors_nontargets = utils.wrap_angles((
                            responses[:, np.newaxis] - nontargets).flatten())

                        utils.hist_angular_data(
                            errors_nontargets,
                            bins=angle_space,
                            # title='Nontarget %s N=%d' % (dataset['name'], n_items),
                            norm='density',
                            ax_handle=axes[2*n_items_i + 1, trecall_i],
                            pretty_xticks=False)

                        axes[2*n_items_i + 1, trecall_i].set_ylim([0., 0.3])

                    axes[2*n_items_i + 1, trecall_i].xaxis.set_major_locator(plt.NullLocator())
                    axes[2*n_items_i + 1, trecall_i].yaxis.set_major_locator(plt.NullLocator())
                else:
                    axes[2*n_items_i, trecall_i].axis('off')
                    axes[2*n_items_i + 1, trecall_i].axis('off')

        return axes


    def plots_mixtcurves_lasttrecall_fig6(self,
                                          num_repetitions=1,
                                          use_cache=True,
                                          use_sem=True,
                                          size=6):
        '''
            Plots memory fidelity and mixture proportions for the last item
            recall.

            This reproduces Figure 6 in Gorgo 11.
        '''

        if self.collapsed_em_fits is None or not use_cache:
            # Collect all data to fit.
            T = self.fit_exp.T_space.size

            search_progress = progress.Progress(
                T*(T + 1)/2.*num_repetitions)

            params_fit_double_all = []
            for repet_i in xrange(num_repetitions):
                model_data_dict = {
                    'responses': np.nan*np.empty((T,
                                                T,
                                                self.fit_exp.num_datapoints)),
                    'targets': np.nan*np.empty((T,
                                                T,
                                                self.fit_exp.num_datapoints)),
                    'nontargets': np.nan*np.empty((T,
                                                T,
                                                self.fit_exp.num_datapoints,
                                                T - 1))}

                for n_items_i, n_items in enumerate(self.fit_exp.T_space):
                    for trecall_i, trecall in enumerate(self.fit_exp.T_space):
                        if trecall <= n_items:
                            self.fit_exp.setup_experimental_stimuli(n_items,
                                                                    trecall)

                            print "== Fit for N={}, trecall={}. %d/%d".format(
                                n_items, trecall, repet_i+1, num_repetitions)
                            print "%.2f%%, %s left - %s" % (
                                search_progress.percentage(),
                                search_progress.time_remaining_str(),
                                search_progress.eta_str())

                            if ('samples' in self.fit_exp.get_names_stored_responses() and
                                repet_i < 1):
                                self.fit_exp.restore_responses('samples')
                            else:
                                self.fit_exp.sampler.force_sampling_round()
                                self.fit_exp.store_responses('samples')

                            responses, targets, nontargets = (
                                self.fit_exp.sampler.collect_responses())

                            # collect all data
                            model_data_dict['responses'][
                                n_items_i,
                                trecall_i] = responses
                            model_data_dict['targets'][
                                n_items_i,
                                trecall_i] = targets
                            model_data_dict['nontargets'][
                                n_items_i,
                                trecall_i,
                                :,
                                :n_items_i] = nontargets

                            search_progress.increment()

                # Fit the collapsed mixture model
                params_fit_double = (
                    em_circularmixture_parametrickappa_doublepowerlaw.fit(
                        self.fit_exp.T_space,
                        model_data_dict['responses'],
                        model_data_dict['targets'],
                        model_data_dict['nontargets'],
                        debug=True))
                params_fit_double_all.append(params_fit_double)

            # Get statistics of powerlaw fits
            self.collapsed_em_fits = collections.defaultdict(dict)
            emfits_keys = params_fit_double.keys()
            for key in emfits_keys:
                repets_param_fit_curr = [
                    param_fit_double[key]
                    for param_fit_double in params_fit_double_all]
                self.collapsed_em_fits['mean'][key] = np.mean(
                    repets_param_fit_curr, axis=0)
                self.collapsed_em_fits['std'][key] = np.std(
                    repets_param_fit_curr, axis=0)
                self.collapsed_em_fits['sem'][key] = (
                    self.collapsed_em_fits['std'][key] / np.sqrt(
                        num_repetitions))

        # Do the plot
        if use_sem:
            errorbars = 'sem'
        else:
            errorbars = 'std'

        f1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        plot_kappa_mean_error(
            self.collapsed_em_fits['mean'][
                'kappa'][:, 0],
            self.collapsed_em_fits[errorbars][
                'kappa'][:, 0],
            xlabel='items', ylabel='Kappa', ax=axes1[0])
        plot_emmixture_mean_error(
            T_space_exp,
            self.collapsed_em_fits['mean'][
                'mixt_target_tr'][:, 0],
            self.collapsed_em_fits[errorbars][
                'mixt_target_tr'][:, 0],
            label='Target', ax=axes1[1])
        plot_emmixture_mean_error(
            T_space_exp,
            self.collapsed_em_fits['mean'][
                'mixt_nontargets_tr'][:, 0],
            self.collapsed_em_fits[errorbars][
                'mixt_nontargets_tr'][:, 0],
            label='Nontargets', ax=axes1[1])
        plot_emmixture_mean_error(
            T_space_exp,
            self.collapsed_em_fits['mean'][
                'mixt_random_tr'][:, 0],
            self.collapsed_em_fits[errorbars][
                'mixt_random_tr'][:, 0],
            label='Random',
            xlabel='items', ylabel='Mixture proportions', ax=axes1[1])

        f1.suptitle('Fig 6: Last trecall')


    # Memory curve kappa
    def __plot_memory_fidelity(self, model_em_fits, suptitle_text=None, ax=None):
        '''
            Nice plot for the memory fidelity, as in Fig6 of the paper theo

            Changes to using the subject fits if FitExperimentAllTSubject used.
        '''

        T_space = self.fit_exp.T_space
        data_em_fits = self.fit_exp.get_em_fits_arrays()

        if ax is None:
            _, ax = plt.subplots()
        else:
            ax.hold(False)

        ax = utils.plot_mean_std_area(
            T_space,
            data_em_fits['mean'][0],
            data_em_fits['std'][0],
            linewidth=3, fmt='o-', markersize=8,
            label='Data',
            ax_handle=ax
        )

        ax.hold(True)

        ax = utils.plot_mean_std_area(
            T_space,
            model_em_fits['mean'][..., 0],
            model_em_fits['std'][..., 0],
            xlabel='Number of items',
            ylabel="Memory error $[rad^{-2}]$",
            linewidth=3,
            fmt='o-', markersize=8,
            label='Model',
            ax_handle=ax
        )

        ax.legend(loc='upper right',
                  bbox_to_anchor=(1., 1.)
                  )
        ax.set_xlim([0.9, T_space.max()+0.1])
        ax.set_xticks(range(1, T_space.max()+1))
        ax.set_xticklabels(range(1, T_space.max()+1))

        if suptitle_text:
            ax.set_title(suptitle_text)
            # ax.get_figure().suptitle(suptitle_text)
        ax.hold(False)
        ax.get_figure().canvas.draw()

        return ax


    def __plot_mixtcurves(self, model_em_fits, suptitle_text=None, ax=None):
        '''
            Similar kind of plot, but showing the mixture proportions, as in Figure13
        '''
        T_space = self.fit_exp.T_space
        data_em_fits = self.fit_exp.get_em_fits_arrays()

        if ax is None:
            _, ax = plt.subplots()
        else:
            ax.hold(False)


        model_em_fits['mean'][np.isnan(model_em_fits['mean'])] = 0.0
        model_em_fits['std'][np.isnan(model_em_fits['std'])] = 0.0

        # Show model fits
        utils.plot_mean_std_area(
            T_space,
            model_em_fits['mean'][..., 1],
            model_em_fits['std'][..., 1],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=3, fmt='o-', markersize=5,
            label='Target',
        )
        ax.hold(True)
        utils.plot_mean_std_area(
            T_space,
            model_em_fits['mean'][..., 2],
            model_em_fits['std'][..., 2],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=3, fmt='o-', markersize=5,
            label='Nontarget'
        )
        utils.plot_mean_std_area(
            T_space,
            model_em_fits['mean'][..., 3],
            model_em_fits['std'][..., 3],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=3, fmt='o-', markersize=5,
            label='Random'
        )

        # Now data
        utils.plot_mean_std_area(
            T_space,
            data_em_fits['mean'][1],
            data_em_fits['std'][1],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=2, fmt='o--', markersize=5,
            label='Data target'
        )
        utils.plot_mean_std_area(
            T_space,
            data_em_fits['mean'][2],
            data_em_fits['std'][2],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=2, fmt='o--', markersize=5, label='Data nontarget'
        )
        utils.plot_mean_std_area(
            T_space,
            data_em_fits['mean'][3],
            data_em_fits['std'][3],
            xlabel='Number of items',
            ylabel="Mixture probabilities",
            ax_handle=ax, linewidth=2, fmt='o--', markersize=5, label='Data random'
        )

        ax.legend(loc='upper left',
                  bbox_to_anchor=(1., 1.)
                  )

        ax.set_xlim([0.9, T_space.max() + 0.1])
        ax.set_ylim([0.0, 1.1])
        ax.set_xticks(range(1, T_space.max() + 1))
        ax.set_xticklabels(range(1, T_space.max() + 1))

        if suptitle_text:
            ax.set_title(suptitle_text)
            # ax.get_figure().suptitle(suptitle_text)

        ax.get_figure().canvas.draw()

        ax.hold(False)

        return ax
