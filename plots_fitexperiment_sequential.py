#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import progress
import utils
import scipy.stats as spst
import pycircstat
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

                    (responses, targets, nontargets) = self.fit_exp.sampler.collect_responses()

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
                                          size=6):
        '''
            Plots the memory fidelity for all T and the mixture proportions for all T
        '''

        if self.result_em_fits_stats is None or not use_cache:
            search_progress = progress.Progress(self.fit_exp.T_space.size*num_repetitions)

            # kappa, mixt_target, mixt_nontarget, mixt_random, ll
            result_em_fits = np.nan*np.ones((self.fit_exp.T_space.size, 5, num_repetitions))

            for T_i, T in enumerate(self.fit_exp.T_space):
                self.fit_exp.setup_experimental_stimuli(T, trecall)

                for repet_i in xrange(num_repetitions):
                    print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())
                    print "Fit for T=%d, %d/%d" % (T, repet_i+1, num_repetitions)

                    if 'samples' in self.fit_exp.get_names_stored_responses() and repet_i < 1:
                        self.fit_exp.restore_responses('samples')
                    else:
                        self.fit_exp.sampler.force_sampling_round()
                        self.fit_exp.store_responses('samples')

                    # Fit mixture model
                    curr_params_fit = self.fit_exp.sampler.fit_mixture_model(use_all_targets=False)
                    result_em_fits[T_i, :, repet_i] = [curr_params_fit[key]
                        for key in ('kappa', 'mixt_target',
                                    'mixt_nontargets_sum', 'mixt_random',
                                    'train_LL')]

                    search_progress.increment()

            # Get stats of EM Fits
            self.result_em_fits_stats = dict(
                mean=utils.nanmean(result_em_fits, axis=-1),
                std=utils.nanstd(result_em_fits, axis=-1)
            )

        if self.do_memcurves_fig6 and self.do_mixtcurves_fig13:
            f, axes = plt.subplots(nrows=2, figsize=(size, size*2))
        else:
            f, ax = plt.subplots(figsize=(size, size))
            axes = [ax]

        if self.do_memcurves_fig6:
            self.__plot_memcurves(self.result_em_fits_stats,
                                  suptitle_text='Memory fidelity',
                                  ax=axes[ax_i]
                                  )
            ax_i += 1


        if self.do_mixtcurves_fig13:
            self.__plot_mixtcurves(self.result_em_fits_stats,
                                   suptitle_text='Mixture proportions',
                                   ax=axes[ax_i]
                                   )
            ax_i += 1

        return axes



    # def plots_KS_comparison_fig2fig5(self,
    #                                  bins=41,
    #                                  show_pval=True,
    #                                  size=6
    #                                  ):
    #     '''
    #         Will plot the ECDF of data/samples and then do Kolmogorov-Smirnov /Kuiper 2-samples tests on them.
    #     '''

    #     f, axes = plt.subplots(nrows=2,
    #                            ncols=self.fit_exp.T_space.size,
    #                            figsize=(size*2, size)
    #                            )

    #     result_KS = dict(human=dict(),
    #                      model=dict(),
    #                      ks_pval=dict(),
    #                      kuiper_pval=dict()
    #                      )

    #     warnings.simplefilter("ignore")

    #     for t_i, T in enumerate(self.fit_exp.T_space):
    #         result_KS['ks_pval'][T] = dict()
    #         result_KS['kuiper_pval'][T] = dict()

    #         self.fit_exp.setup_experimental_stimuli(T)

    #         # Human histograms and CDF
    #         self.fit_exp.restore_responses('human')
    #         result_KS['human'][T] = self.fit_exp.sampler.compute_errors_alltargets_histograms(bins=bins)

    #         # Samples histograms and CDF
    #         if 'samples' in self.fit_exp.get_names_stored_responses():
    #             self.fit_exp.restore_responses('samples')
    #         else:
    #             self.fit_exp.sampler.force_sampling_round()
    #             self.fit_exp.store_responses('samples')
    #         result_KS['model'][T] = self.fit_exp.sampler.compute_errors_alltargets_histograms(bins=bins)

    #         # Compute K-S 2-samples tests stats
    #         for condition in ['targets', 'nontargets']:
    #             if condition in result_KS['human'][T]:
    #                 ks_out = spst.ks_2samp(
    #                     result_KS['human'][T][condition]['samples'],
    #                     result_KS['model'][T][condition]['samples']
    #                 )

    #                 result_KS['ks_pval'][T][condition] = \
    #                     ks_out.pvalue

    #                 result_KS['kuiper_pval'][T][condition] = \
    #                     pycircstat.tests.kuiper(
    #                         result_KS['human'][T][condition]['samples'],
    #                         result_KS['model'][T][condition]['samples'])[0][0]

    #         ### Plot everything
    #         axes[0, t_i].plot(result_KS['human'][T]['targets']['x'],
    #                           result_KS['human'][T]['targets']['ecdf'],
    #                           label='data'
    #                           )
    #         axes[0, t_i].plot(result_KS['model'][T]['targets']['x'],
    #                           result_KS['model'][T]['targets']['ecdf'],
    #                           label='model'
    #                           )
    #         axes[0, t_i].set_title('')
    #         axes[0, t_i].set_xlim((-np.pi, np.pi))
    #         # axes[0, t_i].set_ylim((0, 2))

    #         if show_pval:
    #             axes[0, t_i].text(
    #                 0.02, 0.99,
    #                 "KS p: %.2f" % result_KS['ks_pval'][T]['targets'],
    #                 transform=axes[0, t_i].transAxes,
    #                 horizontalalignment='left',
    #                 verticalalignment='top'
    #             )
    #             axes[0, t_i].text(
    #                 0.02, 0.9,
    #                 "Kuiper p: %.2f" % result_KS['kuiper_pval'][T]['targets'],
    #                 transform=axes[0, t_i].transAxes,
    #                 horizontalalignment='left',
    #                 verticalalignment='top'
    #             )

    #         if T > 1:
    #             axes[1, t_i].plot(result_KS['human'][T]['nontargets']['x'],
    #                               result_KS['human'][T]['nontargets']['ecdf'],
    #                               label='data'
    #                               )
    #             axes[1, t_i].plot(result_KS['model'][T]['nontargets']['x'],
    #                               result_KS['model'][T]['nontargets']['ecdf'],
    #                               label='model'
    #                               )
    #             axes[1, t_i].set_title('')
    #             axes[1, t_i].set_xlim((-np.pi, np.pi))

    #             if show_pval:
    #                 axes[1, t_i].text(
    #                     0.02, 0.99,
    #                     "KS p: %.2f" % result_KS['ks_pval'][T]['nontargets'],
    #                     transform=axes[1, t_i].transAxes,
    #                     horizontalalignment='left',
    #                     verticalalignment='top'
    #                 )
    #                 axes[1, t_i].text(
    #                     0.02, 0.9,
    #                     "Kuiper p: %.2f" % result_KS['kuiper_pval'][T]['nontargets'],
    #                     transform=axes[1, t_i].transAxes,
    #                     horizontalalignment='left',
    #                     verticalalignment='top'
    #                 )
    #         else:
    #             axes[1, t_i].axis('off')

    #     axes[1, 1].legend(loc='center',
    #                       bbox_to_anchor=(0.5, 0.5),
    #                       bbox_transform=axes[1, 0].transAxes
    #                       )

    #     f.suptitle('ECDF between human and model')

    #     return axes, result_KS


    # Memory curve kappa
    def __plot_memcurves(self, model_em_fits, suptitle_text=None, ax=None):
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
