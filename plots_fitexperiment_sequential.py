#!/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import progress
import pycircstat
import scipy.stats as spst
import utils
import warnings
# plt.rcParams['font.size'] = 24


def _plot_kappa_mean_error(T_space, mean, yerror, ax=None, title='', **args):
        '''
            Main plotting function to show the evolution of Kappa.
        '''

        if ax is None:
            f, ax = plt.subplots()

        ax = utils.plot_mean_std_area(
            T_space, mean, np.ma.masked_invalid(yerror).filled(0.0),
            ax_handle=ax, linewidth=3, markersize=8, **args)

        # ax.legend(prop={'size': 15}, loc='best')
        if title:
            ax.set_title('Kappa: %s' % title)
        ax.set_xlim([0.9, T_space.max()+0.1])
        ax.set_ylim([0.0, max(np.max(mean)*1.1, ax.get_ylim()[1])])
        ax.set_xticks(range(1, T_space.max()+1))
        ax.set_xticklabels(range(1, T_space.max()+1))
        ax.get_figure().canvas.draw()

        return ax

def _plot_emmixture_mean_error(T_space, mean, yerror, ax=None, title='',
                                **args):
    '''
        Main plotting function to show the evolution of an EM Mixture.
    '''
    if ax is None:
        f, ax = plt.subplots()

    utils.plot_mean_std_area(
        T_space, mean, np.ma.masked_invalid(yerror).filled(0.0),
        ax_handle=ax, linewidth=3, markersize=8, **args)

    # ax.legend(prop={'size': 15}, loc='best')
    if title:
        ax.set_title('Mixture prop: %s' % title)
    ax.set_xlim([0.9, T_space.max() + 0.1])
    ax.set_ylim([0.0, 1.01])
    ax.set_xticks(range(1, T_space.max()+1))
    ax.set_xticklabels(range(1, T_space.max()+1))

    ax.get_figure().canvas.draw()

    return ax

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

        self.collapsed_em_fits = None
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

        T_space = self.fit_exp.T_space
        data_em_fits = self.fit_exp.get_data_em_fits()
        model_em_fits = self.fit_exp.get_model_em_fits(
            num_repetitions, use_cache)

        if use_sem:
            errorbars = 'sem'
        else:
            errorbars = 'std'

        f, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        # Memory fidelity last trecall
        # Data
        _plot_kappa_mean_error(
            T_space,
            data_em_fits['mean'][
                'kappa'][:, 0],
            data_em_fits[errorbars][
                'kappa'][:, 0],
            label="Data",
            fmt="o-",
            ax=axes[0])
        # Model
        _plot_kappa_mean_error(
            T_space,
            model_em_fits['mean'][
                'kappa'][:, 0],
            model_em_fits[errorbars][
                'kappa'][:, 0],
            label='Model',
            fmt="o-",
            ax=axes[0],
            xlabel='items', ylabel='Memory fidelity $[rad^{-2}]$')
        axes[0].legend(loc='upper right', bbox_to_anchor=(1., 1.))

        # Mixture proportions last trecall
        # Model
        _plot_emmixture_mean_error(
            T_space,
            model_em_fits['mean'][
                'mixt_target_tr'][:, 0],
            model_em_fits[errorbars][
                'mixt_target_tr'][:, 0],
            label='Target',
            fmt="o-",
            ax=axes[1])
        _plot_emmixture_mean_error(
            T_space,
            model_em_fits['mean'][
                'mixt_nontargets_tr'][:, 0],
            model_em_fits[errorbars][
                'mixt_nontargets_tr'][:, 0],
            label='Nontarget',
            fmt="o-",
            ax=axes[1])
        _plot_emmixture_mean_error(
            T_space,
            model_em_fits['mean'][
                'mixt_random_tr'][:, 0],
            model_em_fits[errorbars][
                'mixt_random_tr'][:, 0],
            label='Random',
            fmt="o-",
            ax=axes[1],
            xlabel='items', ylabel='Mixture proportions')
        # Data
        _plot_emmixture_mean_error(
            T_space,
            data_em_fits['mean'][
                'mixt_target_tr'][:, 0],
            data_em_fits[errorbars][
                'mixt_target_tr'][:, 0],
            label='Data target',
            fmt="s--",
            ax=axes[1])
        _plot_emmixture_mean_error(
            T_space,
            data_em_fits['mean'][
                'mixt_nontargets_tr'][:, 0],
            data_em_fits[errorbars][
                'mixt_nontargets_tr'][:, 0],
            label='Data nontarget',
            fmt="s--",
            ax=axes[1])
        _plot_emmixture_mean_error(
            T_space,
            data_em_fits['mean'][
                'mixt_random_tr'][:, 0],
            data_em_fits[errorbars][
                'mixt_random_tr'][:, 0],
            label='Data random',
            fmt="s--",
            ax=axes[1])
        axes[1].legend(loc='upper left', bbox_to_anchor=(1., 1.))

        f.suptitle('Fig 6: Last trecall')
        f.canvas.draw()

        return axes

    def plots_mixtcurves_collapsedpowerlaw_fig7(self,
                                                num_repetitions=1,
                                                use_cache=True,
                                                use_sem=True,
                                                size=6):
        '''
            Plots memory fidelity and mixture proportions for all nitems, with
            trecall on the x-axis.

            This reproduces Figure 7 in Gorgo 11.
        '''
        T_space = self.fit_exp.T_space
        data_em_fits = self.fit_exp.get_data_em_fits()
        model_em_fits = self.fit_exp.get_model_em_fits(
            num_repetitions, use_cache)

        # Do the plot
        if use_sem:
            errorbars = 'sem'
        else:
            errorbars = 'std'
        _, axes_data = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        _, axes_model = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

        # Data
        for nitems_i, nitems in enumerate(T_space):
            # Memory fidelity
            _plot_kappa_mean_error(
                T_space[:nitems],
                data_em_fits['mean'][
                    'kappa'][nitems_i, :nitems],
                data_em_fits[errorbars][
                    'kappa'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_data[0, 0])

            axes_data[0, 0].set_ylim((0, 11))

            # Mixture proportions
            _plot_emmixture_mean_error(
                T_space[:nitems],
                data_em_fits['mean'][
                    'mixt_target_tr'][nitems_i, :nitems],
                data_em_fits[errorbars][
                    'mixt_target_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_data[0, 1])
            _plot_emmixture_mean_error(
                T_space[:nitems],
                data_em_fits['mean'][
                    'mixt_nontargets_tr'][nitems_i, :nitems],
                data_em_fits[errorbars][
                    'mixt_nontargets_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_data[1, 0])
            _plot_emmixture_mean_error(
                T_space[:nitems],
                data_em_fits['mean'][
                    'mixt_random_tr'][nitems_i, :nitems],
                data_em_fits[errorbars][
                    'mixt_random_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_data[1, 1])

        # Model
        for nitems_i, nitems in enumerate(T_space):
            # Memory fidelity
            _plot_kappa_mean_error(
                T_space[:nitems],
                model_em_fits['mean'][
                    'kappa'][nitems_i, :nitems],
                model_em_fits[errorbars][
                    'kappa'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_model[0, 0])
            axes_model[0, 0].set_ylim((0, 11))

            # Mixture proportions
            _plot_emmixture_mean_error(
                T_space[:nitems],
                model_em_fits['mean'][
                    'mixt_target_tr'][nitems_i, :nitems],
                model_em_fits[errorbars][
                    'mixt_target_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_model[0, 1])
            _plot_emmixture_mean_error(
                T_space[:nitems],
                model_em_fits['mean'][
                    'mixt_nontargets_tr'][nitems_i, :nitems],
                model_em_fits[errorbars][
                    'mixt_nontargets_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_model[1, 0])
            _plot_emmixture_mean_error(
                T_space[:nitems],
                model_em_fits['mean'][
                    'mixt_random_tr'][nitems_i, :nitems],
                model_em_fits[errorbars][
                    'mixt_random_tr'][nitems_i, :nitems],
                label='%d items' % nitems,
                xlabel='Serial order (reversed)',
                fmt="o-",
                zorder=7 - nitems,
                ax=axes_model[1, 1])

        axes_data[0, 1].legend(loc='upper left', bbox_to_anchor=(1., 1.))
        axes_model[0, 1].legend(loc='upper left', bbox_to_anchor=(1., 1.))

        axes_data[0, 0].figure.canvas.draw()
        axes_model[0, 0].figure.canvas.draw()

        return axes_data, axes_model
