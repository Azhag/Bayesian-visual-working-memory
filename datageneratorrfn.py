#!/usr/bin/env python
# encoding: utf-8

"""
datageneratorrfn.py

Created by Loic Matthey on 2014-06-21.
Copyright (c) 2014 Gatsby Unit. All rights reserved.
"""


# from scaledimage import *
import matplotlib.pyplot as plt
import matplotlib.ticker as plttic
import matplotlib.patches as plt_patches
import numpy as np
from scipy.spatial.distance import pdist

from populationcode import *
from randomnetwork import *
from randomfactorialnetwork import *
import datagenerator

class DataGeneratorRFN(datagenerator.DataGenerator):
    '''
        DataGenerator for a RandomFactorialNetwork
    '''
    def __init__(self, N, T, random_network, sigma_y = 0.05, sigma_x = 0.02, time_weights=None, time_weights_parameters = dict(weighting_alpha=0.3, weighting_beta = 1.0, specific_weighting = 0.3, weight_prior='uniform'), cued_feature_time=0, enforce_min_distance=0.17, stimuli_generation='random', enforce_first_stimulus=True, stimuli_to_use=None, specific_stimuli_random_centers=False, specific_stimuli_asymmetric=False, enforce_distance_cued_feature_only=False, renormalize_sigmax=False):

        # assert isinstance(random_network, RandomFactorialNetwork), "Use a RandomFactorialNetwork with this DataGeneratorRFN"

        datagenerator.DataGenerator.__init__(self, N, T, random_network, sigma_y = sigma_y, time_weights = time_weights, time_weights_parameters = time_weights_parameters)

        # This is the noise on specific memories. Belongs here.
        self.init_sigmax(sigma_x, renormalize=renormalize_sigmax)

        self.enforce_min_distance = enforce_min_distance

        # Build the correct stimuli
        if stimuli_to_use is not None:
            # Use the provided stimuli
            self.set_stimuli(stimuli_to_use)
        else:
            if stimuli_generation == 'specific_stimuli':
                # Use our specifically built function, to get the special stimuli combination allowing to verify some biases
                self.generate_specific_stimuli(asymmetric=specific_stimuli_asymmetric, centre=np.array([0., 0.]), specific_stimuli_random_centers=specific_stimuli_random_centers)
            elif stimuli_generation is not None:
                # Generate it randomly
                self.generate_stimuli(stimuli_generation=stimuli_generation, enforce_first_stimulus=enforce_first_stimulus, cued_feature_R=1, enforce_distance_cued_feature_only=enforce_distance_cued_feature_only)
            else:
                raise ValueError("No data generation possible.")


        # Build the dataset
        self.build_dataset(cued_feature_time=cued_feature_time)

    def init_sigmax(self, sigma_x_input=0.1, renormalize=False):
        '''
            Will initialise sigma_x properly.
            If desired, we can max it so that sigma_x_input is interpreted as a proportion of the maximal network activation (obviously values close to 1 will be crazy).
            This allows for a more useful setting of sigmax, and should work for R>2 (as the max activation depends on R, most likely as 10^-R)
        '''
        max_network_activation = self.random_network.compute_maximum_activation_network()

        if renormalize:
            # max_network_activation = self.random_network.compute_maximum_activation_network()
            self.sigma_x = max_network_activation*sigma_x_input
        else:
            self.sigma_x = sigma_x_input

        print "sigmax/max network: %.5f" % (self.sigma_x/max_network_activation)



    def generate_stimuli(self, stimuli_generation='random', enforce_first_stimulus=True, cued_feature_R=1, enforce_distance_cued_feature_only=False):
        '''
            Choose N stimuli for this dataset.

            init:
                self.stimuli_correct:   N x T x R
        '''

        random_generation = False

        if is_function(stimuli_generation):
            angle_generator = stimuli_generation
        else:
            if stimuli_generation == 'random':
                angle_generator = lambda T: (np.random.rand(T)-0.5)*2.*np.pi
                random_generation = True
            elif stimuli_generation == 'constant':
                angle_generator = lambda T: 1.2*np.ones(T)
            elif stimuli_generation == 'random_smallrange':
                angle_generator = lambda: (np.random.rand(T)-0.5)*np.pi/2.
                # angle_generator = lambda T: (np.random.rand(T)-0.5)*np.pi
                random_generation = True
            elif stimuli_generation == 'constant_separated':
                angle_generator = lambda T: 1.2*np.ones(T)
            elif stimuli_generation == 'separated':
                angle_generator = lambda T: np.linspace(-np.pi*0.6, np.pi*0.6, T)
            else:
                raise ValueError('Unknown stimulus generation technique')

        # This gives all the true stimuli
        self.stimuli_correct = np.zeros((self.N, self.T, self.R), dtype=float)

        ## Get stimuli with a minimum enforced distance.
        # Sample stimuli uniformly
        # Enforce differences on all dimensions (P. Bays: 10 deg for orientations).
        for n in xrange(self.N):
            for r in xrange(self.R):
                self.stimuli_correct[n, :, r] = angle_generator(self.T)

                # Enforce minimal distance between different times
                if random_generation and self.enforce_min_distance > 0. and (not enforce_distance_cued_feature_only or cued_feature_R == r):
                    tries = 0
                    while np.any(pdist(self.stimuli_correct[n, :, r][:, np.newaxis], 'chebyshev') < self.enforce_min_distance) and tries < 1000:
                        # Some are too close, resample
                        self.stimuli_correct[n, :, r] = angle_generator(self.T)
                        # self.stimuli_correct[n, :, r] = (np.random.rand(self.T)-0.5)*np.pi
                        tries += 1


    def generate_specific_stimuli(self, asymmetric=False, centre=np.array([0., 0.]), specific_stimuli_random_centers=True, randomise_target=True):
        '''
            Construct a specific set of stimuli tailored to discriminate between population code types.

            Will generate different error patterns depending on the population codes used.
        '''

        assert self.R == 2, "works for R=2 only"

        if self.T == 3:
            # Three points on a diagonal. Should produce different biases for conjunctive or feature.
            dx = self.enforce_min_distance/np.sqrt(2)

            if specific_stimuli_random_centers:
                centre_disturb_space = centre + (2.*np.random.rand(self.N, 2) - 1.0)*dx
            else:
                centre_disturb_space = np.ones((self.N, 2))*centre

            if not asymmetric:
                # Mean of ensemble lies on the center point though, which may complicate analysis.
                self.stimuli_correct = np.array([[centre_disturb + np.array([-dx, dx]), centre_disturb + np.array([dx, -dx]), centre_disturb] for centre_disturb in centre_disturb_space])
            else:
                # Asymmetric. Mean of ensemble lies at the left of the two on the right.
                self.stimuli_correct = np.array([[centre_disturb + np.array([-2.*dx, 2.*dx]), centre_disturb + np.array([dx, -dx]), centre_disturb + np.array([2.*dx, -2.*dx])] for centre_disturb in centre_disturb_space])

            # Shuffle targets randomly if desired (always cue T=3, last item)
            if randomise_target:
                map(np.random.shuffle, self.stimuli_correct)

        else:
            raise NotImplementedError("Specific stimuli only works for T=3 for now")




    def set_stimuli(self, stimuli):
        '''
            Set stimuli directly.

            Can come from experimental data for example.
            Should call build_dataset() then
        '''

        assert stimuli.shape == (self.N, self.T, self.R), "Stimuli shapes do not correspond %s != %s" % (stimuli.shape, (self.N, self.T, self.R))

        # Set it, as a copy to be safe
        self.stimuli_correct = stimuli[:]



    def build_dataset(self, cued_feature_time=0):
        '''
            Creates the dataset
                For each datapoint, use the already sampled stimuli_correct, get the network response,
                and then combine them together, with time decay

            input:
                [cued_feature_time: The time of the cue. (Should be random)]


            output:
                Y :                 N x M
                all_Y:              N x T x M
                stimuli_correct:    N x T x R
                cued_features:      N x 2       (feature_cued, time_cued)
        '''


        # Select which item should be recalled (and thus cue one/multiple of the other feature)
        self.cued_features = np.zeros((self.N, 2), dtype='int')

        # Initialise Y (keep intermediate y_t as well)
        self.all_Y = np.zeros((self.N, self.T, self.random_network.M))
        self.Y = np.zeros((self.N, self.random_network.M))
        self.all_X = np.zeros((self.N, self.T, self.random_network.M))


        # TODO Hack for now, add the time contribution
        # self.time_contribution = 0.06*np.random.randn(self.T, self.random_network.M)

        for i in xrange(self.N):

            # For now, always cued the second feature (i.e. color) and retrieve the first feature (i.e. orientation)
            self.cued_features[i, 0] = 1

            # Randomly recall one of the times
            # self.cued_features[i, 1] = np.random.randint(self.T)
            self.cued_features[i, 1] = cued_feature_time

            # Create the memory
            for t in xrange(self.T):
                # Get the 'x' sample (here from the population code)
                x_sample = self.random_network.sample_network_response(self.stimuli_correct[i, t], sigma=self.sigma_x)

                self.Y[i] = self.time_weights[0, t]*self.Y[i].copy() + self.time_weights[1, t]*x_sample + self.sigma_y*np.random.randn(self.random_network.M)
                # self.Y[i] /= np.sum(np.abs(self.Y[i]))
                # self.Y[i] /= fast_1d_norm(self.Y[i])
                self.all_Y[i, t] = self.Y[i]
                self.all_X[i, t] = x_sample

        # For convenience, store the list of nontargets objects.
        self.nontargets_indices = np.array([[t for t in xrange(self.T) if t != self.cued_features[n, 1]] for n in xrange(self.N)], dtype='int')



    def show_datapoint(self, n=0, colormap=None):
        '''
            Show a datapoint.

            Call other routines, depending on their network population_code_type.
        '''

        display_type = self.random_network.population_code_type

        if display_type == 'conjunctive':
            self.show_datapoint_conjunctive(n=n, colormap=colormap)
        elif display_type == 'feature':
            self.show_datapoint_features(n=n, colormap=colormap)
        elif display_type == 'wavelet':
            self.show_datapoint_wavelet(n=n, colormap=colormap)
        elif display_type == 'mixed':
            self.show_datapoint_mixed(n=n, colormap=colormap)
        elif display_type == 'hierarchical':
            self.show_datapoint_hierarchical(n=n, colormap=colormap)
        else:
            raise ValueError("Unknown population type:" + self.random_network.population_code_type)


    def show_datapoint_conjunctive(self, n=0, colormap=None):
        '''
            Show a datapoint, as a 2D grid plot (for now, assumes R==2).

            Works for conjunctive only, as we do not have a nice spatial mapping for other codes.
        '''

        # TODO

        M = self.random_network.M
        M_sqrt = np.floor(M**0.5)

        print M_sqrt
        print M

        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(np.reshape(self.Y[n][:M_sqrt*M_sqrt], (M_sqrt, M_sqrt)).T, origin='lower', aspect='equal', interpolation='nearest', cmap=colormap)
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        # Show ellipses at the stimuli positions
        colmap = plt.get_cmap('gist_rainbow')
        color_gen = [colmap(1.*(i)/self.T) for i in xrange(self.T)][::-1]  # use 22 colors
        # color_gen = ["blue", "green", "red", "cyan", "magenta", "yellow"]

        for t in xrange(self.T):
            w = plt_patches.Wedge((self.stimuli_correct[n, t, 0], self.stimuli_correct[n, t, 1]), 0.25, 0, 360, 0.12, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)


    def show_datapoint_features(self, n=0, colormap=None):
        '''
            Show a datapoint for a features code.
            Will show 2 "lines", with the output of horizontal/vertical neurons.
        '''

        # TODO

        M = self.random_network.M

        # When there are multiple centers per feature, we can reduce the space.

        horiz_cells = np.arange(M/2/self.random_network.nb_feature_centers)
        vert_cells = np.arange(M/2, (M/2+M/2/self.random_network.nb_feature_centers))


        factor_2lines = 1.9

        f = plt.figure()
        ax = f.add_subplot(111)

        ax.plot(np.linspace(-np.pi, np.pi, horiz_cells.size), self.Y[n, horiz_cells], linewidth=2)
        ax.plot(np.linspace(-np.pi, np.pi, vert_cells.size), self.Y[n, vert_cells] + factor_2lines*self.Y[n, horiz_cells].max(), linewidth=2)

        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_yticks(())
        ax.legend(['Horizontal cells', 'Vertical cells'], fancybox=True, borderpad=0.3, columnspacing=0.5, borderaxespad=0.1, handletextpad=0, handlelength=1.5, bbox_to_anchor=(1.0, 1.0))

        # Show ellipses at the stimuli positions
        colmap = plt.get_cmap('gist_rainbow')
        color_gen = [colmap(1.*(i)/self.T) for i in xrange(self.T)][::-1]  # use 22 colors

        for t in xrange(self.T):

            # max_pos = np.argmin((np.linspace(-np.pi, np.pi, horiz_cells.size, endpoint=False) - self.stimuli_correct[n, t, 0])**2.)
            w = plt_patches.Wedge((self.stimuli_correct[n, t, 0], 1.2*self.Y[n, horiz_cells].max()), 0.1, 0, 360, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)

            w = plt_patches.Wedge((self.stimuli_correct[n, t, 1], 1.1*self.Y[n, horiz_cells].max() + factor_2lines*self.Y[n, horiz_cells].max()), 0.1, 0, 360, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)

        plt.xlim((-np.pi, np.pi))
        plt.ylim((-0.5, 1.2*(1.2*self.Y[n, horiz_cells].max() + factor_2lines*self.Y[n, horiz_cells].max())))

        # im = ax.imshow(np.reshape(self.Y[n], (2, M/2)).T, origin='lower', aspect='equal', interpolation='nearest')
        # im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        # ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        # ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))


    def show_datapoint_mixed(self, n=0, colormap=None):
        '''
            Show a datapoint for a mixed code
        '''

        # TODO

        f = plt.figure()

        ##### Show the conjunctive units first
        ax = f.add_subplot(211)
        conj_sqrt = int(self.random_network.conj_subpop_size**0.5)
        # TODO Fix for conj_subpop_size = 0
        im = ax.imshow(np.reshape(self.Y[n][:conj_sqrt**2.], (conj_sqrt, conj_sqrt)).T, origin='lower', aspect='equal', interpolation='nearest', cmap=colormap)
        im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
        ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        # Show ellipses at the stimuli positions
        colmap = plt.get_cmap('gist_rainbow')
        color_gen = [colmap(1.*(i)/self.T) for i in xrange(self.T)][::-1]  # use 22 colors

        for t in xrange(self.T):
            w = plt_patches.Wedge((self.stimuli_correct[n, t, 0], self.stimuli_correct[n, t, 1]), 0.25, 0, 360, 0.12, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)

        ##### Show the feature units
        ax = f.add_subplot(212)

        horiz_cells = np.arange(self.random_network.conj_subpop_size, int(self.random_network.feat_subpop_size/2.+self.random_network.conj_subpop_size))
        vert_cells = np.arange(int(self.random_network.feat_subpop_size/2.+self.random_network.conj_subpop_size), self.random_network.M)

        factor_2lines = 1.9

        ax.plot(np.linspace(-np.pi, np.pi, horiz_cells.size), self.Y[n, horiz_cells], linewidth=2)
        ax.plot(np.linspace(-np.pi, np.pi, vert_cells.size), self.Y[n, vert_cells] + factor_2lines*self.Y[n, horiz_cells].max(), linewidth=2)

        ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

        ax.set_yticks(())
        ax.legend(['Horizontal cells', 'Vertical cells'], fancybox=True, borderpad=0.3, columnspacing=0.5, borderaxespad=0.7, handletextpad=0, handlelength=1.5)

        # Show ellipses at the stimuli positions
        colmap = plt.get_cmap('gist_rainbow')
        color_gen = [colmap(1.*(i)/self.T) for i in xrange(self.T)][::-1]  # use 22 colors

        for t in xrange(self.T):

            # max_pos = np.argmin((np.linspace(-np.pi, np.pi, horiz_cells.size, endpoint=False) - self.stimuli_correct[n, t, 0])**2.)
            w = plt_patches.Wedge((self.stimuli_correct[n, t, 0], 1.2*self.Y[n, horiz_cells].max()), 0.1, 0, 360, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)

            w = plt_patches.Wedge((self.stimuli_correct[n, t, 1], 1.1*self.Y[n, horiz_cells].max() + factor_2lines*self.Y[n, horiz_cells].max()), 0.1, 0, 360, color=color_gen[t], alpha=1.0, linewidth=2)
            ax.add_patch(w)

        ax.set_xlim((-np.pi, np.pi))
        ax.set_ylim((-0.5, 1.5*(1.2*self.Y[n, horiz_cells].max() + factor_2lines*self.Y[n, horiz_cells].max())))



    def show_datapoint_wavelet(self, n=0, single_figure=True, colormap=None):
        '''
            Show a datapoint for wavelet code.
            Will print several 2D grid plots (for now, assumes R==2).

            Do several or an unique figure, depending on single_figure
        '''

        # TODO

        # Get all existing scales
        all_scales = np.unique(self.random_network.neurons_scales)
        all_scales = all_scales[all_scales > 0]

        if single_figure:
            nb_subplots_sqrt = int(np.floor(all_scales.size**0.5))
            f, axes = plt.subplots(nb_subplots_sqrt, nb_subplots_sqrt)
            axes = axes.flatten()

        for curr_scale_i in xrange(all_scales.size):
            # Use this as grid side
            scale_sqrt = np.floor(np.sum(self.random_network.neurons_scales == all_scales[curr_scale_i])**0.5)

            # Find the desired neurons
            filter_neurons = np.nonzero(self.random_network.neurons_scales == all_scales[curr_scale_i])[0]

            # Now do a "normal" conjunctive grid plot
            if single_figure:
                ax = axes[curr_scale_i]
            else:
                f = plt.figure()
                ax = f.add_subplot(111)

            im = ax.imshow(np.reshape(self.Y[n, filter_neurons], (scale_sqrt, scale_sqrt)).T, origin='lower', aspect='equal', interpolation='nearest')
            im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
            ax.set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
            ax.set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))


        plt.show()


    def show_datapoint_flat(self, n=0, t=-1):
        '''
            Show a datapoint in 1d, flatten out.

            For conjunctive, indicate the "correct" stimulus position.
        '''

        # Plot the datapoint
        plt.figure()
        plt.plot(self.Y[n])

        # Put a vertical line at the true answer
        try:
            best_neuron_i = np.argmin(np.sum((self.random_network.neurons_preferred_stimulus - self.stimuli_correct[n, t])**2., axis=1))
            plt.axvline(x=best_neuron_i, color='r', linewidth=3)
        except AttributeError:
            # Most likely a hierarchical network
            pass


    def show_datapoint_hierarchical(self, n=0, colormap=None):
        '''
            Show a datapoint from a hiearchical random network

            Shows the activity of the first level on the different objects, and on top the activity of the second random level
        '''

        # TODO

        # Collect first level activities for the current datapoint
        level_one_activities = np.empty((self.T, self.random_network.M_layer_one))
        level_two_activities = np.empty((self.T, self.random_network.M))
        for t in xrange(self.T):
            level_two_activities[t] = self.random_network.get_network_response(self.stimuli_correct[n, t])
            level_one_activities[t] = self.random_network.current_layer_one_response

        # Now construct the plot. Use gridspec
        plt.figure()
        ax_leveltwo_global = plt.subplot2grid((3, self.T), (0, 0), colspan=self.T)
        axes_levelone = []
        axes_leveltwo = []
        for t in xrange(self.T):
            axes_leveltwo.append(plt.subplot2grid((3, self.T), (1, t)))
            axes_levelone.append(plt.subplot2grid((3, self.T), (2, t)))

        # Plot the level two activation, use a bar, easier to read
        ax_leveltwo_global.bar(np.arange(self.random_network.M), self.Y[n])

        # Plot the activation of the level one subnetwork (and of the individual responses at level two)
        M_sqrt = int(self.random_network.M_layer_one**0.5)
        for t in xrange(self.T):
            # Level two, on individual items
            axes_leveltwo[t].bar(np.arange(self.random_network.M), level_two_activities[t])

            # Level one
            im = axes_levelone[t].imshow(level_one_activities[t].reshape(M_sqrt, M_sqrt).T, origin='lower', aspect='equal', interpolation='nearest')
            im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
            axes_levelone[t].set_xticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            axes_levelone[t].set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))
            axes_levelone[t].set_yticks((-np.pi, -np.pi/2, 0, np.pi/2., np.pi))
            axes_levelone[t].set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'))

            e = Ellipse(xy=self.stimuli_correct[n, t], width=0.4, height=0.4)

            axes_levelone[t].add_artist(e)
            e.set_clip_box(axes_levelone[t].bbox)
            e.set_alpha(0.5)
            e.set_facecolor('white')
            e.set_transform(axes_levelone[t].transData)

        plt.show()


