#!/usr/bin/env python
# encoding: utf-8
"""
highdimensionnetwork.py

Created by Loic Matthey on 2011-11-09.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
from matplotlib.patches import Ellipse
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import scipy.special as scsp
import scipy.stats as spst


# from randomfactorialnetwork import *

import progress

import utils


class HighDimensionNetwork(object):
    """
        Modified paradigm for this Network. Uses a factorial representation
        of K features, and samples them using K-dimensional gaussian
        receptive fields.
        Randomness is in the distribution of orientations and radii of those
        gaussians.
    """
    def __init__(self, M, R=2, response_maxout=False):

        self.M = M
        self.R = R

        self.network_initialised = False
        self.population_code_type = None

        self.neurons_preferred_stimulus = None
        self.neurons_sigma = None
        self.rc_scale = None
        self.mask_neurons_unset = None
        self.normalisation = None

        self._ALL_NEURONS = np.arange(M)

        self.get_network_response_opt = None

        if response_maxout:
            print ' -- new maxout response'
            self.get_network_response_bivariatefisher_callback = self.get_network_response_bivariatefisher_maxoutput
        else:
            self.get_network_response_bivariatefisher_callback = self.get_network_response_bivariatefisher
        self.default_stimulus_input = np.array((0.0,) * self.R)

        # Need to assign to each of the M neurons a preferred stimulus
        # (tuple(orientation, color) for example)
        # By default, random
        self.assign_prefered_stimuli(tiling_type='conjunctive')
        self.assign_aligned_eigenvectors()

        # Used to stored cached network response statistics.
        # Mean_theta(mu(theta)) and Cov_theta(mu(theta))
        self.network_response_statistics = None

        self.network_initialised = True


    def assign_prefered_stimuli(self, tiling_type='conjunctive', specific_neurons=None, reset=False):
        '''
            For all M factorial neurons, assign them a prefered stimulus
            tuple (e.g. (orientation, color) )
        '''

        if self.neurons_preferred_stimulus is None or reset:
            # Handle uninitialized neurons
            self.neurons_preferred_stimulus = np.nan*np.ones((self.M, self.R))

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        if specific_neurons.size != 0:
            # Only do something if non-empty

            if tiling_type == 'conjunctive':
                self.assign_prefered_stimuli_conjunctive(specific_neurons)
            elif tiling_type == 'features':
                self.assign_prefered_stimuli_features(specific_neurons)
            elif tiling_type == 'random':
                self.assign_prefered_stimuli_random(specific_neurons)

            # Handle uninitialized neurons
            #   check if still some nan, any on the first axis.
            self.mask_neurons_unset = np.any(np.isnan(self.neurons_preferred_stimulus), 1)



    def assign_prefered_stimuli_conjunctive(self, neurons_indices):
        '''
            Tile conjunctive neurons along the space of possible angles.
        '''
        # Cover the space uniformly
        N_sqrt = np.floor(np.power(neurons_indices.size, 1./self.R))

        # coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt, endpoint=False)
        coverage_1D = np.linspace(-np.pi + np.pi/N_sqrt, np.pi + np.pi/N_sqrt, N_sqrt, endpoint=False)

        new_stim = np.array(utils.cross(self.R*[coverage_1D.tolist()]))

        # Assign the preferred stimuli
        #   Unintialized neurons will get masked out down there.
        self.neurons_preferred_stimulus[neurons_indices[:new_stim.shape[0]]] = new_stim


    def assign_prefered_stimuli_features(self, neurons_indices):
        '''
            Tile feature neurons in R-D
        '''
        N = np.round(neurons_indices.size/self.R)

        center = 0.0

        # coverage_1D = np.linspace(-np.pi, np.pi, sub_N, endpoint=False)
        coverage_1D = np.linspace(-np.pi + np.pi/N, np.pi + np.pi/N, N, endpoint=False)

        for r in xrange(self.R):
            self.neurons_preferred_stimulus[neurons_indices[N*r:N*(r+1)], :] = center
            self.neurons_preferred_stimulus[neurons_indices[N*r:N*(r+1)], r] = coverage_1D

    def assign_prefered_stimuli_random(self, neurons_indices):
        '''
            Randomly assign preferred stimuli to all neurons
        '''

        new_stim = utils.sample_angle(size=(neurons_indices.size, self.R))

        # Assign the preferred stimuli
        #   Unintialized neurons will get masked out down there.
        self.neurons_preferred_stimulus[neurons_indices[:new_stim.shape[0]]] = new_stim


    def assign_aligned_eigenvectors(self, scale=1.0, ratio=1.0, scaled_dimension=1, specific_neurons=None, reset=False):
        '''
            Each neuron has a gaussian receptive field, defined by its eigenvectors/eigenvalues (principal axes and scale)
            Uses the same eigenvalues for all of those

            scaled_dimension controls which dimension should get ratio*scale as rc_scale. The others get scale.

            input:
                ratio:  1/-1   : circular gaussian.
                        >1      : ellipse, width>height
                        <-1     : ellipse, width<height
                        [-1, 1] : not used
        '''

        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS

        assert ratio <= -1 or ratio >= 1, "respect my authority! Use ratio >= 1 or <=-1: %.2f" % ratio

        self.neurons_sigma[specific_neurons] = scale
        self.neurons_sigma[specific_neurons, scaled_dimension] = ratio * scale

        self.neurons_angle[specific_neurons] = 0.0

        # Update parameters
        self.precompute_parameters(specific_neurons=specific_neurons)


    def precompute_parameters(self, specific_neurons=None):
        '''
            Function called to precompute different parameters to speed-up some computations
        '''
        # Precompute the normalisation constant
        self.compute_normalising_constant_bivariatefisher(specific_neurons=specific_neurons)

        # Save the rc_scale used
        self.rc_scale = np.mean(self.neurons_sigma, axis=0)

        # Assign response function, depending on neurons_sigma
        if np.any(self.neurons_sigma > 700):
            print ">> RandomNetwork has large Kappa, using safe slow function"
            self.get_network_response_opt = self.get_network_response_large_kappa_safe


    def compute_normalising_constant_bivariatefisher(self, specific_neurons=None):
        '''
            Depending on neuron_sigma, we have different normalising constants per neurons.

            The full formula, for kappa3 != 0 is more complex, we do not use it for now:

            Z = 4 pi^2 \sum_{m=0}^\infty n_choose_k(2m, m) (\kappa_3^2/(4 kappa_1 kappa_2))^m I_m(kappa_1) I_m(kappa_2)

            Here, for \kappa_3=0, only m=0 used
        '''

        if self.normalisation is None:
            self.normalisation = np.zeros(self.M)
            self.normalisation_fisher_all = np.zeros((self.M, self.R))
            self.normalisation_gauss_all = np.zeros((self.M, self.R))

        # The normalising constant
        #   Overflows have happened, but they have no real consequence, as 1/inf = 0.0, appropriately.
        if specific_neurons is None:

            # precompute separate ones
            self.normalisation_fisher_all = 2.*np.pi*scsp.i0(self.neurons_sigma)
            self.normalisation_gauss_all = np.sqrt(self.neurons_sigma)/(np.sqrt(2*np.pi))

            self.normalisation = np.prod(self.normalisation_fisher_all, axis=-1)
        else:
            self.normalisation_fisher_all[specific_neurons] = 2.*np.pi*scsp.i0(self.neurons_sigma[specific_neurons])
            self.normalisation_gauss_all[specific_neurons] = np.sqrt(self.neurons_sigma[specific_neurons])/(np.sqrt(2*np.pi))
            self.normalisation[specific_neurons] = np.prod(self.normalisation_fisher_all[specific_neurons], axis=-1)


    def compute_maximum_activation_network(self, nb_samples=50):
        '''
            Try to estimate the maximum activation for the network.

            This can be used to make sure sigmax is adapted, or to renormalize everything.
        '''

        test_samples = utils.sample_angle((nb_samples, self.R))

        max_activation = 0
        for test_sample in test_samples:
            max_activation = max(np.nanmax(self.get_network_response(test_sample)), max_activation)

        return max_activation



    #########################################################################################################


    def get_network_response(self, stimulus_input=None, specific_neurons=None):
        '''
            Function hook for the current way to get the network response.
        '''

        if stimulus_input is None:
            stimulus_input = self.default_stimulus_input

        return self.get_network_response_bivariatefisher_callback(stimulus_input, specific_neurons=specific_neurons)


    def get_network_response_bivariatefisher(self, stimulus_input, specific_neurons=None):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        if self.get_network_response_opt is not None and specific_neurons is None:
            output = self.get_network_response_opt(stimulus_input)
        else:
            if specific_neurons is None:
                specific_neurons = slice(None)

            dmu = stimulus_input - self.neurons_preferred_stimulus[specific_neurons]
            output = np.exp(np.sum(self.neurons_sigma[specific_neurons]*np.cos(dmu), axis=-1))/self.normalisation[specific_neurons]

            output[self.mask_neurons_unset[specific_neurons]] = 0.0

        return output

    def get_network_response_bivariatefisher_maxoutput(self, stimulus_input, specific_neurons=None):
        '''
            Compute the response of the network.

            Use a Von Mises-Fisher general distribution.

            Now computes intermediate vectors, could change everything to use vectors only.
        '''

        if specific_neurons is None:
            specific_neurons = slice(None)

        dmu = stimulus_input - self.neurons_preferred_stimulus[specific_neurons]
        output = np.exp(-np.sum(self.neurons_sigma[specific_neurons]*(1. - np.cos(dmu)), axis=-1))

        output[self.mask_neurons_unset[specific_neurons]] = 0.0

        return output


    def get_network_response_large_kappa_safe(self, stimulus_input):
        '''
            Implement a safe version of the population code, which uses a Normal approximation to the Von Mises for kappa>700
        '''

        output = np.ones(self.M)

        for r in xrange(self.R):
            index_fish = self.neurons_sigma[:, r] <= 700
            index_gauss = self.neurons_sigma[:, r] > 700

            output[index_fish] *= np.exp(self.neurons_sigma[index_fish, r]*np.cos((stimulus_input[r] - self.neurons_preferred_stimulus[index_fish, r])))/self.normalisation_fisher_all[index_fish, r]

            output[index_gauss] *= np.exp(-0.5*self.neurons_sigma[index_gauss, r]*(stimulus_input[r] - self.neurons_preferred_stimulus[index_gauss, r])**2.)*self.normalisation_gauss_all[index_gauss, r]

        output[self.mask_neurons_unset] = 0.0

        return output


    def get_derivative_network_response(self, derivative_feature_target=0, stimulus_input=None):
        '''
            Compute and return the derivative of the network response.
        '''

        if stimulus_input is None:
            stimulus_input = self.default_stimulus_input


        dmu_specific_feature = stimulus_input[derivative_feature_target] - self.neurons_preferred_stimulus[:, derivative_feature_target]

        der_f = self.neurons_sigma[:, derivative_feature_target]*np.sin(dmu_specific_feature)*self.get_network_response(stimulus_input)

        der_f[self.mask_neurons_unset] = 0.0

        return der_f


    ####

    def compute_network_response_statistics(self, num_samples=5000, ignore_cache=False):
        '''
            Will compute the mean and covariance of the network output.
            These are used in some analytical expressions.

            They are currently estimated from samples, there might be a closed-form solution...
        '''

        if ignore_cache or self.network_response_statistics is None:
            # Should compute it

            # Sample responses to measure the statistics on
            responses = self.collect_network_responses(num_samples=num_samples)

            # Compute the mean and covariance
            computed_mean = np.mean(responses, axis=0)
            computed_cov = np.cov(responses.T)

            # Cache them
            self.network_response_statistics = {'mean': computed_mean, 'cov': computed_cov}

        # Return the cached values
        return self.network_response_statistics


    def collect_network_responses(self, num_samples=5000):
        '''
            Sample network responses (population code outputs) over the entire space, to be used for empirical estimates
        '''

        responses = np.empty((num_samples, self.M))

        random_angles = utils.sample_angle((num_samples, self.R))
        for i in progress.ProgressDisplay(xrange(num_samples), display=progress.SINGLE_LINE):
            responses[i] = self.get_network_response(random_angles[i])

        return responses

    ########################################################################################################################


    def get_neuron_response(self, neuron_index, stimulus_input):
        '''
            Get the output of one specific neuron, for a specific stimulus

            returns: Mx1 vector
        '''

        return self.get_network_response(stimulus_input, specific_neurons=np.array([neuron_index]))


    def sample_network_response(self, stimulus_input, sigma=0.2):
        '''
            Get a random response for the given stimulus.

            return: Mx1
        '''

        return self.get_network_response(stimulus_input) + sigma*np.random.randn(self.M)


    def sample_multiple_network_response(self, stimuli_input, sigma=0.2):
        '''
            Get a set of random responses for multiple stimuli

            return: N x M
        '''

        nb_samples = stimuli_input.shape[0]
        net_samples = np.zeros((nb_samples, self.M))

        for i in xrange(nb_samples):
            net_samples[i] = self.sample_network_response(stimuli_input[i], sigma=sigma)

        return net_samples

    # ===================================================================

    def get_neuron_activity(self, neuron_index, precision=100, axes=None):
        '''
            Returns the activity of a specific neuron over the entire space.
        '''

        coverage_1D = self.init_feature_space(precision)

        possible_stimuli = np.array(utils.cross(self.R*[coverage_1D.tolist()]))
        activity = np.empty(possible_stimuli.shape[0])

        # Compute the activity of that neuron over the whole space
        for stimulus_i, stimulus in enumerate(possible_stimuli):
            activity[stimulus_i] = self.get_neuron_response(neuron_index, stimulus)

        # Reshape
        activity.shape = self.R*(precision, )

        if axes is not None:
            for dim_to_avg in set(range(len(activity.shape))) - set(axes):
                activity = np.mean(activity, axis=dim_to_avg, keepdims=True)
            activity = np.squeeze(activity)

        return activity


    def get_mean_activity(self, axes=(0, 1), precision=100, specific_neurons=None, return_axes_vect=False):
        '''
            Returns the mean activity of the network.
        '''

        coverage_1D = self.init_feature_space(precision)

        possible_stimuli = np.array(utils.cross(self.R*[coverage_1D.tolist()]))
        activity = np.empty((possible_stimuli.shape[0], self.M))

        for stimulus_i, stimulus in enumerate(possible_stimuli):
            activity[stimulus_i] = self.get_network_response(stimulus, specific_neurons=specific_neurons)

        # Reshape
        activity.shape = self.R*(precision, ) + (self.M, )

        mean_activity = activity

        for dim_to_avg in (set(range(len(activity.shape))) - set(axes)):
            mean_activity = np.mean(mean_activity, axis=dim_to_avg, keepdims=True)

        mean_activity = np.squeeze(mean_activity)

        if return_axes_vect:
            return (mean_activity, coverage_1D, coverage_1D)
        else:
            return mean_activity


    def compute_covariance_KL(self, num_samples=5000, sigma_x=0.1, sigma_y=0.0001, sigma_baseline=0.00001, beta=1.0, T=1, should_plot=False, ignore_cache=False):
        '''
            Compute the covariance of the Gaussian approximation (through a KL) to the averaged object.

            Sigma* = T (sigma_y^2 + beta^2 sigma_x^2) I + (T-1) beta^2 Cov( mu(theta))_p(theta) + sigma_baseline^2 I
        '''

        # Get the statistics of the network population code
        network_response_statistics = self.compute_network_response_statistics(num_samples=num_samples, ignore_cache=ignore_cache)

        # !!! The actual Fisher Information computation !!!
        covariance = (T-1)*beta**2.*network_response_statistics['cov'] + T*(sigma_x**2. + sigma_y**2.)*np.eye(self.M) + sigma_baseline**2.*np.eye(self.M)

        if should_plot:
            plt.figure()
            plt.imshow(covariance, interpolation='nearest')
            plt.show()

        # Output it
        return covariance


    def compute_fisher_information(self, stimulus_input, cov_stim=None, inv_cov_stim=None):

        der_f = self.get_derivative_network_response(stimulus_input=stimulus_input)

        if cov_stim is not None:
            return np.dot(der_f, np.linalg.solve(cov_stim, der_f))
        elif inv_cov_stim is not None:
            return np.dot(der_f, np.dot(inv_cov_stim, der_f))
        else:
            return 0.


    def compute_sample_inverse_FI(self, nitems, inv_cov_stim, min_distance=0.17):
        '''
            Compute one sample estimate of the Inverse Fisher Information, nitems.

            Those samples can then be averaged together in a Monte Carlo scheme to provide a better estimate of the Inverse Fisher Information for K items.

            Assume the first item is fixed at (0, 0), sample the other uniformly
        '''

        all_items = [np.array([0.0, 0.0])]

        # Add extra items
        for item_i in xrange(nitems - 1):
            new_item = utils.sample_angle(size=2)
            while not utils.enforce_distance_set(new_item,
                                                 all_items,
                                                 min_distance):
                new_item = utils.sample_angle(size=2)

            all_items.append(new_item)

        items_thetas = np.array(all_items)

        # Compute all derivatives
        deriv_mu = np.zeros((2*nitems, self.M))
        for i in xrange(nitems):
            deriv_mu[2*i] = self.get_derivative_network_response(derivative_feature_target=0, stimulus_input=items_thetas[i])
            deriv_mu[2*i + 1] = self.get_derivative_network_response(derivative_feature_target=1, stimulus_input=items_thetas[i])
        deriv_mu[np.isnan(deriv_mu)] = 0.0

        # Compute the Fisher information matrix
        FI_nobj = np.dot(deriv_mu, np.dot(inv_cov_stim, deriv_mu.T))

        try:
            inv_FI_nobj = np.linalg.inv(FI_nobj)
        except np.linalg.linalg.LinAlgError:
            inv_FI_nobj = np.nan*np.empty(FI_nobj.shape)

        return inv_FI_nobj[0, 0], FI_nobj[0, 0]


    def compute_marginal_inverse_FI(self,
                                    nitems,
                                    inv_cov_stim,
                                    max_n_samples=int(1e5),
                                    min_distance=0.1,
                                    convergence_epsilon=1e-7,
                                    debug=True):
        '''
            Compute a Monte Carlo estimate of the Marginal Inverse Fisher Information
            Averages over stimuli values. Requires the inverse of the covariance of the memory.

            Returns dict(inv_FI, inv_FI_std, FI, FI_std)
        '''

        min_num_samples_std = int(2e3)

        marginal_fi_dict = dict(inv_FI=0,
                                inv_FI_std=0,
                                FI=0,
                                FI_std=0
                                )
        inv_FI_cum = 0
        FI_cum = 0
        FI_cum_var = 0
        inv_FI_cum_var = 0
        epsilon = 0

        new_estimates = np.zeros(4)
        previous_estimates = np.zeros(4)

        for i in xrange(max_n_samples):
            # Get sample of invFI and FI
            inv_FI_sample, FI_sample = self.compute_sample_inverse_FI(
                nitems, inv_cov_stim, min_distance
            )

            # Compute mean
            inv_FI_cum += inv_FI_sample/float(max_n_samples)
            FI_cum += FI_sample/float(max_n_samples)

            # Compute std (wrong but oh well...)
            if i > min_num_samples_std:
                inv_FI_cum_var += (inv_FI_sample -
                    inv_FI_cum*max_n_samples/(i+1.))**2./float(max_n_samples)
                FI_cum_var += (FI_sample -
                    FI_cum*max_n_samples/(i+1.))**2./float(max_n_samples)

            # Compute current running averages
            marginal_fi_dict['inv_FI'] = inv_FI_cum*max_n_samples/(i+1.)
            marginal_fi_dict['inv_FI_std'] = np.sqrt(
                inv_FI_cum_var*max_n_samples/(i+1.))/np.sqrt(i+1)
            marginal_fi_dict['FI'] = FI_cum*max_n_samples/(i+1.)
            marginal_fi_dict['FI_std'] = np.sqrt(
                FI_cum_var*max_n_samples/(i+1.))/np.sqrt(i+1)

            if i > 1.5*min_num_samples_std:
                # Check convergence
                for k_i, k in enumerate(marginal_fi_dict):
                    new_estimates[k_i] = marginal_fi_dict[k]

                epsilon = np.nansum(np.abs(previous_estimates - new_estimates))

                if epsilon <= convergence_epsilon:
                    # Converged, stop the loop
                    if debug:
                        print "Converged after %d samples" % i
                    break

                previous_estimates = new_estimates

        return marginal_fi_dict


    def compute_fisher_information_theoretical(self, sigma=None):
        '''
            Compute the theoretical, large N limit estimate of the Fisher Information
            This one assumes a diagonal covariance matrix, wrong for the complete model.
        '''

        assert self.R <= 2, "Not implemented for R>2"

        if self.population_code_type == 'conjunctive':
            rho = 1./(4*np.pi**2/(self.M))
            # rho = 1./(2*np.pi/(self.M))
        elif self.population_code_type == 'feature':
            # M/2 neuron per 2pi dimension.
            print "This looks wrong, doesnt fit at all"
            rho = 1./(np.pi**2./self.M**2.)
        else:
            print 'Fisher information not defined for population type ' + self.population_code_type
            return 0


        kappa1 = self.rc_scale[0]
        kappa2 = self.rc_scale[1]

        return kappa1**2.*rho*(scsp.i0(2*kappa1) - scsp.iv(2, 2*kappa1))*scsp.i0(2*kappa2)/(sigma**2.*8*np.pi**2.*scsp.i0(kappa1)**2.*scsp.i0(kappa2)**2.)



    ########################

    def init_feature_cover_matrices_2d(self, precision=20, endpoint=True):
        '''
            Helper function, creating appropriate linspaces, depending on the chosen coordinate system.
        '''

        feature_space1 = self.init_feature_space(precision=precision, endpoint=endpoint)
        feature_space2 = self.init_feature_space(precision=precision, endpoint=endpoint)
        cross_array = np.zeros((feature_space1.size, feature_space2.size))

        return (feature_space1, feature_space2, cross_array)



    ######################## PLOTS ######################################

    def plot_coverage_feature_space(self, axes=(0, 1), nb_stddev=1.0, specific_neurons=None, alpha_ellipses=0.5, facecolor='rand', ax=None, lim_factor=1.0):
        '''
            Show the features.
            Choose the 2 dimensions you want first.
        '''

        if specific_neurons is None:
            specific_neurons = self._ALL_NEURONS


        ells = [Ellipse(xy=self.neurons_preferred_stimulus[m, axes], width=nb_stddev*utils.kappa_to_stddev(self.neurons_sigma[m, axes[0]]), height=nb_stddev*utils.kappa_to_stddev(self.neurons_sigma[m, axes[1]]), angle=-np.degrees(self.neurons_angle[m])) for m in specific_neurons if not self.mask_neurons_unset[m]]

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')

        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(alpha_ellipses)
            if facecolor is 'rand':
                e.set_facecolor(np.random.rand(3))
            elif facecolor is False or facecolor is None or facecolor == 'none' or facecolor == 'None':
                e.set_facecolor('none')
            else:
                e.set_facecolor(facecolor)
            e.set_transform(ax.transData)

        # ax.autoscale_view()
        # ax.set_xlim(-1.4*np.pi, 1.3*np.pi)
        # ax.set_ylim(-1.4*np.pi, 1.3*np.pi)
        ax.set_xlim(-lim_factor*np.pi, lim_factor*np.pi)
        ax.set_ylim(-lim_factor*np.pi, lim_factor*np.pi)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)

        ax.set_xlabel('Orientation', fontsize=14)
        ax.set_ylabel('Color', fontsize=14)

        ax.set_title('%d vs %d' % (axes[0]+1, axes[1]+1))
        fig.set_tight_layout(True)

        plt.draw()

        plt.show()

        return ax


    def plot_mean_activity(self, precision=100, specific_neurons=None):
        '''
            Plot \sum_i \phi_i(x) at all x
        '''

        (mean_activity, feature_space1, feature_space2) = self.get_mean_activity(precision=precision, specific_neurons=specific_neurons, return_axes_vect=True)

        print "%.3f %.5f" % (np.mean(mean_activity), np.std(mean_activity.flatten()))

        utils.pcolor_2d_data(mean_activity, x=feature_space1, y=feature_space2, xlabel='Color', ylabel='Orientation', colorbar=True, ticks_interpolate=5)

        plt.show()


    def plot_neuron_activity(self, neuron_index=0, nb_stddev=1., precision=100):
        '''
            Plot the activity of one specific neuron over the whole input space.
        '''

        coverage_1D = self.init_feature_space(precision)
        activity = self.get_neuron_activity(neuron_index, precision=precision)

        # Plot it
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(activity.T, origin='lower')
        # im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
        im.set_extent((coverage_1D.min(), coverage_1D.max(), coverage_1D.min(), coverage_1D.max()))
        ax.set_ylabel('Color')
        ax.set_xlabel('Orientation')
        # im.set_interpolation('nearest')
        f.colorbar(im)

        # Plot the ellipse showing one standard deviation
        e = Ellipse(xy=self.neurons_preferred_stimulus[neuron_index], width=nb_stddev*utils.kappa_to_stddev(self.neurons_sigma[neuron_index, 0]), height=nb_stddev*utils.kappa_to_stddev(self.neurons_sigma[neuron_index, 1]), angle=-np.degrees(self.neurons_angle[neuron_index]))

        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)
        e.set_facecolor('white')
        e.set_transform(ax.transData)

        ax.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)
        ax.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
        ax.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'), fontsize=17)

        # Change mouse over behaviour
        def report_pixel(x_mouse, y_mouse):
            # Extract loglik at that position
            try:
                x_display = x_mouse
                y_display = y_mouse

                x_i = np.argmin((coverage_1D - x_display)**2.)
                y_i = np.argmin((coverage_1D - y_display)**2.)

                return "x=%.2f y=%.2f value=%.2f" % (x_display, y_display, activity[x_i, y_i])
            except:
                return ""

        ax.format_coord = report_pixel

        plt.show()


    def plot_neuron_activity_1d(self, neuron_index=0, axis_to_vary=0, fix_preferred_stim=True, fixed_stim=None, precision=100., normalise=True):
        '''
            Plot the activity of a neuron along one dimension. Either provide the stimulus to fix, or let it
            be at its preferred stimulus.
        '''

        feature_space = self.init_feature_space(precision=precision)
        activity = np.zeros(feature_space.shape)

        # Fix the rest of the stimulus
        if fix_preferred_stim:
            stimulus = self.neurons_preferred_stimulus[neuron_index].copy()
        elif fixed_stim is not None:
            stimulus = fixed_stim
        else:
            stimulus = np.zeros(self.R)

        # Compute the response.
        for i in xrange(feature_space.size):
            stimulus[axis_to_vary] = feature_space[i]
            activity[i] = self.get_neuron_response(neuron_index, stimulus)

        # Check the gaussian fit
        gaussian_approx = spst.norm.pdf(feature_space, self.neurons_preferred_stimulus[neuron_index, axis_to_vary], utils.kappa_to_stddev(self.neurons_sigma[neuron_index, axis_to_vary]))

        if normalise:
            activity /= np.max(activity)
            gaussian_approx /= np.max(gaussian_approx)

        # Plot everything
        plt.figure()
        plt.plot(feature_space, activity)
        plt.plot(feature_space, gaussian_approx)

        plt.legend(['Neuron activity', 'Gaussian approximation'])

        plt.show()



    def plot_network_activity(self, stimulus_input=None, nb_stddev=1., ax_handle=None):
        '''
            Plot the activity of the network to a specific stimulus.

            TODO This really should depend on the type of the population code...
        '''

        if stimulus_input is None:
            stimulus_input = self.default_stimulus_input

        # Get the network response
        activity = self.get_network_response(stimulus_input)

        # Plot it
        if ax_handle is None:
            f, ax_handle = plt.subplots()

        ax_handle.plot(activity)
        ax_handle.set_xlabel('Neuron')
        ax_handle.set_ylabel('Activity')

        plt.show()


    def plot_coverage_preferred_stimuli_3d(self, axes=(0, 1)):
        '''
            Show the preferred stimuli coverage on a sphere/torus.
        '''

        utils.scatter3d_torus(self.neurons_preferred_stimulus[:, axes[0]], self.neurons_preferred_stimulus[:, axes[1]])


    def plot_neuron_activity_3d(self, neuron_index=0, precision=20, axes=(0, 1), weight_deform=0.5, draw_colorbar=True):
        '''
            Plot the activity of a neuron on the sphere/torus
        '''

        coverage_1D = self.init_feature_space(precision)
        activity = self.get_neuron_activity(neuron_index, precision=precision, axes=axes)

        utils.plot_torus(coverage_1D, coverage_1D, activity, weight_deform=weight_deform, draw_colorbar=draw_colorbar)


    def plot_mean_activity_3d(self, precision=20, axes=(0, 1), specific_neurons=None, weight_deform=0.5):
        '''
            Plot the mean activity of the network on a sphere/torus
        '''

        (mean_activity, feature_space1, feature_space2) = self.get_mean_activity(precision=precision, specific_neurons=specific_neurons, return_axes_vect=True, axes=axes)

        utils.plot_torus(feature_space1, feature_space2, mean_activity, weight_deform=weight_deform)


    ##########################

    @classmethod
    def create_full_conjunctive(cls, M, R=2, rcscale=None, autoset_parameters=False, response_maxout=False, debug=True):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''

        if debug:
            print "create conjunctive network, R=%d, M=%d, autoset: %d" % (R, M, autoset_parameters)

        rn = HighDimensionNetwork(M, R=R, response_maxout=response_maxout)
        rn.population_code_type = 'conjunctive'

        ## Create receptive fields

        if autoset_parameters:
            # We use the optimum heuristic for the rc_scale: try to cover the space fully, assuming uniform coverage with squares of size 2*(2*utils.kappa_to_stddev(kappa)). We assume that 2*stddev gives a good approximation to the appropriate coverage required.
            # rcscale = utils.stddev_to_kappa(2.*np.pi/int(M**0.5))
            rcscale = cls.compute_optimal_rcscale(M, R, population_code_type='conjunctive')

        # Assume we construct a conjunctive with ratio 1, no need to get random eigenvectors
        rn.assign_aligned_eigenvectors(scale=rcscale, ratio=1.0, reset=True)

        return rn

    @classmethod
    def create_full_features(cls, M, R=2, scale=0.3, ratio=40., autoset_parameters=False, response_maxout=False):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "create feature network, R=%d, M=%d, autoset: %d" % (R, M, autoset_parameters)

        rn = HighDimensionNetwork(M, R=R, response_maxout=response_maxout)

        if autoset_parameters:
            # Use optimal values for the parameters. Be careful, this assumes M/2 and coverage of full 2 pi space
            # Assume one direction should cover width = pi, the other should cover M/2 * width/2. = 2pi
            # width = utils.stddev_to_kappa(stddev)

            scale = utils.stddev_to_kappa(np.pi)
            scale2 = cls.compute_optimal_rcscale(M, R, population_code_type='feature')
            ratio = scale2/scale
        else:
            if ratio < 0.0:
                # Setting ratio < 0 cause some mid-automatic parameter setting.
                # Assume that only one scale is really desired, and the other automatically set.
                scale_fixed = utils.stddev_to_kappa(np.pi)
                ratio = np.max((scale/scale_fixed, scale_fixed/scale))

                print "Semi auto ratio: %f %f %f" % (scale, scale_fixed, ratio)

        M_sub = M/rn.R
        # Assign centers
        rn.assign_prefered_stimuli(tiling_type='features', reset=True)

        # Now assign scales
        resetted = False
        for r in xrange(rn.R):
            rn.assign_aligned_eigenvectors(scale=scale, ratio=ratio, scaled_dimension=r, specific_neurons=np.arange(r*M_sub, (r+1)*M_sub), reset=not resetted)

            resetted = True

        rn.population_code_type = 'feature'

        return rn

    @classmethod
    def create_mixed(cls, M, R=2, ratio_feature_conjunctive=0.5, conjunctive_parameters=None, feature_parameters=None, autoset_parameters=False, response_maxout=False):
        '''
            Create a RandomFactorialNetwork instance, using a pure conjunctive code
        '''
        print "Create mixed network, R=%d autoset: %d" % (R, autoset_parameters)

        conj_scale = 1.0
        feat_scale = 0.3
        feat_ratio = 40.0

        if conjunctive_parameters is not None:
            # Heavily refactored, but keeps compatibility...
            conj_scale = conjunctive_parameters['scale']

        if feature_parameters is not None:
            feat_scale = feature_parameters['scale']
            feat_ratio = feature_parameters['ratio']

            # nb_feature_centers = feature_parameters.get('nb_feature_centers', 1)

        rn = HighDimensionNetwork(M, R=R, response_maxout=response_maxout)

        rn.conj_subpop_size = int(np.round(M*ratio_feature_conjunctive))
        rn.feat_subpop_size = M - rn.conj_subpop_size

        if autoset_parameters:
            # Use optimal values for the parameters. Be careful, this assumes M/2 and coverage of full 2 pi space
            # Assume one direction should cover width = pi, the other should cover M/2 * width/2. = 2pi
            # width = utils.stddev_to_kappa(stddev)
            if rn.conj_subpop_size > 0:
                conj_scale = cls.compute_optimal_rcscale(rn.conj_subpop_size, R, population_code_type='conjunctive')
            if rn.feat_subpop_size > 0:
                feat_scale = utils.stddev_to_kappa(np.pi)
                feat_ratio = cls.compute_optimal_rcscale(rn.feat_subpop_size, R, population_code_type='feature')/feat_scale

        print "Population sizes: ratio: %.2f conj: %d, feat: %d, autoset: %d" % (ratio_feature_conjunctive, rn.conj_subpop_size, rn.feat_subpop_size, autoset_parameters)

        # Create the conjunctive subpopulation
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specific_neurons=np.arange(rn.conj_subpop_size))
        rn.assign_aligned_eigenvectors(scale=conj_scale, ratio=1.0, specific_neurons=np.arange(rn.conj_subpop_size), reset=True)

        # Create the feature subpopulation
        # Assign centers
        rn.assign_prefered_stimuli(tiling_type='features', specific_neurons=np.arange(rn.conj_subpop_size, M))
        # Now assign scales
        feat_sub_M = rn.feat_subpop_size/rn.R
        for r in xrange(rn.R):
            rn.assign_aligned_eigenvectors(scale=feat_scale, ratio=feat_ratio, scaled_dimension=r, specific_neurons=np.arange(rn.conj_subpop_size + r*feat_sub_M, rn.conj_subpop_size + (r+1)*feat_sub_M))

        rn.population_code_type = 'mixed'
        rn.ratio_conj = ratio_feature_conjunctive

        return rn


    @classmethod
    def compute_optimal_rcscale(cls, M, R, population_code_type='conjunctive'):
        '''
            Compute the optimal rcscale, depending on the type of code we use.
        '''
        if population_code_type == 'conjunctive':
            # We use the optimum heuristic for the rc_scale: try to cover the space fully, assuming uniform coverage with squares of size 2*(2*utils.kappa_to_stddev(kappa)). We assume that 2*stddev gives a good approximation to the appropriate coverage required.
            return utils.stddev_to_kappa(2.*np.pi/int(M**(1./R)))
        elif population_code_type == 'feature':
            return utils.stddev_to_kappa(2.*np.pi/int(M/R))


def test_optimised_network_response():
    R = 2
    M = int(50*R + 10**R)
    ratio = 0.1
    rn = HighDimensionNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio, autoset_parameters=True, response_maxout=False)

    print "Testing if optimised and non-optimised network response are the same..."
    rnd_angles = utils.sample_angle((10000, R))
    all_correct = True
    for curr_angles in rnd_angles:
        all_correct = all_correct and np.allclose(rn.get_network_response_bivariatefisher(curr_angles, params=dict(opt=True)), rn.get_network_response_bivariatefisher(curr_angles, params=dict(opt=False)))

    assert all_correct, "Optimised and normal network response do not correspond..."


def test_large_kappa_network_response():
    R = 3
    M = (10**R)
    ratio = 1.0
    rn = HighDimensionNetwork.create_mixed(M, R=R, ratio_feature_conjunctive=ratio, autoset_parameters=True)

    rn.get_network_response()


if __name__ == '__main__':
    pass

    # # Pure conjunctive code
    # if False:
    #     R = 2
    #     M = int(5**R)
    #     rn = HighDimensionNetwork.create_full_conjunctive(M, R=R, autoset_parameters=True)

    #     # rn.plot_coverage_feature_space()

    #     # rn.plot_mean_activity()
    #     # rn.plot_neuron_activity(15)

    # # Pure feature code
    # if False:
    #     R = 2
    #     M = int(20*R)
    #     rn = HighDimensionNetwork.create_full_features(M, R=R, autoset_parameters=True)

    #     # rn.plot_coverage_feature_space(specific_neurons=np.arange(0, M, M/10))

    #     # rn.plot_mean_activity(specific_neurons=np.arange(0, M, M/10))

    # # Mix of two population
    # if False:
    #     R = 2
    #     M = int(20*R + 10**R)
    #     ratio = 0.7
    #     rn = HighDimensionNetwork.create_mixed(M, R=R, ratio_feature_conjunctive = ratio, autoset_parameters=True)

    # # Check effect for R>2
    # if False:
    #     R = 4
    #     M = 500
    #     ratio = 0.7
    #     rn = HighDimensionNetwork.create_mixed(M, R=R, ratio_feature_conjunctive = ratio, autoset_parameters=True)
