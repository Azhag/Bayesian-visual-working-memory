#!/usr/bin/env python
# encoding: utf-8
"""
hierarchicalrandomnetwork.py

Created by Loic Matthey on 2013-05-09.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
# import scipy as sp
# import scipy.special as scsp
# import progress
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Ellipse
import matplotlib.colors as pltcol

from highdimensionnetwork import HighDimensionNetwork
import utils


class HierarchialRandomNetwork(object):
  """Network built hiearchically.

  Consist of two layers:
  - The first one provides some smooth basis on a conjunctive space of features.
  - The second samples the first randomly and computes a non-linear weighted
  sum of them
  """

  def __init__(self,
               M,
               R=2,
               gain=1.0,
               ratio_hierarchical=None,
               M_layer_one=100,
               type_layer_one='feature',
               output_both_layers=False,
               optimal_coverage=True,
               rcscale_layer_one=5.0,
               ratio_layer_one=200.0,
               nonlinearity_fct='positive_linear',
               threshold=0.0,
               sparsity_weights=0.7,
               distribution_weights='exponential',
               sigma_weights=0.5,
               normalise_weights=True,
               normalise_gain=False,
               debug=True):

    assert R == 2, 'HiearchialRandomNetwork defined over two features for now'

    if ratio_hierarchical is not None:
      self.M_layer_two = int(np.round(ratio_hierarchical * M))
      self.M_layer_one = M - self.M_layer_two

      assert self.M_layer_one > 2, "Cannot have such a small layer one: {}".format(
          self.M_layer_one)
    else:
      self.M_layer_two = M
      self.M_layer_one = M_layer_one

    if output_both_layers:
      self.M = self.M_layer_two + self.M_layer_one
    else:
      self.M = self.M_layer_two

    self.rcscale_layer_one = rcscale_layer_one
    self.ratio_layer_one = ratio_layer_one
    self.optimal_coverage = optimal_coverage
    self.sigma_weights = sigma_weights
    self.normalise_weights = normalise_weights
    self.distribution_weights = distribution_weights
    self.type_layer_one = type_layer_one
    self.output_both_layers = output_both_layers

    self._ALL_NEURONS = np.arange(M)

    self.R = R
    self.gain = gain
    self.sparsity_weights = sparsity_weights

    self.layer_one_network = None
    self.current_layer_one_response = None
    self.current_layer_two_response = None

    # Used to stored cached network response statistics. Mean_theta(mu(theta)) and Cov_theta(mu(theta))
    self.network_response_statistics = None

    self.debug = debug

    if self.debug:
      print "-> Building HierarchicalRandomNetwork"

    # Initialise everything
    self.construct_layer_one(type_layer=type_layer_one)
    self.construct_nonlinearity_fct(fct=nonlinearity_fct, threshold=threshold)
    self.construct_A_sampling(
        sparsity_weights=sparsity_weights,
        distribution_weights=distribution_weights,
        sigma_weights=sigma_weights,
        normalise_weights=normalise_weights)

    if normalise_gain:
      self.gain /= self.compute_maximum_activation_network()

    self.population_code_type = 'hierarchical'
    self.coordinates = 'full_angles_sym'
    self.network_initialised = True

    print "done, output_both_layers:%d, M_higher %d, M_lower %d" % (
        self.output_both_layers, self.M_layer_two, self.M_layer_one)

  def construct_layer_one(self, type_layer='conjunctive'):
    """Initialises the first layer of the hiearchical network

    Consists of another RFN, makes everything simpler and more logical
    """
    if type_layer == 'conjunctive':
      self.layer_one_network = \
          HighDimensionNetwork.create_full_conjunctive(
              self.M_layer_one,
              R=self.R,
              rcscale=self.rcscale_layer_one,
              autoset_parameters=self.optimal_coverage,
              response_type='bivariate_fisher')

    elif type_layer == 'feature':
      self.layer_one_network = \
          HighDimensionNetwork.create_full_features(
              self.M_layer_one,
              R=self.R,
              scale=self.rcscale_layer_one,
              ratio=self.ratio_layer_one,
              autoset_parameters=self.optimal_coverage)
    else:
      raise NotImplementedError('type_layer is conjunctive only for now')

  def construct_nonlinearity_fct(self,
                                 fct='exponential',
                                 threshold=0.0,
                                 correct_treshold=True):
    """Set a nonlinearity function for the second layer

    Input:
        fct: if function, used as it is. If string, switch between
            exponential, identity, rectify
    """

    if utils.is_function(fct):
      # Function given, just use that
      self.nonlinearity_fct = fct
    else:

      # Switch based on some supported functions
      if fct == 'exponential':
        self.nonlinearity_fct = np.exp
      elif fct == 'identity':
        self.nonlinearity_fct = lambda x: x
      elif fct == 'positive_linear':

        self.threshold = threshold

        if correct_treshold:
          # Correct response to account for 4 pi^2 term

          def positive_linear(x):
            return (4. * np.pi**2. * x - self.threshold).clip(0.0)
        else:

          def positive_linear(x):
            return (x - self.threshold).clip(0.0)

        self.nonlinearity_fct = positive_linear

  def construct_A_sampling(self,
                           sparsity_weights=0.1,
                           distribution_weights='exponential',
                           sigma_weights=0.1,
                           normalise_weights=False):
    """Creates the sampling matrix A for the network.

    Should have a small (sparsity amount) of non-zero weights. Weights are sampled independently from a gaussian distribution.
    """

    if distribution_weights == 'randn':
      self.A_sampling = sigma_weights * np.random.randn(
          self.M_layer_two, self.M_layer_one) * (np.random.rand(
              self.M_layer_two, self.M_layer_one) <= sparsity_weights)
    elif distribution_weights == 'exponential':
      self.A_sampling = (np.random.exponential(sigma_weights,
                                               (self.M_layer_two,
                                                self.M_layer_one)) *
                         (np.random.rand(self.M_layer_two, self.M_layer_one) <=
                          sparsity_weights))
    else:
      raise ValueError('distribution_weights should be randn/exponential')

    if normalise_weights == 1:
      # Normalise the rows to get a maximum activation level per neuron
      # self.A_sampling = self.A_sampling/np.sum(self.A_sampling, axis=0)
      self.A_sampling /= np.sum(self.A_sampling, axis=1)[:, np.newaxis]
    elif normalise_weights == 2:
      # Normalise the network activity by the number of layer one neurons
      # self.gain /= self.M_layer_one * sigma_weights
      pass
    elif normalise_weights == 3:
      # Normalise by max weight per row, i.e. have inputs normalized
      self.A_sampling = self.A_sampling / np.max(
          self.A_sampling, axis=-1)[:, np.newaxis]

    self.A_sampling[np.isnan(self.A_sampling)] = 0
    return self.A_sampling

  ##
  # Network behaviour
  ##
  def get_network_response(self,
                           stimulus_input=None,
                           specific_neurons=None,
                           params={}):
    """Output the activity of the network for the provided input.

    Can return either layer 2 or layer 1+2.
    """

    if stimulus_input is None:
      stimulus_input = (0.0, ) * self.R

    # Get the response of layer one to the stimulus
    layer_one_response = self.get_layer_one_response(stimulus_input)

    # Combine those responses according the the sampling matrices
    layer_two_response = self.get_layer_two_response(
        layer_one_response, specific_neurons=specific_neurons)

    if self.output_both_layers and specific_neurons is None:
      # Should return the activity of both layers collated
      # (handle stupid specific_neurons filter case in the cheapest way possible: don't support it)
      return np.r_[layer_two_response, layer_one_response]
    else:
      # Only layer two is relevant
      return layer_two_response

  def get_neuron_response(self, neuron_index, stimulus_input):
    """Get the output of one specific neuron, for a specific stimulus.
    """
    return self.get_network_response(stimulus_input)[neuron_index]

  def sample_network_response(self, stimulus_input, sigma=0.1):
    """Get a random response for the given stimulus.

    Returns:
      Mx1
    """
    return self.get_network_response(stimulus_input) + sigma * np.random.randn(
        self.M)

  def get_layer_one_response(self, stimulus_input=None, specific_neurons=None):
    """Compute/updates the response of the first layer to the given stimulus

    The first layer is a normal RFN, so simply query it for its response
    """
    self.current_layer_one_response = self.layer_one_network.get_network_response(
        stimulus_input=stimulus_input, specific_neurons=specific_neurons)

    return self.current_layer_one_response

  def get_layer_two_response(self, layer_one_response, specific_neurons=None):
    """Compute/updates the response of the second layer, based on the response of the first layer

    The activity is given by:
      x_2 = f(A x_1)

    Where:
      - x_1 is the response of the first layer
      - A is the sampling matrix, random and sparse usually
      - f is a nonlinear function
    """

    if specific_neurons is None:
      self.current_layer_two_response = self.gain * self.nonlinearity_fct(
          np.dot(self.A_sampling, layer_one_response))
    else:
      self.current_layer_two_response = self.gain * self.nonlinearity_fct(
          np.dot(self.A_sampling[specific_neurons], layer_one_response))

    return self.current_layer_two_response

  def compute_maximum_activation_network(self, nb_samples=100):
    """Try to estimate the maximum activation for the network.

    This can be used to make sure sigmax is adapted, or to renormalize everything.
    """

    test_samples = utils.sample_angle((nb_samples, self.R))

    max_activation = 0
    for test_sample in test_samples:
      max_activation = max(
          np.nanmax(self.get_network_response(test_sample)), max_activation)

    return max_activation

  def get_neuron_activity_full(self, neuron_index, precision=100, axes=None):
    """Returns the activity of a specific neuron over the entire space.
    """

    coverage_1D = utils.init_feature_space(precision)

    possible_stimuli = np.array(utils.cross(self.R * [coverage_1D.tolist()]))
    activity = np.empty(possible_stimuli.shape[0])

    # Compute the activity of that neuron over the whole space
    for stimulus_i, stimulus in enumerate(possible_stimuli):
      activity[stimulus_i] = self.get_neuron_response(neuron_index, stimulus)

    # Reshape
    activity.shape = self.R * (precision, )

    if axes is not None:
      for dim_to_avg in set(range(len(activity.shape))) - set(axes):
        activity = np.mean(activity, axis=dim_to_avg, keepdims=True)
      activity = np.squeeze(activity)

    return activity

  def get_mean_activity(self,
                        axes=(0, 1),
                        precision=100,
                        specific_neurons=None,
                        return_axes_vect=False):
    """Returns the mean activity of the network.
    """

    coverage_1D = utils.init_feature_space(precision)

    possible_stimuli = np.array(utils.cross(self.R * [coverage_1D.tolist()]))
    activity = np.empty((possible_stimuli.shape[0], self.M))

    for stimulus_i, stimulus in enumerate(possible_stimuli):
      activity[stimulus_i] = self.get_network_response(
          stimulus, specific_neurons=specific_neurons)

    # Reshape
    activity.shape = self.R * (precision, ) + (self.M, )

    mean_activity = activity

    for dim_to_avg in (set(range(len(activity.shape))) - set(axes)):
      mean_activity = np.mean(mean_activity, axis=dim_to_avg, keepdims=True)

    mean_activity = np.squeeze(mean_activity)

    if return_axes_vect:
      return (mean_activity, coverage_1D, coverage_1D)
    else:
      return mean_activity

  ##
  # Theoretical stuff
  ##

  def compute_marginal_inverse_FI(self,
                                  inv_cov_stim,
                                  nitems=1,
                                  max_n_samples=int(1e5),
                                  items_thetas=None,
                                  min_distance=0.1,
                                  convergence_epsilon=1e-7,
                                  debug=True):
    return dict(inv_FI=0.0, inv_FI_std=0.0, FI=0.0, FI_std=0.0)

  def compute_fisher_information(self,
                                 stimulus_input,
                                 cov_stim=None,
                                 inv_cov_stim=None):
    return 0.0

  ##
  # Plots
  ##

  def plot_network_activity(self, stimulus_input=None):
    """Plot the activity of the whole network on a specific stimulus.

    Shows activations of both layers
    """

    if stimulus_input is None:
      stimulus_input = (0, ) * self.R

    # Compute activity of network on the stimulus
    self.get_network_response(stimulus_input=stimulus_input)

    # Do a subplot, second layer on top, first layer on bottom
    plt.figure()
    ax_layertwo = plt.subplot(2, 1, 1)
    ax_layerone = plt.subplot(2, 1, 2)

    # Plot the level two activation, use a bar, easier to read
    ax_layertwo.bar(
        np.arange(self.M_layer_two), self.current_layer_two_response)

    # Plot the activation of the level one subnetwork (and of the individual responses at level two)
    M_sqrt = int(self.M_layer_one**0.5)

    # Level one
    im = ax_layerone.imshow(
        self.current_layer_one_response[:int(M_sqrt**2)].reshape(
            M_sqrt, M_sqrt).T,
        origin='lower',
        aspect='equal',
        cmap='RdBu_r',
        interpolation='nearest')
    im.set_extent((-np.pi, np.pi, -np.pi, np.pi))
    ax_layerone.set_xticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
    ax_layerone.set_xticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$',
                                 r'$\frac{\pi}{2}$', r'$\pi$'))
    ax_layerone.set_yticks((-np.pi, -np.pi / 2, 0, np.pi / 2., np.pi))
    ax_layerone.set_yticklabels((r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$',
                                 r'$\frac{\pi}{2}$', r'$\pi$'))

    e = Ellipse(xy=stimulus_input, width=0.4, height=0.4)

    ax_layerone.add_artist(e)
    e.set_clip_box(ax_layerone.bbox)
    e.set_alpha(0.5)
    e.set_facecolor('white')
    e.set_transform(ax_layerone.transData)

    plt.show()

  def plot_neuron_activity_full(self,
                                neuron_index=0,
                                precision=100,
                                ax_handle=None,
                                draw_colorbar=True,
                                cmap='RdBu_r'):
    """Plot the activity of one specific neuron over the whole input space.
    """
    coverage_1D = utils.init_feature_space(precision)
    activity = self.get_neuron_activity_full(neuron_index, precision=precision)

    # Plot it
    ax_handle, _ = utils.pcolor_2d_data(
        activity,
        x=coverage_1D,
        y=coverage_1D,
        ticks_interpolate=5,
        ax_handle=ax_handle,
        colorbar=draw_colorbar,
        cmap=cmap)

    # Change the ticks
    selected_ticks = np.array(
        np.linspace(0, coverage_1D.size - 1, 5), dtype=int)
    ax_handle.set_xticks(selected_ticks)
    ax_handle.set_xticklabels(
        (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'),
        fontsize=17,
        rotation=0)
    ax_handle.set_yticks(selected_ticks)
    ax_handle.set_yticklabels(
        (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'),
        fontsize=17,
        rotation=0)

  def plot_mean_activity(self, precision=100, specific_neurons=None):
    """Plot \sum_i \phi_i(x) at all x
    """

    coverage_1D = utils.init_feature_space(precision)
    mean_activity = self.get_mean_activity(
        precision=precision, specific_neurons=specific_neurons)

    print(np.mean(mean_activity), np.std(mean_activity.flatten()))

    ax, im = utils.pcolor_2d_data(
        mean_activity,
        x=coverage_1D,
        y=coverage_1D,
        xlabel='Color',
        ylabel='Orientation',
        colorbar=True,
        ticks_interpolate=5,
        cmap='RdBu_r')

    return ax, im

  def plot_coverage_feature(self,
                            nb_layer_two_neurons=3,
                            nb_stddev=1.0,
                            alpha_ellipses=0.5,
                            facecolor_layerone='b',
                            ax=None,
                            lim_factor=1.1,
                            top_neurons=None,
                            precision=100):
    """Plot the coverage of the network

    Do a subplot, show the lower layer coverage, and for the random upper layer show some "receptive fields"
    """
    coverage_1D = utils.init_feature_space(precision)

    # Select layer two neurons to be plotted randomly
    if top_neurons is None:
      top_neurons = np.random.randint(
          self.M_layer_two, size=nb_layer_two_neurons)

    # Get the activities of those layer two neurons
    activities_layertwo = np.zeros((nb_layer_two_neurons, precision,
                                    precision))
    for i, layer_two_neuron in enumerate(top_neurons):
      activities_layertwo[i] = self.get_neuron_activity_full(
          layer_two_neuron, precision=precision)

    # Construct the plot
    f = plt.figure()

    axes_top = ImageGrid(
        f,
        211,
        nrows_ncols=(1, nb_layer_two_neurons),
        direction="row",
        axes_pad=0.2,
        add_all=True,
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.2)

    selected_ticks = np.array(
        np.linspace(0, coverage_1D.size - 1, 5), dtype=int)

    for ax_top, activity_neuron in zip(axes_top, activities_layertwo):
      im = ax_top.imshow(activity_neuron.T, origin='lower left', cmap='RdBu_r')

      ax_top.set_xticks(selected_ticks)
      ax_top.set_xticklabels(
          (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
           r'$\pi$'),
          fontsize=16)
      ax_top.set_yticks(selected_ticks)
      ax_top.set_yticklabels(
          (r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
           r'$\pi$'),
          fontsize=16)

      # ax_top.set_xlabel('Orientation')
      # ax_top.set_ylabel('Colour')

    ax_top.cax.colorbar(im)
    ax_top.cax.toggle_label(True)

    # Bottom part now
    ax_bottom = f.add_subplot(2, 3, 5)
    plt.subplots_adjust(hspace=0.5)
    self.layer_one_network.plot_coverage_feature_space(
        nb_stddev=2,
        facecolor=facecolor_layerone,
        alpha_ellipses=alpha_ellipses,
        lim_factor=lim_factor,
        ax=ax_bottom,
        specific_neurons=np.arange(0, self.M_layer_one, 2))

    return axes_top, ax_bottom