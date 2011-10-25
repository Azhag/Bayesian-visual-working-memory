#!/usr/bin/env python
# encoding: utf-8
"""
random_network.py

Created by Loic Matthey on 2011-06-10.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
import mdp
from matplotlib.patches import Ellipse

from populationcode import *
from hinton_plot import *
from utils import *

class RandomNetwork:
    
    def __init__(self, M, D=50, R=1, sigma_pop=0.6, rho_pop=0.5, gamma_pop=0.1, W_type='identity', W_parameters=[0.5], percentage_population_connections = 0.4, max_angle=2.*np.pi):
        '''
            M: number of random neurons
            D: number of population neurons
            R: number of populations
        '''
        
        self.M = M
        self.D = D
        self.K = 0
        self.R = R
        
        # Create the population codes
        self.popcodes = [PopulationCodeAngle(D, sigma=sigma_pop, rho=rho_pop, gamma=gamma_pop, max_angle=max_angle) for r in np.arange(R)]
        
        # Initialise the possible representations of the orientations and colors
        self.network_representations = None
        
        self.network_initialised = False
        
        self.build_W(W_type=W_type, W_parameters=W_parameters)
        
        
        print "RandomNetwork initialised"
        
    
    
    def assign_possible_orientations(self, possible_angles):
        '''
            Get the mean responses for all the possible stimuli. They will become the possible features.
            [Weird because it should be a continuous variable...]
            
            network_representations:    R x K x M
        '''
        self.K = possible_angles.size
        self.possible_angles = possible_angles
        
        # Those are the "clean" orientations from the population code
        self.popcodes_representations = np.zeros((self.R, self.K, self.D))
        for r in np.arange(self.R):
            self.popcodes_representations[r] = self.popcodes[r].mean_response(possible_angles)
        
        # Those are the network representations
        self.network_representations = np.zeros((self.R, self.K, self.M))
        for r in np.arange(self.R):
            self.network_representations[r] = np.dot(self.popcodes_representations[r], self.W[r].T)
        
        # Define the possible objects
        self.possible_objects_indices = np.array(cross([[x for x in np.arange(self.K)]]*self.R))
        self.possible_objects = np.array(cross([[x for x in self.possible_angles]]*self.R))
        
        self.network_initialised = True
    
    
    def build_W(self, W_type='identity', W_parameters=[0.2]):
        '''
            Build the connectivity matrix.
            
            W:  R x M x D
        '''
        
        self.W_type=W_type
        self.W = np.zeros((self.R, self.M, self.D))
        
        if W_type == 'identity':
            self.build_W_identity()
        elif W_type=='random':
            self.build_W_random(W_parameters)
        elif W_type == 'dirichlet':
            self.build_W_dirichlet(W_parameters)
        elif W_type == 'none':
            pass
        else:
            raise ValueError('Type of connectivity unknown')
    
    
    def build_W_identity(self):
        if self.M >= 2*self.D:
            for r in np.arange(self.R):
                self.W[r, self.D*r:self.D*(r+1), :self.D] = np.eye(self.D)
        else:
            self.W[:, :self.D, :self.D] = np.tile(np.eye(self.D), (self.R, 1, 1))
    
    
    def build_W_random(self, W_parameters):
        # Unpack parameters
        percentage_population_connections = W_parameters[0]
        
        mask = np.random.rand(self.M, self.D) < percentage_population_connections
        empty_rows = np.all(mask == False, axis=1)
        while np.any(empty_rows):
            mask[empty_rows] = np.random.rand(np.sum(empty_rows), self.D) < percentage_population_connections
            empty_rows = np.all(mask == False, axis=1)
        
        self.W = np.random.rand(self.R, self.M, self.D)
        
        self.W = self.W*mask
        for r in np.arange(self.R):
            self.W[r] = (self.W[r].T/np.sum(self.W[r], axis=1)).T
    
    def build_W_dirichlet(self, W_parameters):
        # Unpack parameters
        percentage_population_connections = 0.1
        dirichlet_concentration = 0.5
        sigma_W = 0.8
        
        nb_params = np.size(W_parameters)
        if nb_params >= 1:
            percentage_population_connections = W_parameters[0]
        if nb_params >= 2:
            dirichlet_concentration = W_parameters[1]
        if nb_params >= 3:
            sigma_W = W_parameters[2]
        
        # Get random number of connections to each feature set for each neuron
        # use a Dirichlet, the concentration parameter controls
        # how "even" the sample is (~more binding)
        #   alpha small => no binding (one big, others small)
        #   alpha big => much binding (~ all same number)
        self.ratio_connections = np.random.dirichlet(np.ones(self.R)*dirichlet_concentration, size=self.M)
        
        # Now assume that the number of connections is a percentage of the total number of feature neurons
        mean_number_connections = self.R*self.D*percentage_population_connections
        
        # Get the actual random number of connections for each neuron
        self.number_connections = np.round(self.ratio_connections*mean_number_connections).astype(int)
        
        # Now connect neurons to features accordingly. Choose K_i_n features uniformly.
        for m in np.arange(self.M):
            for r in np.arange(self.R):
                indices = np.random.permutation(np.arange(self.D))[:self.number_connections[m, r]]
                # self.W[r, m, indices] = sigma_W*np.random.randn(np.min((self.D, self.number_connections[m, r])))
                self.W[r, m, indices] = sigma_W
        
        # self.W = np.abs(self.W)
    
    
    ###
    
    def sample_network_response_indices(self, chosen_orientations):
        '''
            Get a random response for a/multiple orientation(s) indices from the population code,
            transform it through W and return that
            
            return: R x number_input_orientations x M
        '''
        
        dim = chosen_orientations.shape
        
        if np.size(dim) > 1:
            net_samples = np.zeros((self.R, dim[1], self.M))
        else:
            net_samples = np.zeros((self.R, dim[0], self.M))
        
        for r in np.arange(self.R):
            if np.size(dim) > 1:
                # We have different orientations for the different population codes. It should be on the first dimension.
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(self.possible_angles[chosen_orientations[r]]), self.W[r].T)
            else:
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(self.possible_angles[chosen_orientations]), self.W[r].T)
        
        return net_samples
    
    
    def sample_network_response(self, chosen_orientations, summed=True):
        '''
            Get a random response for a/multiple orientation(s) from the population code,
            transform it through W and return that
            
            return: R x number_input_orientations x M
        '''
        
        dim = chosen_orientations.shape
        
        if np.size(dim) > 1:
            net_samples = np.zeros((self.R, dim[1], self.M))
        else:
            net_samples = np.zeros((self.R, dim[0], self.M))
        
        for r in np.arange(self.R):
            if np.size(dim) > 1:
                # We have different orientations for the different population codes. It should be on the first dimension.
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(chosen_orientations[r]), self.W[r].T)
            else:
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(chosen_orientations), self.W[r].T)
        
        if summed:
            return np.sum(net_samples, axis=0)
        else:
            return net_samples
        
    
    
    def get_network_features_combined(self, Z):
        
        if Z.ndim == 1:
            #sum_features = np.trace(self.network_representations[:, Z])
            
            # MASSIVE Speed up, for R=2
            sum_features = self.network_representations[0, Z[0]] + self.network_representations[1, Z[1]]
        elif Z.ndim == 3:
            (N, T, R) = Z.shape
            sum_features = np.zeros((N, T, self.M))
            for r in np.arange(self.R):
                sum_features += self.network_representations[r, Z[:,:,r], :]
        else:
            raise ValueError('Wrong dimensionality for Z')
        
        return sum_features
    
    
    def get_network_features_combined_binary(self, Z):
        
        if Z.ndim == 2:
            #sum_features = np.trace(self.network_representations[:, Z])
            
            # sum_features = np.tensordot(Z, self.network_representations)
            sum_features = np.dot(Z[0], self.network_representations[0]) + np.dot(Z[1], self.network_representations[1])
        elif Z.ndim == 4:
            # sum_features = np.tensordot(Z, self.network_representations, axes=[[3,2], [1,0]])
            sum_features = np.tensordot(Z, self.network_representations, axes=2)
        else:
            raise ValueError('Wrong dimensionality for Z')
        
        return sum_features
    
    
    def get_popcode_response(self, theta, r):
        '''
            Return the output of one population code
        '''
        return np.dot(self.popcodes[r].mean_response(theta), self.W[r].T)
    
    
    
    ############
    
    def plot_population_representation(self):
        '''
            Plot the response of the population codes
        '''
        
        for r in np.arange(self.R):
            self.popcodes[r].plot_population_representation(self.possible_angles)
        
    
    
    def plot_spread_full_representation(self):
        '''
            Compute all possibles "objects" (features combinations) representations, and plot them.
            Use PCA to reduce the dimensionality
        '''
        
        
        # Retrieve the representations
        all_objects_repr_r = np.zeros((self.R, self.possible_objects_indices.shape[0], self.M))
        for r in np.arange(self.R):
            all_objects_repr_r[r] = self.network_representations[r, self.possible_objects_indices[:,r]]
        
        # They get summed up
        all_objects_repr = np.sum(all_objects_repr_r, axis=0)
        
        # Plot them
        pca_node = mdp.nodes.PCANode(output_dim=3)
        all_objects_pc = pca_node(all_objects_repr)
        explained_variance = pca_node.get_explained_variance()
        
        import mpl_toolkits.mplot3d as p3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        colors_groups = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        colors_interpolation = np.linspace(0.,1., all_objects_pc.shape[0])
        
        # for k in np.arange(self.K):
            # ax.scatter(all_objects_pc[k*self.K:(k+1)*self.K, 0], all_objects_pc[k*self.K:(k+1)*self.K,1], all_objects_pc[k*self.K:(k+1)*self.K,2], c=colors_groups[k%len(colors_groups)])
            
        ax.scatter(all_objects_pc[:, 0], all_objects_pc[:,1], all_objects_pc[:,2], c=colors_interpolation)
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        
        # Compute some statistics
        U, s, V = np.linalg.svd(all_objects_repr)
        rank_representation = np.sum(s > 1e-6)
        
        print "Rank: %d (%d max)" % (int(rank_representation), all_objects_repr.shape[0])
        
        return (explained_variance, rank_representation, all_objects_repr)
        
    
    
    def plot_network_representation(self):
        '''
            Plot the response of the network
        '''
        
        if self.W_type == 'dirichlet':
            # Sorting that emphasis balance
            balanced_indices_neurons = self.number_connections[:,0].argsort()[::-1]
        else:
            balanced_indices_neurons = np.arange(self.M)
        
        # Plot the population response
        plot_separation_y = 0.3*(np.max(self.network_representations) - np.min(self.network_representations))
        
        fig1, ax1 = plt.subplots(1)
        
        for r in np.arange(self.R):
            ax1.plot(self.network_representations[r, :, balanced_indices_neurons] + np.arange(self.K)*plot_separation_y + r*(self.K+0.5)*plot_separation_y)
            
        ax1.autoscale(tight=True)
        
        # Plot Hinton graphs
        sf, ax = plt.subplots(self.R, 1)
        
        for r in np.arange(self.R):
            hinton(self.W[r, balanced_indices_neurons].T, ax=ax[r])
        
    
    
    @classmethod
    def create_instance_uniform(cls, K, M, D=50, R=1, sigma=0.2, rho=0.01, gamma=0.01, W_type='identity', W_parameters=[0.5], max_angle=2.*np.pi):
        '''
            Create a RandomNetwork instance, and fill-in the K possible orientations, uniformly in [0, 2pi]
                If multiple features (R>1), uses the same parameters for everybody (wait till location needed)
        '''
        rn = RandomNetwork(M, D=D, R=R, sigma_pop=sigma, rho_pop=rho, gamma_pop=gamma, W_type=W_type, W_parameters=W_parameters, max_angle=max_angle)
        
        possible_angles = np.linspace(0., max_angle, K, endpoint=False)
        
        rn.assign_possible_orientations(possible_angles)
        
        return rn
    
    


class RandomNetworkFactorialCode(RandomNetwork):
    def __init__(self, M=1, D=50, R=1, sigma_pop=0.01, rho_pop=0.5, gamma_pop=0.1, W_type='identity', W_parameters=[0.5], percentage_population_connections = 0.4, max_angle=2.*np.pi):
        
        RandomNetwork.__init__(self, M, D=D, R=R, sigma_pop=sigma_pop, rho_pop = rho_pop, gamma_pop =gamma_pop, W_type = W_type, W_parameters = W_parameters, percentage_population_connections = percentage_population_connections, max_angle = max_angle)
        
        self.sigma = sigma_pop
        
        # Here, M is not really important, as it's inferred from the given possible orientations for the full code
        self.M = 0
    
    
    def assign_possible_orientations(self, possible_angles):
        '''
            Assign all the possible factorial representations
            
            network_representations:    R x K x M
        '''
        weight_representation = 1.0
        
        self.K = possible_angles.size
        self.possible_angles = possible_angles
        
        # Define the possible objects
        self.possible_objects_indices = np.array(cross([[x for x in np.arange(self.K)]]*self.R))
        self.possible_objects = np.array(cross([[x for x in self.possible_angles]]*self.R))
        
        # Each representation is a (K)^R matrix. From the outside though, it will be flattened
        self.M = int(self.K**self.R)
        
        # Construct the network representations
        # K x K x ... x M
        self.network_representations = np.zeros(flatten_list([[self.K]*self.R, [self.M]]))
        
        # Weights to convert the KxKx... indices into a flattened vector.
        # flattening_converter = (self.K*np.ones(self.R))**np.arange(self.R)[::-1]
        
        # Hard to get something valid for all R (would need to imbricate for loops further...), so just do it for R=2 and R=3
        cnt = 0
        for obj_ind in self.possible_objects_indices:
            # Build an automated index, from the obj_ind, and put a 1 in the flattened version.
            # self.network_representations[tuple(flatten_list([obj_ind, [np.dot(obj_ind, flattening_converter).astype(int)]]))] = 1
            
            # .... Now being less stupid and using a counter...
            self.network_representations[tuple(flatten_list([obj_ind, [cnt]]))] = weight_representation
            
            # Version 2, put a random sample instead...
            # self.network_representations[tuple(obj_ind)] = weight_representation*np.random.randn(self.M)
            
            cnt += 1
        
        self.W = None
        self.network_initialised = True
    
    
    def sample_network_response_indices(self, chosen_orientations):
        raise NotImplementedError()
    
    
    def sample_network_response(self, chosen_orientations, summed=False):
        '''
            Return the correct factorial code, corrupt it with some independent noise
        '''
        
        dims = chosen_orientations.ndim
        
        if dims == 1:
            # Assumes only a tuple of orientations, i.e. for different features
            assert chosen_orientations.size == self.R, 'Wrong number of features'
            
            # Search the closest factorial code to the given angles
            closest_object = np.argmin(np.abs(chosen_orientations - self.possible_angles[:, np.newaxis]), axis=0)
            
            # Return it with a big of noise on top
            response = self.network_representations[tuple(closest_object)] + self.sigma*np.random.randn(self.M)
        else:
            T = chosen_orientations.shape[0]
            response = np.zeros((T, self.M))
            
            # Find all the closest objects as well
            closest_objects = np.argmin(np.abs(chosen_orientations - self.possible_angles[:, np.newaxis, np.newaxis]), axis=0)
            
            for orientations_i in np.arange(T):
                response[orientations_i] = self.network_representations[tuple(closest_objects[orientations_i])] + self.sigma*np.random.randn(self.M)
            
        
        return response
        
    
    
    def get_network_features_combined(self, Z):
        '''
            Return the true object representation
        '''
        
        if Z.ndim == 1:
            closest_object = np.argmin(np.abs(Z - self.possible_angles[:, np.newaxis]), axis=0)
            
            # Return it with a big of noise on top
            sum_features = self.network_representations[tuple(closest_object)]
            
        elif Z.ndim == 2:
            (N, R) = Z.shape
            sum_features = np.zeros((N, self.M))
            
            closest_objects = np.argmin(np.abs(Z - self.possible_angles[:, np.newaxis, np.newaxis]), axis=0)
            
            for orientations_i in np.arange(N):
                sum_features[orientations_i] = self.network_representations[tuple(closest_objects[orientations_i])]
            
        else:
            raise ValueError('Wrong dimensionality for Z')
        
        return sum_features
        
    
    
    def get_network_features_combined_binary(self, Z):
        raise NotImplementedError()
    
    
    def get_popcode_response(self, theta, r):
        raise NotImplementedError()
    
    
    def plot_spread_full_representation(self):
        '''
            Compute all possibles "objects" (features combinations) representations, and plot them.
            Use PCA to reduce the dimensionality
        '''
        
        
        # Retrieve the representations
        all_objects_repr = np.reshape(self.network_representations, (self.K**self.R, self.M))
        
        # Plot them
        import mdp
        pca_node = mdp.nodes.PCANode(output_dim=0.9)
        all_objects_pc = pca_node(all_objects_repr)
        explained_variance = pca_node.get_explained_variance()
        
        import mpl_toolkits.mplot3d as p3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        colors_groups = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for k in np.arange(self.K):
            ax.scatter(all_objects_pc[k*self.K:(k+1)*self.K, 0], all_objects_pc[k*self.K:(k+1)*self.K,1], all_objects_pc[k*self.K:(k+1)*self.K,2], c=colors_groups[k%len(colors_groups)])
        
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        
        # Compute some statistics
        U, s, V = np.linalg.svd(all_objects_repr)
        rank_representation = np.sum(s > 1e-6)
        
        print "Rank: %d (%d max)" % (int(rank_representation), all_objects_repr.shape[0])
        
        
        return (explained_variance, rank_representation, all_objects_repr)
        
    
    
    @classmethod
    def create_instance_uniform(cls, K, D=50, R=1, sigma=0.2, max_angle=2.*np.pi):
        '''
            Create a RandomNetwork instance, and fill-in the K possible orientations, uniformly in [0, 2pi]
                If multiple features (R>1), uses the same parameters for everybody (wait till location needed)
        '''
        rn = RandomNetworkFactorialCode(M=1, D=D, R=R, sigma_pop=sigma, W_type='none', max_angle=max_angle)
        
        # Assign the angles
        # For this type of Network, will actually compute all the factorial encodings as well.
        possible_angles = np.linspace(-np.pi, np.pi, K, endpoint=False)
        rn.assign_possible_orientations(possible_angles)
        
        return rn
    


class RandomNetworkContinuous(RandomNetwork):
    def __init__(self, M, D=50, R=1, sigma_pop=0.6, rho_pop=0.5, gamma_pop=0.1, W_type='identity', W_parameters=[0.5], percentage_population_connections = 0.4, max_angle=2.*np.pi):
        
        RandomNetwork.__init__(self, M, D=D, R=R, sigma_pop=sigma_pop, rho_pop = rho_pop, gamma_pop =gamma_pop, W_type = W_type, W_parameters = W_parameters, percentage_population_connections = percentage_population_connections, max_angle = max_angle)
        
        self.covariance_network_combined = None
    
    
    def sample_network_response(self, chosen_orientations, summed=True):
        '''
            Get a random response for a/multiple orientation(s) from the population code,
            transform it through W and return that
            
            return: R x number_input_orientations x M
        '''
        
        chosen_orientations = chosen_orientations.T
        
        dim = chosen_orientations.shape
        
        if chosen_orientations.ndim > 1:
            net_samples = np.zeros((self.R, dim[1], self.M))
        else:
            if dim[0] == self.R:
                # Guess/Correct a weird bug/bad prototyping: if one gives a tuple, that should correspond to one angle per population code. Terrible fix, but the other usage (2 angles for the same population code) is obsolete.
                net_samples = np.zeros((self.R, self.M))
            else: 
                net_samples = np.zeros((self.R, dim[0], self.M))
        
        for r in np.arange(self.R):
            if np.size(dim) > 1 or dim[0] == self.R:
                # We have different orientations for the different population codes. It should be on the first dimension.
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(chosen_orientations[r]), self.W[r].T)
            else:
                net_samples[r] = np.dot(self.popcodes[r].sample_random_response(chosen_orientations), self.W[r].T)
        
        if summed:
            return np.sum(net_samples, axis=0)
        else:
            return net_samples
        
        
    
    def get_network_features_combined(self, Z):
        '''
            Compute \sum_r W_r mu(theta_r)
        '''
        if Z.ndim == 1:
            # Hopefully still fast enough...
            sum_features = np.dot(self.popcodes[0].mean_response(Z[0]), self.W[0].T) + np.dot(self.popcodes[1].mean_response(Z[1]), self.W[1].T)
        elif Z.ndim == 2:
            (N, R) = Z.shape
            sum_features = np.zeros((N, self.M))
            for r in np.arange(self.R):
                sum_features += np.dot(self.popcodes[r].mean_response(Z[:,r]), self.W[r].T)
        else:
            raise ValueError('Wrong dimensionality for Z')
        
        return sum_features
    
    
    def get_network_covariance_combined(self):
        '''
            Compute (and cache) the combined transformed covariance of the population codes
            i.e.:
                \sum_r W_r \Sigma_r W_r^T
        '''
        if self.covariance_network_combined is not None:
            return self.covariance_network_combined
        else:
            # First call, compute it.
            if self.R == 2:
                self.covariance_network_combined = np.dot(self.W[0], np.dot(self.popcodes[0].covariance, self.W[0].T)) + \
                                                np.dot(self.W[1], np.dot(self.popcodes[1].covariance, self.W[1].T))
            else:
                self.covariance_network_combined = np.zeros((self.M, self.M))
                for r in np.arange(self.R):
                    self.covariance_network_combined += np.dot(self.W[r], np.dot(self.popcodes[r].covariance, self.W[r].T))
                
            return self.covariance_network_combined
        
    
    def sample_network_response_indices(self, chosen_orientations):
        raise NotImplementedError()
    
    def get_network_features_combined_binary(self, Z):
        raise NotImplementedError()
    
    def plot_population_representation(self):
        raise NotImplementedError()
    
    
    @classmethod
    def create_instance_uniform(cls, K, M, D=50, R=1, sigma=0.2, rho=0.01, gamma=0.01, W_type='identity', W_parameters=[0.5], max_angle=2.*np.pi):
        '''
            Create a RandomNetwork instance, and fill-in the K possible orientations, uniformly in [0, 2pi]
                If multiple features (R>1), uses the same parameters for everybody (wait till location needed)
        '''
        rn = RandomNetworkContinuous(M, D=D, R=R, sigma_pop=sigma, rho_pop=rho, gamma_pop=gamma, W_type=W_type, W_parameters=W_parameters, max_angle=max_angle)
        
        # Used only for compatibility and data generation. Generate a few possible objects.
        #   Even though now during sampling, any angle is possible.
        possible_angles = np.linspace(-np.pi, np.pi, K, endpoint=False)
        rn.assign_possible_orientations(possible_angles)
        
        return rn
    
    


class RandomNetworkContinuousFactorialCode(RandomNetwork):
    '''
        Modified paradigm for this Network. Uses a factorial representation of features, and samples them using k-dimensional gaussian receptive fields.
            Randomness is in the distribution of orientations and radii of those gaussians.
    '''
    def __init__(self, 
                M,   D=50,    R=1, 
                sigma_pop=0.6, rho_pop=0.5, gamma_pop=0.1, 
                W_type='identity', W_parameters=[0.5], 
                percentage_population_connections = 0.4, max_angle=2.*np.pi,
                ):
        
        RandomNetwork.__init__(self, M, D=D, R=R, sigma_pop=sigma_pop, rho_pop = rho_pop, gamma_pop =gamma_pop, W_type = W_type, W_parameters = W_parameters, percentage_population_connections = percentage_population_connections, max_angle = max_angle)
        
        self.sigma = sigma_pop
        self.sigma_fact = 0.1
        
        
        self.neurons_preferred_stimulus = None
        self.neurons_sigma = None
        
        # Need to assign to each of the M neurons a preferred stimulus (tuple(orientation, color) for example)
        # By default, random
        self.assign_prefered_stimuli()
        self.assign_random_eigenvectors()
    
    
    def assign_prefered_stimuli(self, tiling_type='conjunctive', specified_neurons=None, reset=False):
        '''
            For all M factorial neurons, assign them a prefered stimulus tuple (e.g. (orientation, color) )
        '''
        
        if self.neurons_preferred_stimulus is None or reset:
            self.neurons_preferred_stimulus = np.zeros((self.M, self.R))
        
        if specified_neurons is None:
            specified_neurons = np.arange(self.M)
        
        if tiling_type == 'conjunctive':
            N_sqrt = np.floor(np.power(specified_neurons.size, 1./self.R))
            coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt)
            new_stim = np.array(cross(self.R*[coverage_1D.tolist()]))
            self.neurons_preferred_stimulus[specified_neurons[:new_stim.shape[0]]] = new_stim
        elif tiling_type == '2_features':
            N = np.round(specified_neurons.size/self.R)
            coverage_1D = np.linspace(-np.pi, np.pi, N)
            
            # Arbitrary put them along (x, 0) and (0, y) axes
            self.neurons_preferred_stimulus[specified_neurons[0:N], 0] = coverage_1D
            self.neurons_preferred_stimulus[specified_neurons[N:self.R*N], 1] = coverage_1D
        
    
    def assign_aligned_eigenvectors(self, scale=1.0, ratio=1.0, specified_neurons=None, reset=False):
        '''
            Each neuron has a gaussian receptive field, defined by its eigenvectors/eigenvalues (principal axes and scale)
            Uses the same eigenvalues for all of those
            
            input:
                ratio:  1/-1   : circular gaussian.
                        >1      : ellipse, width>height
                        <-1     : ellipse, width<height
                        [-1, 1] : not used
        '''
        
        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)
        
        if specified_neurons is None:
            specified_neurons = np.arange(self.M)
        
        assert ratio <= -1 or ratio >= 1, "respect my authority! Use ratio >= 1 or <=-1"
        
        if ratio>0:
            self.neurons_sigma[specified_neurons, 1] = ratio*scale
            self.neurons_sigma[specified_neurons, 0] = scale
        elif ratio <0:
            self.neurons_sigma[specified_neurons, 1] = scale
            self.neurons_sigma[specified_neurons, 0] = -ratio*scale
        
        
        # Update parameters
        for m in np.arange(specified_neurons.size):
            self.neurons_params[specified_neurons[m], 0] = np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]) + np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1])
            self.neurons_params[specified_neurons[m], 1] = -np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 0]) + np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 1])
            self.neurons_params[specified_neurons[m], 2] = np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]) + np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1])
        
    
    def assign_random_eigenvectors(self, scale=1.0, ratio_concentration=100.0, specified_neurons = None, reset=False):
        '''
            Each neuron has a gaussian receptive field, defined by its eigenvectors/eigenvalues (principal axes and scale)
            Get those randomly, from dirichlet ratios of a given scale
        '''
        
        if self.neurons_sigma is None or reset:
            self.neurons_sigma = np.zeros((self.M, self.R))
            self.neurons_angle = np.zeros(self.M)
            self.neurons_params = np.zeros((self.M, 3))
        
        if specified_neurons is None:
            specified_neurons = np.arange(self.M)
        
        # Sample eigenvalues
        rnd_ratio = np.random.dirichlet(np.ones(self.R)*ratio_concentration, size=specified_neurons.size)
        self.neurons_sigma[specified_neurons] = scale*rnd_ratio
        
        # Sample eigenvectors (only need to sample a rotation matrix)
        # TODO only for R=2 for now, should find a better way.
        # self.neurons_angle[specified_neurons] = np.random.random_integers(0,1, size=specified_neurons.size)*np.pi/2.
        self.neurons_angle[specified_neurons] = np.pi*np.random.random(size=specified_neurons.size)
        
        if self.R == 2:
            for m in np.arange(specified_neurons.size):
                # Simply shuffle the axes, no weird cosampling for now.
                # self.neurons_sigma[specified_neurons[m]] = np.roll(self.neurons_sigma[specified_neurons[m]], self.neurons_angle[specified_neurons[m]] == np.pi/2.)
                
                self.neurons_params[specified_neurons[m], 0] = np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]) + np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1])
                self.neurons_params[specified_neurons[m], 1] = -np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 0]) + np.sin(2.*self.neurons_angle[specified_neurons[m]])/(4.*self.neurons_sigma[specified_neurons[m], 1])
                self.neurons_params[specified_neurons[m], 2] = np.sin(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 0]) + np.cos(self.neurons_angle[specified_neurons[m]])**2./(2.*self.neurons_sigma[specified_neurons[m], 1])
        
    
    def plot_coverage_feature_space(self):
        '''
            Plot (R=2 only)
        '''
        assert self.R == 2, "only works for R=2"
        
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        
        ells = [Ellipse(xy=self.neurons_preferred_stimulus[m], width=self.neurons_sigma[m, 0], height=self.neurons_sigma[m, 1], angle=np.degrees(self.neurons_angle[m])) for m in xrange(self.M)]
        
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.5)
            e.set_facecolor(np.random.rand(3))
            e.set_transform(ax.transData)
        
        # ax.autoscale_view()
        ax.set_xlim(-1.3*np.pi, 1.3*np.pi)
        ax.set_ylim(-1.3*np.pi, 1.3*np.pi)
        
        ax.set_xlabel('Color')
        ax.set_ylabel('Orientation')
        
        plt.show()
        
    
    
    def get_population_response_simple(self, stimulus_input):
        '''
            Compute the response of the network.
            
            Could be optimized, for now only for one stimulus tuple
            
            # TODO Check if vectorisable
        '''
        normalisation = 1.
        
        dx = self.neurons_preferred_stimulus[:,0] - stimulus_input[0]
        dy = self.neurons_preferred_stimulus[:,1] - stimulus_input[1]
        
        
        if self.R == 2:
            return normalisation*np.exp(-self.neurons_params[:,0]*dx**2.0 - 2.*self.neurons_params[:,1]*dx*dy - self.neurons_params[:,2]*dy**2.0)
        elif self.R == 3:
            raise NotImplementedError('R=3 for factorial code...')
        else:
            # not unrolled
            raise NotImplementedError('R>3 for factorial code...')
        
    
    def instantiate_discrete_objects(self, possible_angles):
        '''
            Assign all the possible factorial representations
            
            network_representations:    R x K x M
        '''
        pass
    
    
    def get_network_features_combined(self, Z):
        '''
            Return the true object representation
        '''
        
        if Z.ndim == 1:
            closest_object = np.argmin(np.abs(Z - self.possible_angles[:, np.newaxis]), axis=0)
            
            # Return it with a big of noise on top
            sum_features = self.network_representations[tuple(closest_object)]
            
        elif Z.ndim == 2:
            (N, R) = Z.shape
            sum_features = np.zeros((N, self.M))
            
            closest_objects = np.argmin(np.abs(Z - self.possible_angles[:, np.newaxis, np.newaxis]), axis=0)
            
            for orientations_i in np.arange(N):
                sum_features[orientations_i] = self.network_representations[tuple(closest_objects[orientations_i])]
            
        else:
            raise ValueError('Wrong dimensionality for Z')
        
        return sum_features
    
    
    def sample_network_response(self, chosen_orientations):
        '''
            Get a random response for a/multiple orientation(s) from the population codes.
            Should then return the product, creating the factorial code.
            
            return: R x number_input_orientations x M
        '''
        
        chosen_orientations = chosen_orientations.T
        
        dim = chosen_orientations.shape
        
        if chosen_orientations.ndim > 1:
            net_samples = np.zeros((self.R, dim[1], self.D))
        else:
            net_samples = np.zeros((self.R, self.D))
        
        
        return net_samples
        
    
    
    @classmethod
    def create_instance_uniform(cls, K, M, D=50, R=1, sigma=0.2, rho=0.01, gamma=0.01, W_type='identity', W_parameters=[0.5], max_angle=2.*np.pi):
        '''
            Create a RandomNetworkContinuousFactorialCode instance.
                We need to cover a square/(hypercube of R dimensions) with RBF. We assume uniform coverage, and M units to assign around (M^{1/R} per dimension)(not good for R big, but here it's alright)
                For now, discrete case: K possible values in each dimension, K^2 "objects"
        '''
        rn = RandomNetworkContinuousFactorialCode(M, D=D, R=R, sigma_pop=sigma, rho_pop=rho, gamma_pop=gamma, W_type=W_type, W_parameters=W_parameters, max_angle=max_angle)
        
        possible_angles = np.linspace(-np.pi, np.pi, K, endpoint=False)
        rn.instantiate_discrete_objects(possible_angles)
        
        return rn
    
    


if __name__ == '__main__':
    K = 30
    D = 30
    R = 2
    M = int(17**R)
    
    # rn = RandomNetwork.create_instance_uniform(K, M, D=D, R=2, W_type='dirichlet', W_parameters=[20./(R*D), 0.1], sigma=0.3, gamma=0.002, rho=0.002)
    # rn = RandomNetworkFactorialCode.create_instance_uniform(K, D=D, R=R, sigma=0.1)
    rn = RandomNetworkContinuousFactorialCode.create_instance_uniform(K, M, D=D, R=R, sigma=0.1)
    # rn = RandomNetworkContinuous.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[5./(R*D), 10.0], sigma=0.2, gamma=0.003, rho=0.002)
    
    # Pure conjunctive code
    if True:
        rn.assign_random_eigenvectors(scale=0.7, ratio_concentration=100., reset=True)
        rn.plot_coverage_feature_space()
    
    # Pure feature code
    if False:
        rn.assign_prefered_stimuli(tiling_type='2_features', reset=True, specified_neurons = np.arange(M/4))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specified_neurons = np.arange(M/8), reset=True)
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specified_neurons = np.arange(M/8, M/4))
    
        rn.plot_coverage_feature_space()
    
    # Mix of two population
    if False:
        rn.assign_prefered_stimuli(tiling_type='conjunctive', reset=True, specified_neurons = np.arange(M/2))
        rn.assign_random_eigenvectors(scale=0.8, specified_neurons = np.arange(M/2), reset=True)
        rn.assign_prefered_stimuli(tiling_type='2_features', specified_neurons = np.arange(M/2, M))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=20.0, specified_neurons = np.arange(M/2, 3*M/4))
        rn.assign_aligned_eigenvectors(scale=0.3, ratio=-20.0, specified_neurons = np.arange(3*M/4, M))
    
        rn.plot_coverage_feature_space()
    
    # net_samples = rn.sample_network_response_indices(np.array([[5], [2]]))
    # plt.figure()
    # for r in np.arange(1):
        # plt.plot(np.linspace(0, np.pi, M), net_samples[r].T, 'g')
    # plt.autoscale(tight=True)
    #     
    # 
    # 
    # rn.plot_population_representation()
    # rn.plot_network_representation()
    # (explained_variance, rank_objects, all_objects_repr) = rn.plot_spread_full_representation()
    # plt.close('all')
    # print explained_variance
    
    # 
    # plt.figure()
    # plt.plot(rn.network_orientations)
    # #hinton(rn.W)
    
    # net_samples = rn.sample_network_response(np.array([0.0, 2.0]))
    # plt.plot(net_samples)
    
    # Do a LLE projection
    # k = 20 # nb of closest neighbors to consider
    #     
    #     lle_projected_data = mdp.nodes.LLENode(k, output_dim=2)(all_objects_repr)
    #     hlle_projected_data = mdp.nodes.HLLENode(k, output_dim=2)(all_objects_repr)
    #     
    #     plt.figure()
    #     plt.scatter(lle_projected_data[:,0], lle_projected_data[:,1], c=np.arange(lle_projected_data.shape[0]))
    #     plt.title('LLE projection of the objects')
    #     
    #     plt.figure()
    #     plt.scatter(hlle_projected_data[:,0], hlle_projected_data[:,1], c=np.arange(lle_projected_data.shape[0]))
    #     plt.title('HLE projection of the objects')
    #     
    #     plt.show()
    # plt.close('all')
    # 

