#!/usr/bin/env python
# encoding: utf-8
"""
random_network.py

Created by Loic Matthey on 2011-06-10.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np

from populationcode import *
from hinton_plot import *
from utils import *

class RandomNetwork:
    
    def __init__(self, M, D=50, R=1, sigma_pop=0.6, rho_pop=0.5, gamma_pop=0.1, W_type='identity', W_parameters=[0.5], percentage_population_connections = 0.4, max_angle=np.pi):
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
        self.number_connections = np.ceil(self.ratio_connections*mean_number_connections).astype(int)
        
        # Now connect neurons to features accordingly. Choose K_i_n features uniformly.
        for m in np.arange(self.M):
            for r in np.arange(self.R):
                indices = np.random.permutation(np.arange(self.D))[:self.number_connections[m, r]]
                self.W[r, m, indices] = sigma_W*np.random.randn(np.min((self.D, self.number_connections[m, r])))
        
    
    
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
    
    def sample_network_response(self, chosen_orientations):
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
        
        return explained_variance
        
    
    
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
    
    

class RandomNetworkContinuous(RandomNetwork):
    def __init__(self, M, D=50, R=1, sigma_pop=0.6, rho_pop=0.5, gamma_pop=0.1, W_type='identity', W_parameters=[0.5], percentage_population_connections = 0.4, max_angle=2.*np.pi):
        
        RandomNetwork.__init__(self, M, D=D, R=R, sigma_pop=sigma_pop, rho_pop = rho_pop, gamma_pop =gamma_pop, W_type = W_type, W_parameters = W_parameters, percentage_population_connections = percentage_population_connections, max_angle = max_angle)
        
        self.covariance_network_combined = None
        
    
    
    def sample_network_response(self, chosen_orientations):
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
    
    
    def get_popcode_response(self, Z, r):
        '''
            Return the output of one population code
        '''
        return np.dot(self.popcodes[r].mean_response(Z), self.W[r].T)
    
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
        possible_angles = np.linspace(0., max_angle, K, endpoint=False)
        rn.assign_possible_orientations(possible_angles)
        
        return rn
    
    

if __name__ == '__main__':
    K = 10
    M = 200
    D = 100
    R = 2
    
    rn = RandomNetwork.create_instance_uniform(K, M, D=D, R=2, W_type='identity', W_parameters=[0.1, 0.5], sigma=0.6)
    # rn = RandomNetwork.create_instance_uniform(K, M, D=D, R=2, W_type='identity', W_param=0.2)
    
    net_samples = rn.sample_network_response_indices(np.array([[5], [2]]))
    plt.figure()
    for r in np.arange(1):
        plt.plot(np.linspace(0, np.pi, M), net_samples[r].T, 'g')
    plt.autoscale(tight=True)
    #     
    # 
    # 
    # rn.plot_population_representation()
    # rn.plot_network_representation()
    # explained_variance = rn.plot_spread_full_representation()
    
    # print explained_variance
    
    # 
    # plt.figure()
    # plt.plot(rn.network_orientations)
    # #hinton(rn.W)
    # 
    plt.show()
    # 
