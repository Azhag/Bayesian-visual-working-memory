#!/usr/bin/env python
# encoding: utf-8
"""
launchers_profile.py


Created by Loic Matthey on 2012-10-10
Copyright (c) 2012 . All rights reserved.
"""

# import matplotlib.pyplot as plt

from datagenerator import *
from randomfactorialnetwork import *
from statisticsmeasurer import *
from slicesampler import *
from utils import *
from dataio import *
from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *



def profile_me(args):
    print "-------- Profiling ----------"
    
    import cProfile
    import pstats
    
    cProfile.runctx('profiling_run()', globals(), locals(), filename='profile_sampler.stats')
    
    stat = pstats.Stats('profile_sampler.stats')
    stat.strip_dirs().sort_stats('cumulative').print_stats()
    
    return {}


def profiling_run(args):
    
    N = 100
    T = 2
    # D = 64
    M = 128
    R = 2
    
    # random_network = RandomNetwork.create_instance_uniform(K, M, D=D, R=R, W_type='identity', W_parameters=[0.2, 0.7])
    #     data_gen = DataGenerator(N, T, random_network, type_Z='discrete', weighting_alpha=0.6, weight_prior='recency', sigma_y = 0.02)
    #     sampler = Sampler(data_gen, dirichlet_alpha=0.5/K, sigma_to_sample=True, sigma_alpha=3, sigma_beta=0.5)
    #     
    #     (log_y, log_z, log_joint) = sampler.run(10, verbose=True)
    
    N = args.N
    T = args.T
    # K = args.K
    M = args.M
    R = args.R
    # num_samples = args.num_samples
    # weighting_alpha = args.alpha
    
    # Build the random network
    sigma_y = 0.02
    sigma_x = 0.1
    time_weights_parameters = dict(weighting_alpha=0.9, weighting_beta=1.0, specific_weighting=0.1, weight_prior='uniform')
    cued_feature_time = T-1

    random_network = RandomFactorialNetwork.create_full_conjunctive(M, R=R)
    # random_network = RandomFactorialNetwork.create_full_features(M, R=R, sigma=sigma_x)
    # random_network = RandomFactorialNetwork.create_mixed(M, R=R, sigma=sigma_x, ratio_feature_conjunctive=0.2)
    
    # Construct the real dataset
    print "Building the database"
    data_gen = DataGeneratorRFN(N, T, random_network, sigma_y=sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time, sigma_x=sigma_x)
    
    # Measure the noise structure
    print "Measuring noise structure"
    data_gen_noise = DataGeneratorRFN(3000, T, random_network, sigma_y=sigma_y, time_weights_parameters=time_weights_parameters, cued_feature_time=cued_feature_time)
    stat_meas = StatisticsMeasurer(data_gen_noise)
    # stat_meas = StatisticsMeasurer(data_gen)
    
    print "Sampling..."
    sampler = Sampler(data_gen, theta_kappa=0.01, n_parameters=stat_meas.model_parameters, tc=cued_feature_time)

    sampler.plot_likelihood(sample=True, num_samples=1000)
