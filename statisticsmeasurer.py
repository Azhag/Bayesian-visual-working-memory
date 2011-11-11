#!/usr/bin/env python
# encoding: utf-8
"""
statisticsmeasurer.py

Created by Loic Matthey on 2011-08-02.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import numpy as np
from datagenerator import *
from randomnetwork import *
from utils import *
import pylab as plt
import matplotlib.mlab as mlab

class StatisticsMeasurer:
    def __init__(self, data_gen):
        self.data_gen = data_gen
        
        (self.N, self.T, self.M) = data_gen.all_Y.shape
        self.Y = data_gen.all_Y
        
        self.measure_moments()
        
        self.compute_collapsed_model_parameters()
        
        print "StatisticMeasurer has measured"
    
    
    def measure_moments(self):
        '''
            Compute the moments of Y for each time.
        '''
        
        self.means = np.zeros((self.T, self.M))
        self.covariances = np.zeros((self.T, self.M, self.M))
        
        for t in np.arange(self.T):
            self.means[t] = np.mean(self.Y[:,t,:], axis=0)
            self.covariances[t] = np.cov(self.Y[:,t,:].T)
        
    
    def plot_moments(self):
        '''
            Plot the fitted moments
        '''
        
        # Plot the means
        plt.figure()
        plt.plot(self.means.T)
        
        (f, subaxes) = pcolor_square_grid(self.covariances)
        
    
    def compute_collapsed_model_parameters(self):
        '''
            Compute m_t^s, m_t^e, \Sigma_t^s and \Sigma_t^e, for all possible times t_c.
            
                m_t^s : mean of the "starting" noise process
                m_t^e : mean of the "ending" noise process
                and their covariances
        '''
        
        model_means = np.zeros((3, self.T, self.M))
        model_covariances = np.zeros((3, self.T, self.M, self.M))
        
        # Mean and covariance of the starting noise is easy, just take the measured marginals of the previous time, transform them once.
        for t in np.arange(1, self.T):
            model_means[0, t] = np.dot(self.data_gen.time_weights[0][t], self.means[t-1])
            model_covariances[0, t] = np.dot(self.data_gen.time_weights[0][t], np.dot(self.covariances[t-1], self.data_gen.time_weights[0][t].T))
        
        
        # Mean and covariance of the ending noise requires a small mapping
        for t in np.arange(self.T-1):
            ATmtc = np.power(self.data_gen.time_weights[0][t], self.T-1-t)
            model_means[1, t] = self.means[self.T-1] - np.dot(ATmtc,  self.means[t])
            model_covariances[1, t] = self.covariances[self.T-1] - np.dot(ATmtc,  np.dot(self.covariances[t], ATmtc.T))
        
        # Measured means and covariances
        model_means[2] = self.means
        model_covariances[2] = self.covariances
            
        self.model_parameters = dict(means=model_means, covariances=model_covariances)
    
    
    def compute_plot_information_different_times(self):
        '''
            Compute and plot the information between the inferred memory at time t_c and the marginal at time t'
        '''
        
        # Assume that sigma_tc (noise of object at time tc, transformed by network already), is an average object (could do with popcode covariance instead).
        sigma_tc = self.covariances[0]
        sigma_y = self.data_gen.sigma_y
        
        # Same noise object for tprime
        sigma_tprime = sigma_tc
        
        # Get the inverses
        sigma_tc_inv = np.linalg.inv(sigma_tc)
        sigma_tprime_inv = sigma_tc_inv
        
        # Just notation things, for now uses the same A and B for all times
        B = self.data_gen.time_weights[1][0]
        A = self.data_gen.time_weights[0][0]
        
        # For now, assumes t' < tc
        mut_inf = np.zeros((self.T, self.T))
        for tp in np.arange(self.T):
            for tc in np.arange(self.T):
                if False:
                # if tp==tc:
                    # Special case, assume that I(xtc | yT, xtc) = h(xtc | yT) - 0
                    #   possibly wrong I think...
                    ATmtc = np.power(A, self.T-tc-1)
                    
                    P = self.covariances[tc]
                    APA_Sigtcp = self.covariances[self.T-1]
                    APA_Sigtcp_inv = np.linalg.inv(APA_Sigtcp)
                    
                    K_xx = P - np.dot(np.dot(P*ATmtc, APA_Sigtcp_inv), ATmtc*P)
                    
                    mut_inf[tp, tc] = 0.5*(np.linalg.slogdet(2.*np.pi*np.e*K_xx)[1])
                    # mut_inf[tp, tc] = 0.5*(np.linalg.slogdet(K_xx)[1]) # should be wrong
                    # mut_inf[tp, tc] = 0.5*np.log(np.linalg.det(2.*np.pi*np.e*K_xx))
                    
                    # Convert to bits
                    mut_inf[tp, tc] /= np.log(2.)
                
                else:
                    
                    ATmtc = np.power(A, self.T-tc-1)
                    ATmtp = np.power(A, self.T-tp-1)
                    
                    # Compute the variance along the chain, without the contribution at tc and tp
                    # simple version, assume same Sigma_i for all, just get the contribution of all times and remove tc and tp
                    sigma_T_nottptc = np.sum(np.power(A, 2*np.arange(self.T)))*(sigma_tc + sigma_y**2.*np.eye(self.M)) - np.sum(np.power(A, 2.*np.array([self.T-tp-1, self.T-tc-1])))*sigma_tc
                    sigma_T_nottptc_inv = np.linalg.inv(sigma_T_nottptc)
                    
                    # sigma_T_nottptc = np.zeros_like(sigma_T_nottptc)
                    #                     for t in np.arange(self.T):
                    #                         if t != tp and t != tc:
                    #                             sigma_T_nottptc += A**(self.T-1-t)*sigma_tc*A**(self.T-1-t)
                    #                     sigma_T_nottptc_inv = np.linalg.inv(sigma_T_nottptc)
                    #                     
                    
                    BATsig_T_nottctc = B*ATmtc*sigma_T_nottptc_inv*ATmtc*B
                    BATsig_T_nottctp = B*ATmtc*sigma_T_nottptc_inv*ATmtp*B
                    
                    # Compute the variance contribution on x_tc
                    sigma_tc_tilde_inv = sigma_tc_inv + BATsig_T_nottctc
                    sigma_tc_tilde = np.linalg.inv(sigma_tc_tilde_inv)
                    
                    K_inv_xx = sigma_tc_tilde_inv
                    K_inv_xy = BATsig_T_nottctp
                    K_inv_yy = sigma_tprime_inv + np.dot(BATsig_T_nottctp, np.dot(sigma_tc_tilde, BATsig_T_nottctp))
                    
                    K_inv = np.zeros((2*self.M, 2*self.M))
                    K_inv[0:self.M, 0:self.M] = K_inv_xx
                    K_inv[self.M:2*self.M, 0:self.M] = K_inv_xy
                    K_inv[0:self.M, self.M:2*self.M] = K_inv_xy
                    K_inv[self.M:2*self.M, self.M:2*self.M] = K_inv_yy
                    
                    K = np.linalg.inv(K_inv)
                    
                    # Now compute the Information
                    # I(x,y) = 0.5 log |Kxx||Kyy|/|K|
                    (Hxtc, Hxt, Hxtcxt) = (np.linalg.slogdet(K[0:self.M, 0:self.M])[1], np.linalg.slogdet(K[self.M:2*self.M, self.M:2*self.M])[1], np.linalg.slogdet(K)[1])
                    print "(tc: %d, tp: %d) > H(xtc) = %.5f, H(xt) = %.5f, H(xtc, xt) = %.5f, H(xtc | xt) = %.5f, condt < marg = %d" % (tc, tp, Hxtc, Hxt, Hxtcxt, Hxtcxt - Hxt, (Hxtcxt - Hxt) < Hxtc)
                    mut_inf[tp, tc] = 0.5*(np.linalg.slogdet(K[0:self.M, 0:self.M])[1] + np.linalg.slogdet(K[self.M:2*self.M, self.M:2*self.M])[1] - np.linalg.slogdet(K)[1])
                    # mut_inf[tp, tc] = (Hxtcxt)
                    
                    # Convert to bits
                    # mut_inf[tp, tc] /= np.log(2.)
                
            
            
        
        plt.figure()
        plt.imshow(mut_inf, interpolation='nearest', origin='lower')
        plt.xlabel('$x_{tc} | y_T$')
        plt.ylabel('$x_t''$')
        plt.colorbar()
        plt.title('Mutual information [nats]. A=%.2f' % A)
        
        return mut_inf
    



if __name__ == '__main__':
    K = 20
    D = 32
    M = 128
    R = 2
    T = 5
    N = 1000
    
    sigma_y = 0.05
    
    if True:
        
        # random_network = RandomNetworkContinuous.create_instance_uniform(K, M, D=D, R=R, W_type='dirichlet', W_parameters=[0.1, 0.5], sigma=0.2, gamma=0.003, rho=0.002)
        # random_network = RandomNetworkFactorialCode.create_instance_uniform(K, D=D, R=R, sigma=0.05)
        random_network = RandomFactorialNetwork(M, R=R, sigma=0.1)
        ratio_concentration = 2.
        random_network.assign_random_eigenvectors(scale_parameters=(10., 1/150.), ratio_parameters=(ratio_concentration, 4./(3.*ratio_concentration)), reset=True)
    

        # data_gen = DataGeneratorContinuous(N, T, random_network, sigma_y = sigma_y, time_weights_parameters = dict(weighting_alpha=0.9, weighting_beta = 1.0, specific_weighting = 0.1, weight_prior='uniform'))
        data_gen = DataGeneratorRFN(N, T, random_network, sigma_y = 0.02, sigma_x = 0.02, time_weights_parameters = dict(weighting_alpha=0.6, weighting_beta = 1.0, specific_weighting = 0.2, weight_prior='uniform'))

        
        stat_meas = StatisticsMeasurer(data_gen)
        
        n_means_start = stat_meas.model_parameters['means'][0]
        n_means_end = stat_meas.model_parameters['means'][1]
        n_means_measured = stat_meas.model_parameters['means'][2]
        n_covariances_start = stat_meas.model_parameters['covariances'][0]
        n_covariances_end = stat_meas.model_parameters['covariances'][1]
        n_covariances_measured = stat_meas.model_parameters['covariances'][2]
        
        # Some sanity checks...
        
        # Get covariance of one item
        if False:
            nb_samples = 100
            cov_x_meanangle = np.cov(np.reshape(random_network.popcodes[0].sample_random_response(np.linspace(-np.pi, np.pi, nb_samples), nb_samples=nb_samples), (nb_samples*nb_samples, D)).T)
            cov_one_item = np.dot(random_network.W[0], np.dot(cov_x_meanangle, random_network.W[0].T)) + np.dot(random_network.W[1], np.dot(cov_x_meanangle, random_network.W[1].T)) + sigma_y**2.*np.eye(M)
            
            # Check if covariance at time T is correct
            for t in np.arange(T):
                ATmtc = np.power(data_gen.time_weights[0][t], T-1-t)
                covariance_fixed_contrib_wrong = n_covariances_end[t] + ATmtc*ATmtc*(n_covariances_start[t]) + ATmtc*ATmtc*(random_network.get_network_covariance_combined() + sigma_y**2.*np.eye(M))
                covariance_fixed_contrib_correct = n_covariances_end[t] + ATmtc*ATmtc*(n_covariances_start[t]) + ATmtc*ATmtc*cov_one_item
                
                print "Mean squared error, correct covariance: %.5f" % np.mean(np.abs(covariance_fixed_contrib_correct - n_covariances_measured[-1]))
                print "Mean squared error, wrong covariance: %.5f" % np.mean(np.abs(covariance_fixed_contrib_wrong - n_covariances_measured[-1]))
        
        # stat_meas.compute_plot_information_different_times()
        
        # stat_meas.plot_moments()
        #     
        #     plt.figure()
        #     for t in np.arange(T)[::-1]:
        #         plt.hist(np.mean(stat_meas.Y[:,t,:] - np.mean(stat_meas.Y[:,t, :], axis=0), axis=1), bins=50)
        #     
        #     
    
    if False:
        
        # Small test
        sigma_x = 0.1
        sigma_x_2 = sigma_x**2.
        sigma_y = 0.02
        sigma_y_2 = sigma_y**2.
        mu_1 = 0.0
        mu_2 = 1.0
        mu_3 = 2.0
        T = 5
        alpha=0.8
        
        # construct y3
        # y_3 = mu_3 + sigma_x*np.random.randn() + alpha*(mu_2 + sigma_x*np.random.randn() + alpha*(mu_1 + sigma_x*np.random.randn() + sigma_y*np.random.randn()))
        
        # sigma_y3post = sigma_y_2 + sigma_x_2 + alpha**2.*(sigma_y_2 + sigma_x_2) + alpha**4.*sigma_y_2
        #         
        #         sigma_x1_y3 = (sigma_x**-2. + alpha**4.*sigma_y3post**-1.)**-1.
        #         
        #         mu_1_possible = np.linspace(0.0, 3.0, 100.)
        #         
        #         # mu_x1_y3 = sigma_x_2*alpha**2.*(alpha**4.*sigma_x_2 + sigma_y3post)**-1.*(y_3 - mu_3 - alpha*mu_2 - alpha**2.*mu_1) + mu_1
        #         mu_x1_y3 = sigma_x_2*alpha**2.*(alpha**4.*sigma_x_2 + sigma_y3post)**-1.*(y_3 - mu_3 - alpha*mu_2 - alpha**2.*mu_1_possible) + mu_1_possible
        #         
        #         x = np.linspace(-3, 3., 1000)
        #         like_out = -0.5*(np.tile(mu_x1_y3, (x.size, 1)).T - x)**2./sigma_x1_y3
        #         
        #         f = plt.figure()
        #         # plt.plot(x, like_out)
        #         im = plt.imshow(like_out, origin='lower')
        #         # im.set_extent((0.0, 3.0, -3., 3.))
        #         im.set_interpolation('nearest')
        #         f.colorbar(im)
        #         
        
        K_inv = np.zeros((2,2))
        
        mut_inf = np.zeros((T, T))
        
        for tc in np.arange(T):
            for tp in np.arange(T):
                sigma_yTpost = np.sum(np.power(alpha, 2*np.arange(T)))*(sigma_y_2+sigma_x_2) - sigma_x_2*np.sum(np.power(alpha, 2.*np.array([T-1-tc, T-1-tp])))
                sigma_yTpost_inv = sigma_yTpost**-1.
                
                sigma_xtc_yT_inv = (sigma_x**-2. + alpha**(2.*(T-tc-1.))*sigma_yTpost_inv)
                sigma_xtc_yT = sigma_xtc_yT_inv**-1.
                
                E = alpha**(T-tc-1+T-tp-1)*sigma_xtc_yT_inv
                
                sigma_xtp = sigma_x**-2. + E*E*sigma_xtc_yT
                                
                K_inv[0,0] = sigma_xtc_yT_inv
                K_inv[1,0] = E
                K_inv[0,1] = E
                K_inv[1,1] = sigma_xtp
                
                K = np.linalg.inv(K_inv)
                
                mut_inf[tp, tc] = 0.5*(np.log(K[0,0]) + np.log(K[1,1]) - np.linalg.slogdet(K)[1])
        
        
        plt.imshow(mut_inf, origin='lower', interpolation='nearest')
    
    if False:
        alpha = 0.7
        beta = 1.
        sigma_y = 0.05
        sigma_y_2 = sigma_y**2.
        T = 3
        M = 10.
        sigma_xy = sigma_y/M
        sigma_xy_2 = sigma_xy**2.
        
        # Generate data
        x_possible = np.eye(M)
        x_mean = np.mean(x_possible, axis=0)
        
        cov_xy = np.cov(x_possible)
        
        chosen_x = np.random.permutation(np.arange(M))[:T]
        all_y = np.zeros((T, M))
        yT = beta*x_possible[chosen_x[0]] + sigma_y*np.random.randn(M)
        all_y[0] = yT
        for t in np.arange(1, T):
            yT *= alpha
            yT += beta*x_possible[chosen_x[t]] + sigma_y*np.random.randn(M)
            all_y[t] = yT
        
        # Get likelihood of different patterns
        lik_resp_uncorr = np.zeros((T, M))
        lik_resp = np.zeros((T, M))
        lik_ytc = np.zeros((T, M))
        mu_yT_xtc = np.zeros((T, M, M))
        mu_yT_xtc_uncorr = np.zeros((T, M, M))
        invsigma_yT_xtc = np.zeros((T, M, M))
        
        for tc in np.arange(1, T+1):
            
            # Likelihood of yT
            # invsigma_yT_xtc = (alpha**(2.*(T-tc))*sigma_y_2 + alpha**(2*(T-tc+1))*(1-alpha**(2*tc-2))/(1-alpha**2.)*sigma_xy_2 + (1-alpha**(2*(T-tc)))/(1-alpha**2.)*sigma_xy_2)**-1.
            invsigma_yT_xtc[tc-1] = np.linalg.inv(alpha**(2.*(T-tc))*sigma_y_2*np.eye(M) + alpha**(2*(T-tc+1))*(1-alpha**(2*tc-2))/(1-alpha**2.)*cov_xy + (1-alpha**(2*(T-tc)))/(1-alpha**2.)*cov_xy)
            
            # Likelihood of ytc
            # Linv = (1-alpha**(2*(T-tc)))/(1-alpha**2.)*sigma_xy_2
            # L = Linv**-1.
            # Lambdainv = sigma_y_2 + alpha**2.*(1-alpha**(2*tc-2))/(1-alpha**2.)*sigma_xy_2
            # Lambda = Lambdainv**-1.
            # Sigmainv = Lambda + alpha**(2*(T-tc))*L
            # Sigma = Sigmainv**-1.
            
            if tc < T:
                Linv = (1-alpha**(2*(T-tc)))/(1-alpha**2.)*cov_xy
                L = np.linalg.inv(Linv)
                Lambdainv = sigma_y_2*np.eye(M) + alpha**2.*(1-alpha**(2*tc-2))/(1-alpha**2.)*cov_xy
                Lambda = np.linalg.inv(Lambdainv)
                Sigmainv = Lambda + alpha**(2*(T-tc))*L
                Sigma = np.linalg.inv(Sigmainv)
            
            
            for pat in np.arange(M):
                mu_yT_xtc[tc-1, pat] = alpha**(T-tc)*beta*x_possible[pat] + alpha**(T-tc+1)*(1-alpha**(tc-1))/(1-alpha)*beta*(x_mean - 1/M*x_possible[pat]) + (1-alpha**(T-tc))/(1-alpha)*beta*(x_mean - 1/M*x_possible[pat])
                
                mu_yT_xtc_uncorr[tc-1, pat] = alpha**(T-tc)*beta*x_possible[pat] + alpha**(T-tc+1)*(1-alpha**(tc-1))/(1-alpha)*beta*x_mean + (1-alpha**(T-tc))/(1-alpha)*beta*x_mean
                
                # mu_ytc_yTxtc = Sigma*( alpha**(T-tc)*L*(yT - (1-alpha**(T-tc))/(1-alpha)*beta*x_mean )  + Lambda*(beta*x_possible[pat] + alpha*(1-alpha**(tc-1))/(1-alpha)*beta*x_mean ) )
                if tc< T:
                    mu_ytc_yTxtc = np.dot(Sigma, ( alpha**(T-tc)*np.dot(L, (yT - (1-alpha**(T-tc))/(1-alpha)*beta*x_mean))  + np.dot(Lambda, (beta*x_possible[pat] + alpha*(1-alpha**(tc-1))/(1-alpha)*beta*x_mean )) ))
                
                lik_resp_uncorr[tc-1, pat] = -0.5*np.dot(yT - mu_yT_xtc_uncorr[tc-1, pat], np.dot(invsigma_yT_xtc[tc-1], yT - mu_yT_xtc_uncorr[tc-1, pat]))
                lik_resp[tc-1, pat] = -0.5*np.dot(yT - mu_yT_xtc[tc-1, pat], np.dot(invsigma_yT_xtc[tc-1], yT - mu_yT_xtc[tc-1, pat]))
                if tc<T:
                    # lik_ytc[tc-1, pat] = -0.5*np.dot(all_y[tc-1] - mu_ytc_yTxtc, np.dot(Sigmainv, all_y[tc-1] - mu_ytc_yTxtc))
                    lik_ytc[tc-1, pat] = -0.5*np.dot(all_y[tc-1] - mu_ytc_yTxtc, np.dot(np.eye(M), all_y[tc-1] - mu_ytc_yTxtc))
            
            lik_ytc[tc-1] -= np.mean(lik_ytc[tc-1])
        
        
        # TODO ERROR HERE, we add a contribution when we shouldn't => this biases everything towards the "end magnitude"
        # TODO Actually not the problem...
    
    
    plt.show()

     