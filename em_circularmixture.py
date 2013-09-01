#!/usr/bin/env python
# encoding: utf-8
"""
EM_circularmixture.py

Created by Loic Matthey on 2013-09-20.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np
import scipy.special as scsp


def fit(responses, target_angle, nontarget_angles):
    '''
        Return maximum likelihood values for a mixture model, with:
            - 1 target Von Mises component
            - 1 circular uniform random component 
            - K nontarget Von Mises components
        Inputs in radian, in the -pi:pi range.
            - responses: Nx1
            - target_angle: 1x1
            - nontarget_angles Kx1

        Adapted from Matlab code by P. Bays
        Ref: Bays PM, Catalao RFG & Husain M. The precision of visual working 
            memory is set by allocation of a shared resource. Journal of Vision 9(10): 7, 1-11 (2009)
    '''

    # Clean inputs
    nontarget_angles = nontarget_angles[:, ~np.all(np.isnan(nontarget_angles), axis=0)]

    N = float(np.sum(~np.isnan(responses)))
    K = float(nontarget_angles.shape[1])
    max_iter = 1000
    epsilon = 1e-4

    # Initial parameters
    (kappa, mixt_target, mixt_random, mixt_nontargets, resp_ik) = initialise_parameters(responses.size, K)
    old_LL = -np.inf

    i = 0
    converged=False

    # Precompute some matrices
    error_to_target = wrap(target_angle - responses)
    error_to_nontargets = wrap(nontarget_angles - responses[:, np.newaxis])

    # EM loop
    # TODO handle NAN
    while i<max_iter and not converged:

        # E-step
        resp_ik[:, 0] = mixt_target * vonmisespdf(error_to_target, 0.0, kappa)
        resp_r = mixt_random/(2.*np.pi)
        if K>0:
            resp_ik[:, 1:] = mixt_nontargets/K  * vonmisespdf(error_to_nontargets, 0.0, kappa)
        W = np.sum(resp_ik, axis=1) + resp_r
        
        # Compute likelihood
        LL = np.nansum(np.log(W))
        dLL = LL - old_LL
        old_LL = LL

        if (np.abs(dLL) < epsilon):
            converged = True
            break

        # M-step
        mixt_target = np.nansum(resp_ik[:, 0]/W)/N
        mixt_nontargets = np.nansum(resp_ik[:, 1:]/W[:, np.newaxis])/N
        mixt_random = np.nansum(resp_r/W)/N

        # Update kappa, a bit harder. Could be done in complex angular coordinates I think.
        rw = resp_ik/W[:, np.newaxis]
        S = np.c_[np.sin(error_to_target), np.sin(error_to_nontargets)]
        C = np.c_[np.cos(error_to_target), np.cos(error_to_nontargets)]
        r1 = np.nansum(S*rw)
        r2 = np.nansum(C*rw)

        if np.abs(np.nansum(rw)) < 1e-10:
            kappa = 0
        else:
            R = (r1**2 + r2**2)**0.5/np.nansum(rw)
            kappa = A1inv(R)


        # Weird correction...
        if N <= 15:
            if kappa < 2:
                kappa = np.max([kappa - 2./(N*kappa), 0])
            else:
                kappa = kappa*(N-1)**3/(N**3 + N)

        i += 1

    if not converged:
        print "Warning, Em_circularmixture.fit() did not converge before ", max_iter, "iterations"
        # kappa = np.nan
        # mixt_target = np.nan
        # mixt_nontargets = np.nan
        # mixt_random = np.nan

    return (kappa, mixt_target, mixt_nontargets, mixt_random, rw)


def initialise_parameters(N, K):
    '''
        Initialises all parameters:
         - Von mises concentration
         - Mixture proportions, for Target, Nontarget and random
         - Responsabilities, per datapoint
    '''

    kappa = 5.
    mixt_nontargets = 0.2
    mixt_random = 0.1
    mixt_target = 1. - mixt_nontargets - mixt_random
    
    resp_ik = np.empty((N, K+1))

    return (kappa, mixt_target, mixt_random, mixt_nontargets, resp_ik)


def wrap(angles):
    '''
        Wrap angles in a -max_angle:max_angle space
    '''

    max_angle = np.pi

    angles = np.mod(angles + max_angle, 2*max_angle) - max_angle
    
    return angles


def vonmisespdf(x, mu, K):
    return np.exp(K*np.cos(x-mu)) / (2.*np.pi * scsp.i0(K))


def A1inv(R):
    if R >= 0.0 and R < 0.53:
        return 2*R + R**3 + (5.*R**5)/6.
    elif R < 0.85:
        return -0.4 + 1.39*R + 0.43/(1. - R)
    else:
        return 1./(R**3 - 4*R**2 + 3*R)



if __name__ == '__main__':
    # fit(responses, target_angle, nontarget_angles)
    pass

