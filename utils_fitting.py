#!/usr/bin/env python
# encoding: utf-8
"""
utils_fitting.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

import scipy as sp
import scipy.optimize as spopt
import scipy.stats as spst

import matplotlib.pyplot as plt

from utils_plot import plot_powerlaw_fit, hist_angular_data


################################# FITTING ########################################

def fit_powerlaw(xdata, ydata, should_plot=False, debug=False):
    '''
        Fit a power law to the provided data.
        y = a x**p
        Actually fit to the mean of the provided data, if multiple columns given (axis 1)

        Look for next function in order to fit a powerlaw while taking std dev into account.

        returns (power p, amplitude a)
    '''

    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    #

    if ydata.ndim == 1:
        # We have a 1D array, need a flat 2D array...
        ydata = ydata[:, np.newaxis]

    if xdata.ndim == 1:
        # We need to tile the x values appropriately, to fit the size of y
        # (just because leastsq is not funny)
        xdata = np.tile(xdata, (ydata.shape[1], 1)).T

    logx = np.log(xdata.astype('float'))
    logy = np.log(ydata)

    if np.any(np.isnan(logy)):
        # Something went wrong, just stop here...
        return np.array([np.nan, np.nan])

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: np.mean(y - fitfunc(p, x), axis=1)

    # Initial parameters
    pinit = np.array([1.0, -1.0])
    out = spopt.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)
    # out = spopt.fmin(errfunc_mse, pinit, args=(logx, logy))

    pfinal = out[0]

    index = pfinal[1]
    amp = np.exp(pfinal[0])

    if debug:
        print pfinal
        print 'Ampli = %5.2f' % amp
        print 'Index = %5.2f' % index

    ##########
    # Plotting data
    ##########
    if should_plot:
        plot_powerlaw_fit(xdata, ydata, amp, index)


    return np.array([index, amp])


def fit_powerlaw_covariance(xdata, ydata, yerr = None, should_plot=True, debug=False):
    '''
        Implements http://www.scipy.org/Cookbook/FittingData

        We actually compute the standard deviation for each x-value, and use this
        when computing the fit.

    '''

    assert not (ydata.ndim == 1 and yerr is None), 'Use another function or provide yerr, this function requires the standard deviation'

    if ydata.ndim == 2:
        # Y data 2-dim, compute the standard errors, along the second dimension
        yerr = np.std(ydata, axis=1)
        ydata = np.mean(ydata, axis=1)

    logx = np.log(xdata.astype('float'))
    logy = np.log(ydata)
    logyerr = yerr/ydata

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    # Initial parameters
    pinit = np.array([1.0, -1.0])
    out = spopt.leastsq(errfunc, pinit, args=(logx, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = np.exp(pfinal[0])


    if debug:
        indexErr = np.sqrt( covar[0][0] )
        ampErr = np.sqrt( covar[1][1] ) * amp

        print pfinal
        print 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr)
        print 'Index = %5.2f +/- %5.2f' % (index, indexErr)

    ##########
    # Plotting data
    ##########
    if should_plot:
        plot_powerlaw_fit(xdata, ydata, amp, index, yerr=yerr)


    return np.array([index, amp])


def fit_gaussian(xdata, ydata, should_plot = True, return_fitted_data = True, normalise = True, debug = False):
    '''
        Fit a gaussian to the given points.
        Doesn't take samples! Only tries to fit a gaussian function onto some points.

        Can plot the result if desired
    '''

    mean_fit = np.sum(ydata*xdata)/np.sum(ydata)
    std_fit = np.sqrt(np.abs(np.sum((xdata - mean_fit)**2*ydata)/np.sum(ydata)))
    max_fit = ydata.max()

    fitted_data = spst.norm.pdf(xdata, mean_fit, std_fit)

    if debug:
        print "Mean: %.3f, Std: %.3f, Max: %.3f" % (mean_fit, std_fit, max_fit)

    if should_plot:
        plt.figure()
        if normalise:
            plt.plot(xdata, ydata/ydata.sum(), xdata, fitted_data/fitted_data.sum())
        else:
            plt.plot(xdata, ydata, xdata, fitted_data)
        plt.legend(['Data', 'Fit'])
        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([mean_fit, std_fit, max_fit]), fitted_data=fitted_data)
    else:
        return np.array([mean_fit, std_fit, max_fit])


def fit_gaussian_samples(samples, num_points=500, bound=np.pi, should_plot = True, return_fitted_data = True, normalise = True, debug = False):
    """
        Fit a 1D Gaussian on the samples provided.

        Plot the result if desired
    """

    mean_fit = np.mean(samples)
    std_fit = np.std(samples)

    # x = np.linspace(samples.min()*1.5, samples.max()*1.5, 1000)
    x = np.linspace(-bound, bound, num_points)
    dx = np.diff(x)[0]

    print mean_fit
    print std_fit

    fitted_data = spst.norm.pdf(x, mean_fit, std_fit)

    if debug:
        print "Mean: %.3f, Std: %.3f" % (mean_fit, std_fit)

    if should_plot:
        if normalise:
            hist_angular_data(samples, norm='max', bins=num_points)
            plt.plot(x, fitted_data/np.max(fitted_data), 'r')
        else:
            hist_angular_data(samples, bins=num_points)
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([mean_fit, std_fit]), fitted_data=fitted_data)
    else:
        return np.array([mean_fit, std_fit])


def fit_line(xdata, ydata, title='', should_plot=True, fignum=None):
    '''
        Fit a simple line, y = ax + b
    '''

    p = np.zeros(2)
    (p[0], p[1], r, p_val, stderr) = spst.linregress(xdata, ydata)

    if should_plot:
        plt.figure(fignum)
        plt.plot(xdata, ydata, 'k', xdata, sp.polyval(p, xdata), 'r')
        plt.legend(['Data', 'Fit'])
        plt.title(title)

    return p


def fit_gamma_samples(samples, num_points=500, bound=np.pi, fix_location=False, return_fitted_data=True, should_plot=True, normalise=True, debug=True):
    '''
        Fit a Gamma distribution on the samples, optionaly plotting the fit
    '''

    if fix_location:
        fit_alpha, fit_loc, fit_beta = spst.gamma.fit(samples, floc=1e-18)
    else:
        fit_alpha, fit_loc, fit_beta = spst.gamma.fit(samples)

    # x = np.linspace(samples.min()*1.5, samples.max()*1.5, 1000)
    x = np.linspace(-bound, bound, num_points)
    # dx = 2.*bound/(num_points-1.)

    fitted_data = spst.gamma.pdf(x, fit_alpha, fit_loc, fit_beta)

    if debug:
        print "Alpha: %.3f, Location: %.3f, Beta: %.3f" % (fit_alpha, fit_loc, fit_beta)

    if should_plot:
        if normalise:
            hist_angular_data(samples, norm='max', bins=num_points)
            plt.plot(x, fitted_data/np.max(fitted_data), 'r')
        else:
            hist_angular_data(samples, bins=num_points)
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([fit_alpha, fit_loc, fit_beta]), fitted_data=fitted_data)
    else:
        return np.array([fit_alpha, fit_loc, fit_beta])


def fit_beta_samples(samples, num_points=500, fix_location=False, return_fitted_data=True, should_plot=True, normalise=True, debug=True):
    '''
        Fit a Beta distribution on the samples, optionaly plotting the fit
    '''

    fit_a, fit_b, fit_loc, fit_scale = spst.beta.fit(samples, fscale=1.0)

    # x = np.linspace(samples.min()*1.5, samples.max()*1.5, 1000)
    x = np.linspace(0.0, 1.0, num_points)
    # dx = 2.*bound/(num_points-1.)

    fitted_data = spst.beta.pdf(x, fit_a, fit_b, fit_loc, fit_scale)

    if debug:
        print "A: %.3f, B: %.3f" % (fit_a, fit_b)

    if should_plot:
        if normalise:
            plt.hist(samples, bins=num_points, normed='density')
            plt.plot(x, fitted_data, 'r')
        else:
            plt.hist(samples, bins=num_points, normed='density')
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([fit_a, fit_b]), fitted_data=fitted_data)
    else:
        return np.array([fit_a, fit_b])


def fit_vonmises_samples(samples, num_points=500, return_fitted_data=True, should_plot=True, normalise=True, debug=True):
    '''
        Fit a Von Mises distribution on samples, optionaly plotting the fit.
    '''

    fit_kappa, fit_mu, fit_scale = spst.vonmises.fit(samples, fscale=1.0)

    # x = np.linspace(samples.min()*1.5, samples.max()*1.5, 1000)
    x = np.linspace(-np.pi, np.pi, num_points)
    # dx = 2.*np.pi/(num_points-1.)

    fitted_data = spst.vonmises.pdf(x, fit_kappa, loc=fit_mu, scale=fit_scale)

    if debug:
        print "mu: %.3f, kappa: %.3f" % (fit_mu, fit_kappa)

    if should_plot:
        if normalise:
            hist_angular_data(samples, norm='density', bins=num_points)
            plt.plot(x, fitted_data, 'r')
        else:
            hist_angular_data(samples, bins=num_points)
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([fit_mu, fit_kappa]), fitted_data=fitted_data, support=x)
    else:
        return np.array([fit_mu, fit_kappa])


def fit_gaussian_mixture(xdata, ydata, should_plot = True, return_fitted_data = True, normalise = True, debug = False):
    '''
        Fit a mixture of gaussians to the given points.
        Doesn't take samples! Only tries to fit a gaussian function onto some points.

        Can plot the result if desired
    '''

    if normalise:
        # Normalise
        ydata /= np.trapz(ydata, xdata)

    # define the fitting function
    gauss_fct = lambda x, mu, sigma: 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-(x-mu)**2./(2.*sigma**2.))
    mixt_fct = lambda p, x: (p[0]*gauss_fct(x, p[1], p[2]) + (1. - p[0])*gauss_fct(x, p[3], p[4]))

    # gauss_fct_der = lambda x, mu, sigma: -(x-mu)/(np.sqrt(2.*np.pi)*sigma**3.)*np.exp(-(x-mu)**2./(2.*sigma**2.))
    # mixt_fct_der = lambda p, x: (p[0]*gauss_fct_der(x, p[1], p[2]) + (1. - p[0])*gauss_fct_der(x, p[3], p[4]))
    # errfunc = lambda p, x, y: np.sum((y - mixt_fct(p, x))**2.)
    # errfunc_der = lambda p, x, y: 2*np.sum((y - mixt_fct(p, x))*mixt_fct_der(p, x))

    # def errfunc_jac(p, x, y):
    #     return 2.*np.sum((y - mixt_fct(p, x))*mixt_fct_jac(p, x), axis=-1)

    # def mixt_fct_jac(p, x):
    #     jac_p = np.zeros((p.size, x.size))
    #     jac_p[0] = gauss_fct(x, p[1], p[2]) - gauss_fct(x, p[3], p[4])
    #     jac_p[1] = p[0]*gauss_fct(x, p[1], p[2])*(x - p[1])/p[2]**2.
    #     jac_p[2] = p[0]*gauss_fct(x, p[1], p[2])*( (x - p[1])**2./p[2]**3. - p[2]**-1)
    #     jac_p[3] = (1. - p[0])*gauss_fct(x, p[3], p[4])*(x - p[3])/p[4]**2.
    #     jac_p[4] = (1. - p[0])*gauss_fct(x, p[3], p[4])*( (x - p[3])**2./p[4]**3. - p[4]**-1)
    #     return jac_p

    errfunc_leastsq = lambda p, x, y: y - mixt_fct(p, x)

    # Bounds
    # bounds = ((0.0, 1.0), (-np.pi, np.pi), (0.0, None), (-np.pi, np.pi), (0.0, None))

    # Initial parameters
    pinit = np.array([0.5, -0.5, 0.05, 0.5, 0.05])

    # Optimize
    out = spopt.leastsq(errfunc_leastsq, pinit, args=(xdata, ydata), full_output=1)
    # out = spopt.minimize(errfunc, pinit, method='L-BFGS-B', bounds=bounds, args=(xdata, ydata), options={'disp': True})
    # out = spopt.fmin(errfunc_mse, pinit, args=(logx, logy))

    pfinal = out[0]

    fitted_data = mixt_fct(pfinal, xdata)

    if debug:
        print pfinal

    ##########
    # Plotting data
    ##########

    if should_plot:
        plt.figure()
        if normalise:
            plt.plot(xdata, ydata/np.trapz(ydata, xdata), xdata, fitted_data/np.trapz(fitted_data, xdata))
        else:
            plt.plot(xdata, ydata, xdata, fitted_data)
        plt.legend(['Data', 'Fit'])
        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array(pfinal), fitted_data=fitted_data, support=xdata)
    else:
        return np.array(pfinal)


def fit_gaussian_mixture_fixedmeans(xdata, ydata, fixed_means=np.array([0.0, 0.0]), should_plot = True, return_fitted_data = True, normalise = True, debug = False):
    '''
        Fit a mixture of gaussians to the given points. Fixed means.
        Doesn't take samples! Only tries to fit a gaussian function onto some points.

        Can plot the result if desired
    '''

    if normalise:
        # Normalise
        ydata /= np.trapz(ydata, xdata)

    # define the fitting function
    gauss_fct = lambda x, mu, sigma: 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-(x-mu)**2./(2.*sigma**2.))
    mixt_fct = lambda p, x, mus: (p[0]*gauss_fct(x, mus[0], p[1]) + (1. - p[0])*gauss_fct(x, mus[1], p[2]))
    errfunc_leastsq = lambda p, x, y, mus: y - mixt_fct(p, x, mus)

    # Initial parameters
    pinit = np.array([0.5, 0.05, 0.05])

    # Optimize
    out = spopt.leastsq(errfunc_leastsq, pinit, args=(xdata, ydata, fixed_means), full_output=1)

    pfinal = out[0]

    fitted_data = mixt_fct(pfinal, xdata, fixed_means)

    if debug:
        print pfinal

    ##########
    # Plotting data
    ##########

    if should_plot:
        plt.figure()
        if normalise:
            plt.plot(xdata, ydata/np.trapz(ydata, xdata), xdata, fitted_data/np.trapz(fitted_data, xdata))
        else:
            plt.plot(xdata, ydata, xdata, fitted_data)
        plt.legend(['Data', 'Fit'])
        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array(pfinal), fitted_data=fitted_data, support=xdata)
    else:
        return np.array(pfinal)
