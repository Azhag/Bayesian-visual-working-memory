#!/usr/bin/env python
# encoding: utf-8
"""
utils.py

Created by Loic Matthey on 2011-06-16.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.ticker as plttic
# import scipy.io as sio
import scipy.optimize as spopt
import scipy.stats as spst
import scipy.interpolate as spint
import scipy.special as scsp
import uuid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import LogNorm
import smtplib
from email.mime.text import MIMEText
import datetime

__maxexp__ = np.finfo('float').maxexp


############################ MATH ##################################

def powerlaw(x, amp, index):
    '''
        Computes the powerlaw:
        y = amp * (x**index)
    '''

    return amp * (x**index)

############################## DIRECTIONAL STATISTICS ################################

def mean_angles(angles):
    return np.angle(np.mean(np.exp(1j*angles), axis=0))


def wrap_angles(angles, bound=np.pi):
    '''
        Wrap angles in a [-bound, bound] space.

        For us: get the smallest angle between two responses
    '''

    # if np.isscalar(angles):
    #     while angles < -bound:
    #         angles += 2.*bound
    #     while angles > bound:
    #         angles -= 2.*bound
    # else:
    #     while np.any(angles < -bound):
    #         angles[angles < -bound] += 2.*bound
    #     while np.any(angles > bound):
    #         angles[angles > bound] -= 2.*bound
    
    angles = np.mod(angles + bound, 2*bound) - bound

    return angles

def kappa_to_stddev(kappa):
    '''
        Convert kappa to wrapped gaussian std dev

        std = 1 - I_1(kappa)/I_0(kappa)
    '''
    # return 1.0 - scsp.i1(kappa)/scsp.i0(kappa)
    return np.sqrt(-2.*np.log(scsp.i1e(kappa)/scsp.i0e(kappa)))

def stddev_to_kappa(stddev):
    '''
        Converts stddev to kappa

        No closed-form, does a line optimisation
    '''

    errfunc = lambda kappa, stddev: (np.exp(-0.5*stddev**2.) - scsp.i1e(kappa)/scsp.i0e(kappa))**2.
    kappa_init = 1.0
    kappa_opt = spopt.fmin(errfunc, kappa_init, args=(stddev, ), disp=False)

    return kappa_opt[0]


def test_stability_stddevtokappa(target_kappa=2.):
    '''
        Small test, shows how stable the inverse relationship between stddev and kappa is
    '''

    nb_iterations = 1000
    kappa_evolution = np.empty(nb_iterations)
    for i in xrange(nb_iterations):
        if i == 0:
            kappa_evolution[i] = stddev_to_kappa(kappa_to_stddev(target_kappa))
        else:
            kappa_evolution[i] = stddev_to_kappa(kappa_to_stddev(kappa_evolution[i-1]))


    print kappa_evolution[-1]
    
    plt.figure()
    plt.plot(kappa_evolution)
    plt.show()


############################ SPHERICAL/3D COORDINATES ##################################

def lbtoangle(b1, l1, b2, l2):
    if b1 < 0.:
        b1 = 360.+b1
    if l1 < 0.:
        l1 = 360.+l1
    if b2 < 0.:
        b2 = 360.+b2
    if l2 < 0.:
        l2 = 360.+l2
        
    p1 = np.cos(np.radians(l1-l2))
    p2 = np.cos(np.radians(b1-b2))
    p3 = np.cos(np.radians(b1+b2))

    return np.degrees(np.arccos(((p1*(p2+p3))+(p2-p3))/2.))

def dist_torus(points1, points2):
    # compute distance:
    # d = sqrt( min(|x1-x2|, 2pi - |x1-x2|)^2. +  min(|y1-y2|, 2pi - |y1-y2|)^2.)
    xx = np.abs(points1 - points2)
    d = (np.fmin(2.*np.pi - xx, xx))**2.

    return (d[:, 0]+d[:, 1])**0.5


def dist_sphere(point1, point2):
    point1_pos = point1.copy()
    point2_pos = point2.copy()
    point1_pos[point1_pos<0.0] += 2.*np.pi
    point2_pos[point2_pos<0.0] += 2.*np.pi

    p1 = np.cos(point1_pos[1]-point2_pos[1])
    p2 = np.cos(point1_pos[0]-point2_pos[0])
    p3 = np.cos(point1_pos[0]+point2_pos[0])

    return np.arccos((p1*(p2+p3) + p2-p3)/2.)


def dist_sphere_mat(points1, points2):
    '''
        Get distance between two sets of spherical coordinates (angle1, angle2)
        points1: Nx2
    '''
    points1_pos = points1.copy()
    points2_pos = points2.copy()
    # points1_pos[points1_pos<0.0] += 2.*np.pi
    # points2_pos[points2_pos<0.0] += 2.*np.pi

    p12 = np.cos(points1_pos - points2_pos)
    p3 = np.cos(points1_pos + points2_pos)[:, 0]

    return np.arccos((p12[:, 1]*(p12[:, 0]+p3) + p12[:, 0]-p3)/2.)


def spherical_to_vect(angles):
    output_vect = np.zeros(3)
    output_vect[0] = np.cos(angles[0])*np.sin(angles[1])
    output_vect[1] = np.sin(angles[0])*np.sin(angles[1])
    output_vect[2] = np.cos(angles[1])

    return output_vect


def spherical_to_vect_array(angles):
    output_vect = np.zeros((angles.shape[0], angles.shape[1]+1))

    output_vect[:, 0] = np.cos(angles[:, 0])*np.sin(angles[:, 1])
    output_vect[:, 1] = np.sin(angles[:, 0])*np.sin(angles[:, 1])
    output_vect[:, 2] = np.cos(angles[:, 1])

    return output_vect


def create_2D_rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def create_3D_rotation_around_vector(vector, angle):
    '''
        Performs a rotation around the given vector.
        From Wikipedia.
    '''

    return np.sin(angle)*np.array([[0, -vector[2], vector[1]], \
              [ vector[2], 0., -vector[0]], \
              [ -vector[1], vector[0], 0.]]) + \
           np.cos(angle)*np.eye(3) + \
           (1. - np.cos(angle))*np.outer(vector, vector)


def gs_ortho(input_vect, ortho_target):
    output = input_vect - np.dot(ortho_target, input_vect)*ortho_target
    return output/np.linalg.norm(output)


########################## TRICKS AND HELPER FUNCTIONS #################################

def flatten_list(ll):
    return [item for sublist in ll for item in sublist]


def list_2_tuple(arg):
    if (isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], list)):
        a = tuple([list_2_tuple(x) for x in arg])
    else:
        a = tuple(arg)
    
    return a

def cross(*args):
    '''
        Compute the cross product between multiple arrays
        Quite intensive, be careful...
    '''
    ans = [[]]
    for arg in args:
        if isinstance(arg[0], list) or isinstance(arg[0], tuple):
            for a in arg:
                ans = [x+[y] for x in ans for y in a]
        else:
            ans = [x+[y] for x in ans for y in arg]
    return ans

def strcat(*strings):
    return ''.join(strings)


def fast_dot_1D(x, y):
    out = 0
    for i in xrange(x.size):
        out += x[i]*y[i]
    return out

def fast_1d_norm(x):
    return np.sqrt(np.dot(x, x.conj()))

def array2string(array):
    # return np.array2string(array, suppress_small=True)
    if array.ndim == 2:
        return '  |  '.join([' '.join(str(k) for k in item) for item in array])
    elif array.ndim == 3:
        return '  |  '.join([', '.join([' '.join([str(it) for it in obj]) for obj in item]) for item in array])

def argmin_indices(array):
    return np.unravel_index(np.nanargmin(array), array.shape)

def argmax_indices(array):
    return np.unravel_index(np.nanargmax(array), array.shape)

def nanmean(array, axis=None):
    if not axis is None:
        return np.ma.masked_invalid(array).mean(axis=axis)
    else:
        return np.ma.masked_invalid(array).mean()

def nanmedian(array, axis=None):
    if not axis is None:
        return np.ma.extras.median(np.ma.masked_invalid(array), axis=axis)
    else:
        return np.ma.extras.median(np.ma.masked_invalid(array))


def nanstd(array, axis=None):
    if not axis is None:
        return np.ma.masked_invalid(array).std(axis=axis)
    else:
        return np.ma.masked_invalid(array).std()

def tril_set(array, vector_input, check_sizes=False):
    '''
        Sets the lower triangular part of array with vector_input

        Hope all sizes work...
    '''

    if check_sizes:
        num_elements = np.sum(np.fromfunction(lambda i,j: i>j, array.shape))
        assert vector_input.size == num_elements, "Wrong number of inputs, need %d" % num_elements

    array[np.fromfunction(lambda i,j: i>j, array.shape)] = vector_input


def triu_set(array, vector_input, check_sizes=False):
    '''
        Sets the upper triangular part of array with vector_input

        Hope all sizes work...
    '''

    if check_sizes:
        num_elements = np.sum(np.fromfunction(lambda i,j: i<j, array.shape))
        assert vector_input.size == num_elements, "Wrong number of inputs, need %d" % num_elements

    array[np.fromfunction(lambda i,j: i<j, array.shape)] = vector_input

    
def triu_2_tril(array):
    '''
        Copy the upper triangular part of an array into its lower triangular part
    '''
    
    array[np.fromfunction(lambda i,j: i>j, array.shape)] = array[np.fromfunction(lambda i,j: i<j, array.shape)]


def say_finished(text='Work complete', additional_comment='', email_failed=True):
    '''
        Uses the text-to-speech capabilities to indicate when
         something is finished.

        If say doesn't work, try to send an email instead.
    '''
    try:
        import sh
        sh.say(text)
    except Exception:
        if email_failed:
            email_finished(text=additional_comment, subject=text)


def email_finished(text='Work complete', to='loic.matthey@gatsby.ucl.ac.uk', subject='Work complete'):
    '''
        Sends an email to the given email address.
    '''

    sender = 'lmatthey+robot@gatsby.ucl.ac.uk'
    receivers = [to]
    
    msg = MIMEText(text + "\n%s" % str(datetime.datetime.now()))
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to

    s = smtplib.SMTP('localhost')
    s.sendmail(sender, receivers, msg.as_string())         
    s.quit()

    print "Finished email sent"


def is_function(arg):
    '''
        Checks if the argument looks like a function

        It looks like a function if it can be called.
    '''

    return hasattr(arg, '__call__')


########################## FILE I/O #################################

def unique_filename(prefix=None, suffix=None, unique_id=None, return_id=False):
    """
    Get an unique filename with uuid4 random strings
    """

    print 'DEPRECATED, use DataIO instead'

    fn = []
    if prefix:
        fn.extend([prefix, '-'])
    
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    
    fn.append(unique_id)
    
    if suffix:
        fn.extend(['.', suffix.lstrip('.')])
    
    if return_id:
        return [''.join(fn), unique_id]
    else:
        return ''.join(fn)


########################## PLOTTING FUNCTIONS #################################

def plot_multiple_mean_std_area(x, y, std, ax_handle=None, fignum=None):
    '''
        Plots multiple x-y data with standard error, on the same graph

        Will iterate over the first axis, has to...
    '''
    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
    
    if x.ndim == 1:
        # x should be extended, for convenience
        x = np.tile(x, (y.shape[0], 1))

    for curr_plt in xrange(x.shape[0]):
        ax_handle = plot_mean_std_area(x[curr_plt], y[curr_plt], std[curr_plt], ax_handle=ax_handle)

    return ax_handle


def plot_mean_std_area(x, y, std, ax_handle=None, fignum=None):
    '''
        Plot a given x-y data, with a transparent area for its standard deviation

        If ax_handle is given, plots on this figure.
    '''

    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
    
    ishold = plt.ishold()

    ax = ax_handle.plot(x, y)
    current_color = ax[-1].get_c()
    
    plt.hold(True)

    if np.any(std > 1e-6):
        ax_handle.fill_between(x, y-std, y+std, facecolor=current_color, alpha=0.4,
                        label='1 sigma range')
    
    ax_handle.get_figure().canvas.draw()

    plt.hold(ishold)

    return ax_handle


def plot_multiple_median_quantile_area(x, y=None, quantiles=None, axis=-1, ax_handle=None, fignum=None):
    '''
        Plots multiple x-y data with median and quantiles, on the same graph

        Will iterate over the first axis, has to...

        Assume that you give either the raw data in y, or the quantiles.
    '''
    
    assert (y is not None or quantiles is not None), "Give either y or quantiles"

    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
    
    if x.ndim == 1:
        # x should be extended, for convenience
        if y is not None:
            x = np.tile(x, (y.shape[0], 1))
        else:
            x = np.tile(x, (quantiles.shape[0], 1))

    for curr_plt in xrange(x.shape[0]):
        if y is not None:
            ax_handle = plot_median_quantile_area(x[curr_plt], y[curr_plt], quantiles=None, axis=axis, ax_handle=ax_handle)
        elif quantiles is not None:
            ax_handle = plot_median_quantile_area(x[curr_plt], quantiles=quantiles[curr_plt], axis=axis, ax_handle=ax_handle)


    return ax_handle


def plot_median_quantile_area(x, y=None, quantiles=None, axis=-1, ax_handle=None, fignum=None):
    """
        Plot the given x-y data, showing the median of y, and its 25 and 75 quantiles as a shaded area

        If ax_handle is given, plots on this figure
    """

    assert (y is not None or quantiles is not None), "Give either y or quantiles"

    if quantiles is None:
        quantiles = spst.mstats.mquantiles(y, axis=axis, prob=[0.25, 0.5, 0.75])

    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
    
    ax = ax_handle.plot(x, quantiles[..., 1])

    current_color = ax[-1].get_c()
    
    ax_handle.fill_between(x, quantiles[..., 0], quantiles[..., 2], facecolor=current_color, alpha=0.4,
                        label='quantile')
    
    ax_handle.get_figure().canvas.draw()

    return ax_handle


def semilogy_mean_std_area(x, y, std, ax_handle=None, fignum=None):
    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
    
    ax = ax_handle.semilogy(x, y)
    current_color = ax[-1].get_c()
    
    y_p = y+std
    y_m = y-std
    y_m[y_m < 0.0] = y[y_m < 0.0]

    ax_handle.fill_between(x, y_m, y_p, facecolor=current_color, alpha=0.4,
                        label='1 sigma range')
    
    return ax_handle


def plot_square_grid(x, y, nb_to_plot=-1):
    '''
        Construct a square grid of plots
        
        Uses the first dimension as number of subplots.
    '''
    if nb_to_plot < 0:
        nb_to_plot = y.shape[0]
    
    nb_plots_sqrt = np.round(np.sqrt(nb_to_plot)).astype(np.int32)
    f, subaxes = plt.subplots(nb_plots_sqrt, nb_plots_sqrt)
    
    for i in xrange(nb_plots_sqrt):
        for j in xrange(nb_plots_sqrt):
            try:
                subaxes[i, j].plot(x[nb_plots_sqrt*i+j], y[nb_plots_sqrt*i+j])
                subaxes[i, j].xaxis.set_major_locator(plttic.NullLocator())
                subaxes[i, j].yaxis.set_major_locator(plttic.NullLocator())
            except IndexError:
                subaxes[i, j].set_visible(False)
    
    return (f, subaxes)    


def histogram_angular_data(data, bins=20, in_degrees=False, title=None, norm=None, fignum=None):

    if in_degrees:
        bound_x = 180.
        data *= 180./np.pi
    else:
        bound_x = np.pi
    
    x = np.linspace(-bound_x, bound_x, bins)
    x_edges = x - bound_x/bins  # np.histogram wants the left-right boundaries...
    x_edges = np.r_[x_edges, -x_edges[0]]  # the rightmost boundary is the mirror of the leftmost one
    
    bar_heights, _ = np.histogram(data, bins=x_edges)

    if norm == 'max':
        bar_heights = bar_heights/np.max(bar_heights).astype('float')
    elif norm == 'sum':
        bar_heights = bar_heights/np.sum(bar_heights.astype('float'))
    elif norm == 'density':
        bar_heights, _ = np.histogram(data, bins=x_edges, density=True)
    
    plt.figure(fignum)
    plt.bar(x, bar_heights, alpha=0.75, width=2.*bound_x/(bins-1), align='center')
    if title:
        plt.title(title)
    plt.xlim([x[0]*1.1, 1.1*x[-1]])


def pcolor_2d_data(data, x=None, y=None, xlabel='', ylabel='', title='', colorbar=True, ax_handle=None, label_format="%.2f", fignum=None, interpolation='nearest', log_scale=False, ticks_interpolate=None):
    '''
        Plots a Pcolor-like 2d grid plot. Can give x and y arrays, which will provide ticks.

        Options:
         x                  array for x values
         y                  array for y values
         {x,y}_label        labels for axes
         log_scale          True for log scale of axis
         ticks_interpolate  If set, number of ticks to use instead of the x/y values directly
    '''

    if ax_handle is None:
        f = plt.figure(fignum)
        ax_handle = f.add_subplot(111)
        ax_handle.clear()

    if log_scale:
        im = ax_handle.imshow(data.T, interpolation=interpolation, origin='lower left', norm=LogNorm())
    else:
        im = ax_handle.imshow(data.T, interpolation=interpolation, origin='lower left')

    if not x is None:
        assert data.shape[0] == x.size, 'Wrong x dimension'

        if not ticks_interpolate is None:
            selected_ticks = np.array(np.linspace(0, x.size-1, ticks_interpolate), dtype=int)
            ax_handle.set_xticks(selected_ticks)
            ax_handle.set_xticklabels([label_format % x[tick_i] for tick_i in selected_ticks], rotation=90)
        else:
            ax_handle.set_xticks(np.arange(x.size))
            ax_handle.set_xticklabels([label_format % curr for curr in x], rotation=90)

    if not y is None:
        assert data.shape[1] == y.size, 'Wrong y dimension'

        if not ticks_interpolate is None:
            selected_ticks = np.array(np.linspace(0, y.size-1, ticks_interpolate), dtype=int)
            ax_handle.set_yticks(selected_ticks)
            ax_handle.set_yticklabels([label_format % y[tick_i] for tick_i in selected_ticks])
        else:
            ax_handle.set_yticks(np.arange(y.size))
            ax_handle.set_yticklabels([label_format % curr for curr in y])
    
    if xlabel:
        ax_handle.set_xlabel(xlabel)

    if ylabel:
        ax_handle.set_ylabel(ylabel)

    if colorbar:
        ax_handle.get_figure().colorbar(im)
    
    if title:
        ax_handle.set_title(title)
    
    ax_handle.axis('tight')

    # Change mouse over behaviour
    def report_pixel(x_mouse, y_mouse): 
        # Extract loglik at that position

        try:
            x_i = int(x_mouse)
            y_i = int(y_mouse)
            
            if x is not None:
                x_display = x[x_i]
            else:
                x_display = x_i
            
            if y is not None:
                y_display = y[y_i]
            else:
                y_display = y_i

            return "x=%.2f y=%.2f value=%.2f" % (x_display, y_display, data[x_i, y_i])
        except:
            return ""
    
    ax_handle.format_coord = report_pixel

    return ax_handle, im



def contourf_interpolate_data(all_points, data, xlabel='', ylabel='', title='', interpolation_numpoints=200, interpolation_method='linear', mask_when_nearest=True, contour_numlevels=20, show_scatter=True, show_colorbar=True, fignum=None):
    '''
        Take (x,y) and z tuples, construct an interpolation with them and plot them nicely.

        all_points: Nx2
        data:       Nx1

        mask_when_nearest: trick to hide points outside the convex hull of points even when using 'nearest' method
    '''

    # Construct the interpolation
    param1_space_int = np.linspace(all_points[:, 0].min(), all_points[:, 0].max(), interpolation_numpoints)
    param2_space_int = np.linspace(all_points[:, 1].min(), all_points[:, 1].max(), interpolation_numpoints)

    data_interpol = spint.griddata(all_points, data, (param1_space_int[None, :], param2_space_int[:, None]), method=interpolation_method)

    if interpolation_method == 'nearest' and mask_when_nearest:
        # Let's mask the points outside of the convex hull

        # The linear interpolation will have nan's on points outside of the convex hull of the all_points
        data_interpol_lin = spint.griddata(all_points, data, (param1_space_int[None, :], param2_space_int[:, None]), method='linear')

        # Mask
        data_interpol[np.isnan(data_interpol_lin)] = np.nan

    # Plot it
    f1 = plt.figure(fignum)
    ax1 = f1.add_subplot(111)
    cs = ax1.contourf(param1_space_int, param2_space_int, data_interpol, contour_numlevels)   # cmap=plt.cm.jet
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)

    if show_scatter:
        ax1.scatter(all_points[:, 0], all_points[:, 1], marker='o', c='b', s=5)
    
    ax1.set_xlim(param1_space_int.min(), param1_space_int.max())
    ax1.set_ylim(param2_space_int.min(), param2_space_int.max())
    
    if show_colorbar:
        f1.colorbar(cs)


def pcolor_square_grid(data, nb_to_plot=-1):
    '''
        Construct a square grid of pcolor
        
        Uses the first dimension as number of subplots.
    '''
    if nb_to_plot < 0:
        nb_to_plot = data.shape[0]
    
    nb_plots_sqrt = np.ceil(np.sqrt(nb_to_plot)).astype(int)
    f, subaxes = plt.subplots(nb_plots_sqrt, nb_plots_sqrt)
    
    for i in xrange(nb_plots_sqrt):
        for j in xrange(nb_plots_sqrt):
            try:
                subaxes[i, j].imshow(data[nb_plots_sqrt*i+j], interpolation='nearest')
                subaxes[i, j].xaxis.set_major_locator(plttic.NullLocator())
                subaxes[i, j].yaxis.set_major_locator(plttic.NullLocator())
            except IndexError:
                subaxes[i, j].set_visible(False)
                
    return (f, subaxes)

    

def plot_sphere(theta, gamma, Z, weight_deform=0.5, sphere_radius=1., try_mayavi=True):
    '''
        Plot a sphere, with the color set by Z.
            Also possible to deform the sphere according to Z, by putting a nonzero weight_deform.
    
        Need theta \in [0, 2pi] and gamma \in [0, pi]
    '''

    Z_norm = Z/Z.max()

    x = sphere_radius * np.outer(np.cos(theta), np.sin(gamma))*(1.+weight_deform*Z_norm)
    y = sphere_radius * np.outer(np.sin(theta), np.sin(gamma))*(1.+weight_deform*Z_norm)
    z = sphere_radius * np.outer(np.ones(np.size(theta)), np.cos(gamma))*(1.+weight_deform*Z_norm)

    # Have fun and try Mayavi for 3D plotting instead. Super faaaast.
    use_mayavi = False
    if try_mayavi:
        try:
            import mayavi.mlab as mplt

            use_mayavi = True
        except:
            pass
        
    if use_mayavi:
        mplt.figure()
        mplt.mesh(x, y, z, scalars=Z_norm)
        mplt.show()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, facecolors=cm.jet(Z_norm), rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
        
        # Colorbar
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(Z_norm)
        plt.colorbar(m)

        plt.show()


def plot_torus(theta, gamma, Z, weight_deform=0., torus_radius=5., tube_radius=3.0, try_mayavi=True, draw_colorbar=True):
    '''
        Plot a torus, with the color set by Z.
            Also possible to deform the sphere according to Z, by putting a nonzero weight_deform.
    
        Need theta \in [0, 2pi] and gamma \in [0, pi]
    '''

    
    Z_norm = Z/Z.max()

    X, Y = np.meshgrid(theta, gamma)
    x = (torus_radius+ tube_radius*np.cos(X)*(1.+weight_deform*Z_norm))*np.cos(Y)
    y = (torus_radius+ tube_radius*np.cos(X)*(1.+weight_deform*Z_norm))*np.sin(Y)
    z = tube_radius*np.sin(X)*(1.+weight_deform*Z_norm)
    
    use_mayavi = False
    if try_mayavi:
        try:
            import mayavi.mlab as mplt

            use_mayavi = True
        except:
            pass
        
    if use_mayavi:
        # mplt.figure(bgcolor=(0.7,0.7,0.7))
        mplt.figure(bgcolor=(1.0, 1.0, 1.0))
        mplt.mesh(x, y, z, scalars=Z_norm, vmin=0.0)
        
        if draw_colorbar:
            cb = mplt.colorbar(title='', orientation='vertical', label_fmt='%.2f', nb_labels=5)
        
        mplt.outline(color=(0., 0., 0.))
        mplt.show()

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, z, facecolors=cm.jet(Z_norm), rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
        
        # Colorbar
        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(Z_norm)

        if draw_colorbar:
            plt.colorbar(m)

        plt.show()


def plot_powerlaw_fit(xdata, ydata, amp, index, yerr=None, fignum=None):
    '''
        Plot a powerlaw with some associated datapoints
    '''

    plt.figure(fignum)
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    
    if yerr is None:
        plt.plot(xdata, ydata, 'k.')  # Data
    else:
        plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')

    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim((xdata.min()*0.9, xdata.max()*1.1))

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    
    if yerr is None:
        plt.plot(xdata, ydata, 'k.')  # Data
    else:
        plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')
    
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim((xdata.min()*0.9, xdata.max()*1.1))


def plot_fft2_power(data, s=None, axes=(-2, -1), fignum=None):
    '''
        Compute the 2D FFT and plot the 2D power spectrum.
    '''

    FS = np.fft.fft2(data, s=s, axes=axes)

    plt.figure(fignum)
    plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2.))
    plt.title('Power spectrum')
    plt.colorbar()


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
            histogram_angular_data(samples, norm='max', bins=num_points)
            plt.plot(x, fitted_data/np.max(fitted_data), 'r')
        else:
            histogram_angular_data(samples, bins=num_points)
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
    dx = 2.*bound/(num_points-1.)

    fitted_data = spst.gamma.pdf(x, fit_alpha, fit_loc, fit_beta)

    if debug:
        print "Alpha: %.3f, Location: %.3f, Beta: %.3f" % (fit_alpha, fit_loc, fit_beta)

    if should_plot:
        if normalise:
            histogram_angular_data(samples, norm='max', bins=num_points)
            plt.plot(x, fitted_data/np.max(fitted_data), 'r')
        else:
            histogram_angular_data(samples, bins=num_points)
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([fit_alpha, fit_loc, fit_beta]), fitted_data=fitted_data)
    else:
        return np.array([fit_alpha, fit_loc, fit_beta])


def fit_beta_samples(samples, num_points=500, bound=np.pi, fix_location=False, return_fitted_data=True, should_plot=True, normalise=True, debug=True):
    '''
        Fit a Beta distribution on the samples, optionaly plotting the fit
    '''

    fit_a, fit_b, fit_loc, fit_shape = spst.beta.fit(samples)

    # x = np.linspace(samples.min()*1.5, samples.max()*1.5, 1000)
    x = np.linspace(-bound, bound, num_points)
    dx = 2.*bound/(num_points-1.)

    fitted_data = spst.beta.pdf(x, fit_a, fit_b, fit_loc, fit_shape)

    if debug:
        print "A: %.3f, B: %.3f" % (fit_a, fit_b)

    if should_plot:
        if normalise:
            histogram_angular_data(samples, norm='max', bins=num_points)
            plt.plot(x, fitted_data/np.max(fitted_data), 'r')
        else:
            histogram_angular_data(samples, bins=num_points)
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
            histogram_angular_data(samples, norm='density', bins=num_points)
            plt.plot(x, fitted_data, 'r')
        else:
            histogram_angular_data(samples, bins=num_points)
            plt.plot(x, fitted_data, 'r')

        plt.show()

    if return_fitted_data:
        return dict(parameters=np.array([fit_mu, fit_kappa]), fitted_data=fitted_data, support=x)
    else:
        return np.array([fit_mu, fit_kappa])



if __name__ == '__main__':
    pass

