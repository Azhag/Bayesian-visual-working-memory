#!/usr/bin/env python
# encoding: utf-8
"""
utils.py

Created by Loic Matthey on 2011-06-16.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt
import numpy as np
import matplotlib.ticker as plttic
import scipy.io as sio
import scipy.optimize as spopt
import uuid
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator


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

    return (d[:,0]+d[:,1])**0.5


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
    for i in np.arange(x.size):
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


########################## FILE I/O #################################

def unique_filename(prefix=None, suffix=None, unique_id=None, return_id=False):
    """
    Get an unique filename with uuid4 random strings
    """
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

def numpy_2_mat(array, filename, arrayname):
    sio.savemat('%s.mat' % filename, {'%s' % arrayname: array})



########################## PLOTTING FUNCTIONS #################################


def plot_mean_std_area(x, y, std, ax_handle=None):
    if ax_handle is None:
        f = plt.figure()
        ax_handle = f.add_subplot(111)
    
    ax = ax_handle.plot(x, y)
    current_color = ax[-1].get_c()
    
    ax_handle.fill_between(x, y-std, y+std, facecolor=current_color, alpha=0.4,
                        label='1 sigma range')
    
    return ax_handle

def semilogy_mean_std_area(x, y, std, ax_handle=None):
    if ax_handle is None:
        f = plt.figure()
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
    
    for i in np.arange(nb_plots_sqrt):
        for j in np.arange(nb_plots_sqrt):
            try:
                subaxes[i, j].plot(x[nb_plots_sqrt*i+j], y[nb_plots_sqrt*i+j])
                subaxes[i, j].xaxis.set_major_locator(plttic.NullLocator())
                subaxes[i, j].yaxis.set_major_locator(plttic.NullLocator())
            except IndexError:
                subaxes[i, j].set_visible(False)
    
    return (f, subaxes)    


def histogram_angular_data(data, bins=20, in_degrees=False, title=None, norm=None):

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
    
    plt.figure()
    plt.bar(x, bar_heights, alpha=0.75, width=2.*bound_x/(bins-1), align='center')
    if title:
        plt.title(title)
    plt.xlim([x[0]*1.1, 1.1*x[-1]])


def pcolor_square_grid(data, nb_to_plot=-1):
    '''
        Construct a square grid of pcolor
        
        Uses the first dimension as number of subplots.
    '''
    if nb_to_plot < 0:
        nb_to_plot = data.shape[0]
    
    nb_plots_sqrt = np.ceil(np.sqrt(nb_to_plot)).astype(int)
    f, subaxes = plt.subplots(nb_plots_sqrt, nb_plots_sqrt)
    
    for i in np.arange(nb_plots_sqrt):
        for j in np.arange(nb_plots_sqrt):
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


def plot_powerlaw_fit(xdata, ydata, amp, index):
    '''
        Plot a powerlaw with some associated datapoints
    '''

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    plt.plot(xdata, ydata, 'k.')  # Data
    # plt.text(0.0, 0.0, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    # plt.text(0.0, 0.0, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    
    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.plot(xdata, ydata, 'k.')  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')



################################# FITTING ########################################

def fit_powerlaw(xdata, ydata, should_plot=False, debug=False):
    '''
        Fit a power law to the provided data.
        Actually fit to the mean of the provided data, if multiple columns given (axis 1)

        Look for next function in order to fit a powerlaw while taking std dev into account.
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

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: np.mean(y - fitfunc(p, x), axis=1)
    # errfunc = lambda p, x, y: (y - fitfunc(p, x))
    # errfunc_mse = lambda p, x, y: np.mean((y - fitfunc(p, x))**2.)

    # Initial parameters
    pinit = np.array([1.0, -1.0])
    out = spopt.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)
    # out = spopt.fmin(errfunc_mse, pinit, args=(logx, logy))

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = np.exp(pfinal[0])

    indexErr = np.sqrt( covar[0][0] )
    ampErr = np.sqrt( covar[1][1] ) * amp

    if debug:
        print pfinal
        print covar
        print 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr)
        print 'Index = %5.2f +/- %5.2f' % (index, indexErr)
    
    ##########
    # Plotting data
    ##########
    if should_plot:
        plot_powerlaw_fit(xdata, ydata, amp, index)
                

    return np.array([index, amp])
        




if __name__ == '__main__':
    pass

