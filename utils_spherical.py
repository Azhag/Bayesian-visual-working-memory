#!/usr/bin/env python
# encoding: utf-8
"""
utils_spherical.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np
import math

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
    point1_pos[point1_pos < 0.0] += 2.*np.pi
    point2_pos[point2_pos < 0.0] += 2.*np.pi

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

def create_3D_rotation_matrix_bis(axis, theta):
    '''
        Create a rotation matrix of theta around axis, using the Euler–Rodrigues formula
    '''
    axis = axis/math.sqrt(np.dot(axis,axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])



def gs_ortho(input_vect, ortho_target):
    output = input_vect - np.dot(ortho_target, input_vect)*ortho_target
    return output/np.linalg.norm(output)



