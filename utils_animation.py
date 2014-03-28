#!/usr/bin/env python
# encoding: utf-8
"""
utils_animation.py

Created by Loic Matthey on 2014-03-25.
Copyright (c) 2014 Gatsby Unit. All rights reserved.
"""

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.animation as plt_anim


# switch interactive mode on
plt.ion()

########################## ANIMATE PLOTS #############################

def rotate_plot3d(ax, output_filename='', rotation=360, init_fct=None, elev=20., bitrate=-1, nb_frames=360, fps=30., min_duration=5, dpi=None):
    '''
        Take an existing 3D plot and rotate it.

        Save to a movie if output_filename is set
    '''

    # If desired FPS would imply a super short movie, change it
    target_fps = int(np.min((fps, nb_frames/float(min_duration))))

    def animate(i):
        ax.view_init(elev=elev, azim=i*float(rotation)/nb_frames)

    anim = plt_anim.FuncAnimation(ax.get_figure(), animate, init_func=init_fct, frames=nb_frames, interval=10)

    if output_filename:
        file_format = os.path.splitext(output_filename)[-1]
        writer = None

        if file_format == '.gif':
            writer = 'imagemagick'
        elif file_format == '':
            # if empty, assume error and add .mp4
            output_filename += '.mp4'

        anim.save(output_filename, fps=target_fps, bitrate=bitrate, writer=writer, dpi=dpi)

