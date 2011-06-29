#!/usr/bin/env python
# encoding: utf-8
"""
utils.py

Created by Loic Matthey on 2011-06-16.
Copyright (c) 2011 Gatsby Unit. All rights reserved.
"""

import pylab as plt

def cross(*args):
    ans = [[]]
    for arg in args:
        if isinstance(arg[0], list) or isinstance(arg[0], tuple):
            for a in arg:
                ans = [x+[y] for x in ans for y in a]
        else:
            ans = [x+[y] for x in ans for y in arg]
    return ans


def plot_std_area(x, y, std, ax_handle=None):
    if ax_handle is None:
        f = plt.figure()
        ax_handle = f.add_subplot(111)
    
    ax = ax_handle.plot(x, y)
    current_color = ax[-1].get_c()
    
    ax_handle.fill_between(x, y-std, y+std, facecolor=current_color, alpha=0.4,
                        label='1 sigma range')
    
    return ax_handle

def array2string(self, array):
    # return np.array2string(array, suppress_small=True)
    return '  |  '.join([' '.join(str(k) for k in item) for item in array])


if __name__ == '__main__':
    pass