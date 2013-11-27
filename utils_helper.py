#!/usr/bin/env python
# encoding: utf-8
"""
utils_helper.py

Created by Loic Matthey on 2013-09-08.
Copyright (c) 2013 Gatsby Unit. All rights reserved.
"""

import numpy as np

import datetime
import uuid
import pprint

import smtplib
from email.mime.text import MIMEText
import errno
from socket import error as socket_error

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


def say_finished(text='Work complete', additional_comment='', email_failed=True):
    '''
        Uses the text-to-speech capabilities to indicate when
         something is finished.

        If say doesn't work, try to send an email instead.
    '''
    try:
        import sh
        sh.say(text)
    except ImportError:
        if email_failed:
            try:
                email_finished(text=additional_comment, subject=text)
            except socket_error as serr:
                if serr.errno != errno.ECONNREFUSED:
                    # Not the error we are looking for, re-raise
                    raise serr
                print text, additional_comment


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


def remove_functions_dict(dict_input):
    '''
        Goes over the dictionary and removes the functions it finds.
    '''

    return dict((k, v) for (k, v) in dict_input.iteritems() if not is_function(v))

def argparse_2_dict(args):
    '''
        Take an Argparse.Namespace and converts it to a dictionary.
    '''
    try:
        # Convert Argparse.Namespace to dict
        all_parameters = vars(args)
    except TypeError:
        # Assume it's already done
        assert type(args) is dict, "args is neither Namespace nor dict, WHY?"
        all_parameters = args

    return all_parameters

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


def load_npy(filename, debug=True):
    '''
        Load and returns a .npy file. Plots its keys as well.
    '''

    data = np.load(filename).item()

    if debug:
        pprint.pprint(data.keys())

    return data

