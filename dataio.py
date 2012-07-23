#!/usr/bin/env python
# encoding: utf-8
"""
data_io.py


Created by Loic Matthey on 2012-07-10
Copyright (c) 2011 . All rights reserved.
"""

import uuid
import scipy.io as sio
import os.path
import numpy as np
import inspect


class DataIO:
    '''
        Class handling data (from experiments) inputs and outputs.

        Provides basic outputting functionalities (could hold a dictionary for reloading the data as well later)
    '''

    def __init__(self, output_folder='./Data/', label='', calling_function=None, debug=True):
        '''
            Will use the provided output_folder.

            The filename will be created automatically, using the following format:
             label-calling_function-randomid
        '''

        if calling_function is None:
            # No calling function given, let's try and detect it!
            calling_function = inspect.stack()[1][3]

        self.output_folder = output_folder
        self.label = label
        self.calling_function = calling_function

        # Initialize unique_filename
        self.create_filename()

        if debug:
            print "FileIO ready: %s" % self.filename


    def create_filename(self):
        '''
            Create and store a filename.
        '''

        # Initialize unique_filename
        self.filename = os.path.join(self.output_folder, self.unique_filename(prefix=[self.label, self.calling_function]))        


    def unique_filename(self, prefix=None, suffix=None, extension=None, unique_id=None, return_id=False, separator='-'):
        """
            Get an unique filename with uuid4 random strings
        """

        fn = []
        if prefix:
            if isinstance(prefix, str):
                fn.append(prefix)
            else:
                fn.extend(prefix)

        if unique_id is None:
            unique_id = str(uuid.uuid4())

        fn.append(unique_id)

        if suffix:
            if isinstance(suffix, str):
                fn.append(suffix)
            else:
                fn.extend(suffix)

        # Concatenate everything, but remove empty strings already
        outstring = separator.join(filter(None, fn))

        if extension:
            outstring = '.'.join([outstring, extension.strip('.')])

        if return_id:
            return [outstring, unique_id]
        else:
            return outstring


    def numpy_2_mat(self, array, filename, arrayname):
        sio.savemat('%s.mat' % filename, {'%s' % arrayname: array})



    def new_filename(self, output_folder=None, label=None, calling_function=None):
        '''
            Modify the filename, get a new unique filename directly
        '''

        if output_folder:
            self.output_folder = output_folder
        if label:
            self.label = label
        if calling_function:
            self.calling_function = calling_function

        self.create_filename()


    def save_variables(self, selected_variables, all_variables):
        '''
            Takes a list of variables and save them in a numpy file.
            (could be changed to a cPickle or something)
        '''

        # Select only a subset of the variables to save
        dict_selected_vars = dict()
        for var in selected_variables:
            if var in all_variables:
                dict_selected_vars[var] = all_variables[var]

        # Save them as a numpy array
        np.save(self.filename, dict_selected_vars)



if __name__ == '__main__':
    # Some tests

    dataio = DataIO(label='test_io', calling_function='')

    print dataio.filename

    a= 12
    b = 31

    selected_variables = ['a', 'b', 'c']
    dataio.save_variables(selected_variables, locals())


