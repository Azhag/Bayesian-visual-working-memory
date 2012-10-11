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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import git

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
            # (this is useful when the calling command is a specific launcher for a given experiment)
            calling_function = inspect.stack()[1][3]

        self.output_folder = output_folder
        self.label = label
        self.calling_function = calling_function
        self.unique_id = ''  # Keep the unique ID for further uses
        self.git_infos = None
        self.debug = debug

        # Initialize unique_filename
        self.create_filename()

        # Gather Git informations if relevant
        self.gather_git_informations()

        if debug:
            print "=== FileIO ready: %s ===" % self.filename


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

            self.unique_id = unique_id

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


    def gather_git_informations(self):
        '''
            Check if we are in a current Git repository.

            If so, will find the repository name and current commit number, to be saved with
            the data and in figures metadata.
        '''
        try:
            # Get the repository
            self.git_repo = git.Repo(os.getcwd())

            # Get the current branch
            branch_name = self.git_repo.active_branch.name

            # Get the current commit
            commit_num = self.git_repo.active_branch.commit.hexsha

            # Check if the repo is dirty (hence the commit is incorrect, may be important)
            repo_dirty = self.git_repo.is_dirty()

            # Save them up
            self.git_infos = dict(repo=str(self.git_repo), branch_name=branch_name, commit_num=commit_num, repo_dirty=repo_dirty)

            if self.debug:
                print "Found Git informations: %s" % self.git_infos

        except git.InvalidGitRepositoryError:
            # No Git repository here, just stop
            pass


    def numpy_2_mat(self, array, arrayname):
        '''
            If you really want to save .mat files, you can...
        '''
        sio.savemat('%s.mat' % self.filename, {'%s' % arrayname: array})



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
            Main function

            Takes a list of variables and save them in a numpy file.
            (could be changed to a cPickle or something)

            all_variables should usually just be "locals()". It's slightly ugly but it's easy.
        '''

        # Select only a subset of the variables to save
        dict_selected_vars = dict()
        for var in selected_variables:
            if var in all_variables:
                dict_selected_vars[var] = all_variables[var]

        # Add the Git informations if appropriate
        if self.git_infos:
            dict_selected_vars['git_infos'] = self.git_infos

        # Make sure to save the arguments if forgotten
        if 'args' in all_variables and not 'args' in dict_selected_vars:
            dict_selected_vars['args'] = all_variables['args']

        # Save them as a numpy array
        np.save(self.filename, dict_selected_vars)


    def save_current_figure(self, filename):
        '''
            Will save the current figure to the desired filename.

            the filename can contain some fields:
            {unique_id}

            The output directory will be automatically prepend.

            Adds Git informations into the PDF metadata if available.
        '''

        # Complete the filename if needs be.
        formatted_filename = os.path.join(self.output_folder, filename.format(unique_id=self.unique_id))

        ## Save the figure.

        extension = os.path.splitext(formatted_filename)[1]

        if extension == '.pdf':
            # We save in PDF, let's try to write some additional metadata
            pdf = PdfPages(formatted_filename)

            pdf.savefig()
            pdf_metadata = pdf.infodict()

            if self.git_infos:
                # If we are in a Git repository, add the informations about the current branch and commit
                # (it may not be properly commited, but close enough, check the next commit if dirty)
                pdf_metadata['Subject'] = "Created on branch {branch_name}, commit {commit_num} (dirty: {repo_dirty:d})".format(**self.git_infos)

            pdf.close()
        else:
            plt.savefig(formatted_filename)




if __name__ == '__main__':

    print "Testing..."

    # Some tests
    dataio = DataIO(output_folder='.', label='test_io', calling_function='')

    a = 12
    b = 31
    c = 'ba'
    d = 11

    # Now define which variables you want to save to disk
    selected_variables = ['a', 'b', 'c']

    # Save them
    dataio.save_variables(selected_variables, locals())

    print "Variables %s saved" % selected_variables

    # Create a figure and save it
    plt.figure()
    plt.plot(np.linspace(0, 10, 100.), np.sin(np.linspace(0, 10, 100.)))

    dataio.save_current_figure('test_figure-{unique_id}.pdf')

    print "Figure saved"


