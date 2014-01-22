#!/usr/bin/env python
# encoding: utf-8
"""
jobwrapper.py

Created by Loic Matthey on 2014-01-17
Copyright (c) 2014 . All rights reserved.
"""

import numpy as np
import hashlib
import json
import os

import resultcomputation
import experimentlauncher
import utils


class JobWrapper(object):
    """
        Class wrapping an Experiment.

        An Experiment is defined by its dictionary of parameters, to be fed to an ExperimentLauncher.
        Implements a compute() function that runs the experiment.
        Uses the compute_result() function, which is offloaded to another class (like Aggregators for Heiko, but simpler).

        Could support pickling for easy storage/resubmit.
    """

    def __init__(self, experiment_parameters, session_id='', debug=True):

        self.experiment_parameters = experiment_parameters
        self.job_name = self.create_unique_job_name(session_id)
        self.debug = debug
        self.result = None
        self.job_state = 'idle'

        if experiment_parameters.get('result_computation', '') != '':
            self.result_computation = resultcomputation.ResultComputation(experiment_parameters['result_computation'])

            self.result_filename = self.create_result_filename()
        else:
            self.result_computation = None
            self.result_filename = None

        if self.debug:
            print "JobWrapper initialised\n Name: %s" % (self.job_name)

            if self.result_computation is not None:
                print " Result aggregator: %s, %s" % (self.result_computation, self.result_filename)


    def create_unique_job_name(self, session_id=''):
        '''
            Generate a unique string from the parameters of the experiment.

            If the parameters are the same, should be the same job anyway.
        '''

        return session_id + hashlib.md5(json.dumps(self.experiment_parameters, sort_keys=True)).hexdigest()


    def flag_job_submitted(self):
        '''
            Just update job status when it is submitted onto PBS
        '''
        self.job_state = 'submitted'


    def compute(self):
        '''
            Instantiate an ExperimentLauncher and run it.

            Then run a compute_result() function, that should compute some metric from the result_* arrays we pass to it (actually pass all variables from the ExperimentLauncher).
                These functions could be specific to launchers_*, but let's say we put them in their own class
        '''

        # Instantiate ExperimentLauncher and run it.
        #  this will create some output arrays, that will be accessible later on if desired.
        print "--- Running job %s ---" % self.job_name
        experiment_launcher = experimentlauncher.ExperimentLauncher(run=True, arguments_dict=self.experiment_parameters)
        print "--- job completed ---"

        # Compute metric result out of the results from the computation if desired
        if self.result_computation is not None:
            if self.debug:
                print "Running %s" % self.result_computation

            # Compute the result
            self.result = self.result_computation.compute_result(experiment_launcher.all_vars)

            # Store it
            self.store_result()

        # Set completed flag
        self.job_state = 'completed'

        return experiment_launcher.all_vars


    ############
    ### Handle results

    def create_result_filename(self, prefix='result_sync_job_'):
        '''
            Create a filename from the unique job name
        '''

        return os.path.join(self.experiment_parameters['output_directory'], prefix + self.job_name + '.npy')


    def store_result(self):
        '''
            Store the result into a file.

            This file should have a specific name format, as it is awaited upon by this Job (but not the instance running on PBS).
        '''

        # dict_output_result = dict(result=self.result)

        np.save(self.result_filename, self.result)


    def check_completed(self):
        '''
            Keep probing for a specifically named file (basically the result outputs).

            Once it exists, it means our sister JobWrapper PBS job has finished working.
        '''

        if self.result_computation is not None and utils.file_exists_new_shell(self.result_filename):
            self.job_state = 'completed'
            return True
        else:
            return False


    def reload_result(self):
        '''
            Reloads result of computation from a specifically formatted file.
        '''

        self.result = np.load(self.result_filename).item()


    def get_result(self):
        '''
            Return the result, if the computation has finished.
            If not, return None.
        '''

        if self.result_computation is not None and self.job_state == 'completed' and self.result is None:
            # Reload the result from file
            self.reload_result()

        return self.result


def test_job_wrapper():
    '''
        Basic test
    '''

    experiment_parameters = dict(code_type='mixed',
                                               output_directory='.',
                                               ratio_conj=0.5,
                                               M=100,
                                               sigmax=0.1,
                                               N=10,
                                               T=2,
                                               sigmay=0.0001,
                                               inference_method='none',
                                               autoset_parameters=None,
                                               result_computation='random'
                                               )

    # Create job
    job = JobWrapper(experiment_parameters)

    # Check some stuff
    print "Job name:", job.job_name
    print "Result:", job.get_result()
    print "Is completed?", job.check_completed()

    # Create a sister job and execute it (nearly what happens, on PBS)
    job_bis = JobWrapper(experiment_parameters)
    job_bis.compute()

    # Re-check stuff
    print "Is completed?", job.check_completed()
    print "Result:", job.get_result()

    # Delete result filename
    os.remove(job.result_filename)


if __name__ == '__main__':
    test_job_wrapper()





