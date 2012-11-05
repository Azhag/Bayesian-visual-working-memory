#!/usr/bin/env python
# encoding: utf-8
"""
submitpbs.py

Created by Loic Matthey on 2012-10-29
Copyright (c) 2012 . All rights reserved.
"""

import glob
import re
import numpy as np
import argparse
import hashlib
import stat
import sys
import os


PBS_SCRIPT = """#!/bin/bash
{pbs_options}
hostn=`hostname`
echo "Job execution host: $hostn" 
echo "Filename: {filename}"
cd {working_dir}
nice -n 15 {cmd}
"""


class SubmitPBS():
    """
        Class creating scripts to be submitted to PBS and sending them.

        Adapted from J. Gasthaus's run_pbs script.
    """

    def __init__(self, working_directory=None, memory='2gb', walltime='1:00:00', set_env=True, scripts_dir='pbs_scripts', output_dir='pbs_output', debug=False):

        self.debug = debug

        self.pbs_options = {'mem': memory, 'pmem': memory, 'walltime': walltime, 'ncpus': '1'}
        self.set_env = set_env

        # Initialise the directories
        if working_directory is None:
            self.working_directory = os.getcwd()
        else:
            self.working_directory = working_directory

        self.scripts_dir = os.path.join(self.working_directory, scripts_dir)
        self.output_dir = os.path.join(self.working_directory, output_dir)
        self.make_dirs()

        # Open the Submit file, containing instructions to submit everything
        self.open_submit_file()

        if self.debug:
            print "SubmitPBS initialised:\n %s, %s, %s, %s" % (self.working_directory, self.scripts_dir, self.output_dir, self.pbs_options)


    def getEnvCommandString(self, command="export", env_keys_to_capture=frozenset(["PATH", "PYTHONPATH", "LD_LIBRARY_PATH"])):
        '''
            Catch and construct the list of environment variables to put in the script
        '''
        
        out = []
        out.append(command + " OMP_NUM_THREADS=1")
        for key in os.environ:
            if key in env_keys_to_capture:
                out.append(command + " " + key + '="'+ os.environ[key] + '"')
        return '\n'.join(out)


    def make_dirs(self):
        '''
            Create the directories for PBS submit scripts and PBS outputs.
        '''

        try:
            os.makedirs(self.scripts_dir)
        except:
            pass
        try:
            os.makedirs(self.output_dir)
        except:
            pass

        self.directories_created = True


    def open_submit_file(self, filename='submit_all.sh'):
        '''
            Open/create a file holding all the scripts created in the current script_directory.

            Could be improved to be a database of script/parameters/etc.. but not sure if really useful.
        '''

        self.submit_fn = os.path.join(self.scripts_dir, filename)

        try:
            # Try opening the file. If exist, we will just append to it
            with open(self.submit_fn, 'r'):
                pass

        except IOError:
            # No file, need to create it.
            with open(self.submit_fn, 'w') as submit_f:
                submit_f.write('#!/bin/sh\n')


    def add_to_submit_file(self, new_script_filename, command):
        '''
            Add a line in the Submit file
        '''

        with open(self.submit_fn, 'a') as submit_f:
            submit_f.write("qsub " + new_script_filename + " #" + command + '\n')



    def make_script(self, command):
        '''
            Create a script for a given command.

            Will output in the script_directory.
        '''
        
        # Create a new script. Its name is a md5 hash of the command to execute.
        fn = os.path.join(self.scripts_dir, "script." + str(hashlib.md5(command).hexdigest()))
        
        with open(fn, 'w') as f:
            # Construct the pbs options
            pbs_options = '\n'.join(["#PBS -l " + k + "=" + v for (k, v) in self.pbs_options.items()]) 

            # Add the environment
            if self.set_env:
                pbs_options = pbs_options + "\n" + self.getEnvCommandString()

            # Fill in the template script
            filled_script = PBS_SCRIPT.format(pbs_options=pbs_options, working_dir=self.working_directory, cmd=command, filename=fn)

            f.write(filled_script)

        # Make the script executable
        os.chmod(fn, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)
        
        # Add it to the list of jobs
        self.add_to_submit_file(fn, command)
        
        if self.debug:
            print "\n--- Script created: ---"
            print command

        return fn


    def submit_job(self, command):
        '''
            Take a command, create a script to run it on PBS and submits it
        '''

        # Create job
        new_script_filename = self.make_script(command)

        # Submit!
        if self.debug:
            print "\n-> Submitting job " + new_script_filename

        # Change to the PBS output directory first
        os.chdir(self.output_dir)

        # Submit the job
        os.popen("qsub " + new_script_filename)

        # Change back to the working directory
        os.chdir(self.working_directory)


    def create_simulation_command(self, pbs_command_infos, parameters):
        '''
            Generates a simulation command to be written in a script (and then possibly submitted to PBS).

            Takes a parameters dictionary, will automatically create an option with its content.

            Puts all other_options at the end, directly.
        '''
        
        assert pbs_command_infos is not None, "Provide pbs_command_infos..."

        # The command itself (e.g. python scriptname)
        simulation_command = pbs_command_infos['command']

        # The additional parameters we just found
        for param in parameters:
            simulation_command += " --{param} {param_value}".format(param=param, param_value=parameters[param])

        # Put the other parameters
        # assumes that if a parameter is already set in 'parameters', will not use the value from other_options (safer)
        for (param, param_value) in pbs_command_infos['other_options'].items():
            if param not in parameters:
                simulation_command += " --{param} {param_value}".format(param=param, param_value=param_value)

        return simulation_command


    def create_submit_job_parameters(self, pbs_command_infos, parameters, submit=True):
        '''
            Given some pbs_command_infos (command='python script', other_options='--stuff 10') and extra parameters,
            automatically generate a simulation command, a script to execute it and submits it to PBS (if desired)
        '''

        # Generate the command
        sim_cmd = self.create_simulation_command(pbs_command_infos, parameters)

        # Create the script and submits
        if submit:
            self.submit_job(sim_cmd)
        else:
            self.make_script(sim_cmd)





if __name__ == '__main__':
    submit_pbs = SubmitPBS(debug=True)

    # submit_pbs.make_script('echo "it works"')
    submit_pbs.submit_job('echo "it works"')

