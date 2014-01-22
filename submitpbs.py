#!/usr/bin/env python
# encoding: utf-8
"""
submitpbs.py

Created by Loic Matthey on 2012-10-29
Copyright (c) 2012 . All rights reserved.
"""

import numpy as np
import hashlib
import stat
import os
import subprocess
import progress
import functools
import time
import getpass

from utils import cross


PBS_SCRIPT = """#!/bin/bash
{pbs_options}
hostn=`hostname`
echo "Job execution host: $hostn"
echo "Filename: {filename}"
cd {working_dir}
nice -n 15 {cmd}
echo "+++ Job completed +++"
"""


### Random unix commands
# - Rerun some pbs scripts that failed:
# find ../pbs_output/ -name "script.*.e*" -size +1k -exec sh -c "basename {} | cut -d "." -f1-2 | qsub" \;
# (error logs bigger than 1K had to be rerun)
#
# - Rerun unfinished scripts: (run in pbs_output)
# for f in ../pbs_scripts/script.*; do basef=`basename $f`; if grep -Fq $basef *; then echo "$basef ok"; else echo "$basef rerun"; qsub $f; fi; done;


class SubmitPBS():
    """
        Class creating scripts to be submitted to PBS and sending them.

        Also handles generating sets of parameters, checking them for a condition and then submitting them on PBS.

        Adapted from J. Gasthaus's run_pbs script.
    """

    def __init__(self, pbs_submission_infos=None, working_directory=None, memory='2gb', walltime='1:00:00', set_env=True, scripts_dir='pbs_scripts', output_dir='pbs_output', wait_submitting=False, submit_label='', pbs_submit_cmd='qsub', limit_max_queued_jobs=0, debug=False):

        self.debug = debug

        if pbs_submission_infos is not None:
            # Extract informations from this dictionary instead
            working_directory   = pbs_submission_infos['simul_out_dir']
            memory              = pbs_submission_infos['memory']
            walltime            = pbs_submission_infos['walltime']

            wait_submitting = pbs_submission_infos.get('wait_submitting', wait_submitting)

            submit_label = pbs_submission_infos.get('submit_label', submit_label)

            pbs_submit_cmd = pbs_submission_infos.get('pbs_submit_cmd', pbs_submit_cmd)

            limit_max_queued_jobs = pbs_submission_infos.get('limit_max_queued_jobs', limit_max_queued_jobs)

        self.pbs_options = {'mem': memory, 'pmem': memory, 'walltime': walltime, 'ncpus': '1'}
        self.set_env = set_env
        self.wait_submitting = wait_submitting
        self.limit_max_queued_jobs = limit_max_queued_jobs
        self.num_queued_jobs = 0
        self.submit_label = submit_label
        self.pbs_submit_cmd = pbs_submit_cmd

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


    def update_running_jobs_number(self):
        '''
            Get the current number of jobs on the queue

            Should work for both PBS and SLURM
        '''
        username = getpass.getuser()

        if self.pbs_submit_cmd == 'qsub':
            # Using PBS
            queue_status = subprocess.Popen(['qstat', '-u', username], stdout=subprocess.PIPE)
        elif self.pbs_submit_cmd == 'sbatch':
            # Using SLURM
            queue_status = subprocess.Popen(['squeue', '-h'], stdout=subprocess.PIPE)

        lines = queue_status.communicate()[0].splitlines()
        self.num_queued_jobs = len([line for line in lines if username in line])


    def wait_queue_not_full(self, sleeping_period=dict(min=30, max=180)):
        '''
            Wait for the queue to have empty slots, compared to self.limit_max_queued_jobs

            Sleeps for random amounts between [sleeping_period.min, sleeping_period.max]
        '''

        # Update running jobs number
        self.update_running_jobs_number()

        while self.num_queued_jobs >= self.limit_max_queued_jobs:
            # Decide for how long to sleep
            sleep_time_rnd = np.random.randint(sleeping_period['min'], sleeping_period['max'])

            if self.debug:
                print "Queue full (%d queued/%d limit max), waiting %d sec..." % (self.num_queued_jobs, self.limit_max_queued_jobs, sleep_time_rnd)

            # Sleep for a bit
            time.sleep(sleep_time_rnd)

            # Update running jobs number
            self.update_running_jobs_number()



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
            submit_f.write(self.pbs_submit_cmd + " " + new_script_filename + " #" + command + '\n')


    ########################################

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

        # Increment submitted job number
        self.num_queued_jobs += 1


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
                if param_value is not None:
                    simulation_command += " --{param} {param_value}".format(param=param, param_value=param_value)
                else:
                    simulation_command += " --{param}".format(param=param)

        return simulation_command


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

            # Add SLURM options
            pbs_options += "\n#SBATCH -n1 --time={walltime} --mem-per-cpu={mem}".format(**self.pbs_options)

            # Add the label
            if self.submit_label:
                pbs_options += "\n#PBS -N " + self.submit_label
                pbs_options += "\n#SBATCH -J " + self.submit_label

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
            print command + "\n"

        return fn


    def submit_job(self, command):
        '''
            Take a command, create a script to run it on PBS and submits it
        '''

        # Create job
        new_script_filename = self.make_script(command)

        # Wait for queue to have space for our jobs, if desired
        if self.limit_max_queued_jobs > 0:
            # Wait for the queue to be nearly full before checking and waiting
            if self.num_queued_jobs > 3*self.limit_max_queued_jobs/4:
                self.wait_queue_not_full()

        # Submit!
        if self.debug:
            print "\n-> Submitting job " + new_script_filename + "\n"

        # Change to the PBS output directory first
        os.chdir(self.output_dir)

        # Submit the job
        # subprocess.Popen([self.pbs_submit_cmd, new_script_filename], shell=True, env=os.environ, stderr=subprocess.STDOUT, stdout=subprocess.STDOUT)
        os.popen(self.pbs_submit_cmd + " " + new_script_filename)

        # Change back to the working directory
        os.chdir(self.working_directory)

        # Wait a bit, randomly
        if self.wait_submitting:
            try:
                sleeping_time = np.random.normal(1.0, 0.5)
                if sleeping_time < 10.:
                    time.sleep(sleeping_time)
            except:
                pass



    #################################################

    def generate_submit_constrained_parameters_from_module_parameters(self, module_parameters):
        '''
            Entry-point for general parameter tuples creation. Optionally submits to PBS directly.

            Uses the new system that loads .py files as modules, which can contain parameters/functions directly.

            Loads generate_submit_constrained_parameters next, which then loads generate_submit_constrained_parameters_grid/generate_submit_constrained_parameters_random accordingly.
        '''


        # If no filtering function given, just set it to None for the rest of the script
        if 'filtering_function' not in module_parameters.__dict__:
            module_parameters.__dict__['filtering_function'] = None
        if 'filtering_function_parameters' not in module_parameters.__dict__:
            module_parameters.__dict__['filtering_function_parameters'] = None
        if 'num_random_samples' not in module_parameters.__dict__:
            module_parameters.__dict__['num_random_samples'] = 100
        if 'max_queued_jobs' not in module_parameters.__dict__:
            module_parameters.__dict__['max_queued_jobs'] = 500

        # Keep track of submitted jobs numbers and available queue size if desired
        if self.limit_max_queued_jobs > 0:
            # Get current queue state. Expensive operation so don't do it too much...
            self.update_running_jobs_number()

        if module_parameters.parameter_generation == 'grid':
            return self.generate_submit_constrained_parameters_grid(module_parameters.dict_parameters_range, filtering_function=module_parameters.filtering_function, filtering_function_parameters=module_parameters.filtering_function_parameters, pbs_submission_infos=module_parameters.pbs_submission_infos, submit_jobs=module_parameters.submit_jobs)
        elif module_parameters.parameter_generation == 'random':
            return self.generate_submit_constrained_parameters_random(module_parameters.dict_parameters_range, num_random_samples=module_parameters.num_random_samples, max_queued_jobs=module_parameters.max_queued_jobs, filtering_function=module_parameters.filtering_function, filtering_function_parameters=module_parameters.filtering_function_parameters, pbs_submission_infos=module_parameters.pbs_submission_infos, submit_jobs=module_parameters.submit_jobs)


    def generate_submit_constrained_parameters_random(self, dict_parameters_range, num_random_samples=100, max_queued_jobs=500, filtering_function=None, filtering_function_parameters=None, pbs_submission_infos=None, submit_jobs=True):
        '''
            Takes a dictionary of parameters (which should contain low/high values for each or generator function), and
            generates num_random_samples possible parameters randomly.
            Checks them with some provided function before accepting them if provided.
            If pbs_submission_infos is provided, will additionally automatically create scripts and submit them to PBS. The PBS informations were already read from it in __init__ though, be careful.
        '''

        # Convert specific parameter sampling methods into generators
        self.create_generator_random_parameter(dict_parameters_range)

        # We will keep all constrained parameters for further use
        constrained_parameters = []

        fill_parameters_progress = progress.Progress(num_random_samples)
        tested_parameters = 0

        # Provide as many experimentally constrained parameters as desired
        while len(constrained_parameters) < num_random_samples:
            print "Parameters tested %d, found %d. %d requested. %.2f%%, %s left - %s" % (tested_parameters, len(constrained_parameters), num_random_samples, fill_parameters_progress.percentage(), fill_parameters_progress.time_remaining_str(), fill_parameters_progress.eta_str())

            # Sample new parameter values
            new_parameters = {}
            for curr_param, param_dict in dict_parameters_range.items():
                # Use the provided sampling function (some are predefined)
                new_parameters[curr_param] = param_dict['sampling_fct']()

            # Check if the new parameters are within the constraints
            if (filtering_function and filtering_function(new_parameters, dict_parameters_range, filtering_function_parameters)) or (filtering_function is None):
                # Yes, all good

                # Append to our parameters
                constrained_parameters.append(new_parameters)

                # If desired, generate a script and submits it to PBS
                if pbs_submission_infos:
                    self.create_submit_job_parameters(pbs_submission_infos, new_parameters, submit=submit_jobs)

                fill_parameters_progress.increment()

            tested_parameters += 1

        if self.debug:
            print "\n-- Submitted/created %d jobs --\n" % self.num_queued_jobs

        return constrained_parameters


    def generate_submit_constrained_parameters_grid(self, dict_parameters_range, filtering_function=None, filtering_function_parameters=None, pbs_submission_infos=None, submit_jobs=True):
        '''
            Takes a dictionary of parameters, with their list of values, and generates a list of all the combinations.
            Filter them with a specific function if provided.
            if pbs_submission_infos is provided, will create a script and submit it to PBS when an acceptable set of parameters if found
        '''

        candidate_parameters = []

        # Get all cross combinations of parameters.
        # Also converts the type on the fly to correspond with the specified dtype.
        cross_comb = cross([[dict_parameters_range[param]['dtype'](param_elem) for param_elem in dict_parameters_range[param]['range'].tolist()] for param in dict_parameters_range])
        # Convert them back into dictionaries
        candidate_parameters = [dict(zip(dict_parameters_range.keys(), x)) for x in cross_comb]

        # Some debuging info
        if self.debug:
            print "\n=== Generating up to %d candidate parameter sets ===\n" % len(candidate_parameters)

        # Now filter them
        constrained_parameters = []
        for new_parameters in progress.ProgressDisplay(candidate_parameters, display=progress.SINGLE_LINE):
            if (filtering_function is None) or (filtering_function and filtering_function(new_parameters, dict_parameters_range, filtering_function_parameters)):

                constrained_parameters.append(new_parameters)

                # Submit to PBS if required
                if pbs_submission_infos:

                    self.create_submit_job_parameters(pbs_submission_infos, new_parameters, submit=submit_jobs)

        if self.debug:
            print "\n-- Submitted/created %d jobs --\n" % self.num_queued_jobs

        return constrained_parameters


    def create_generator_random_parameter(self, dict_parameters_range):
        '''
            Support some predefined random parameter sampling:
                - Uniform[a,b]
                - randint(a, b)

            After this function, a sampling_fct() will be set for each parameter
        '''

        for param_name, dict_specific_param_range in dict_parameters_range.items():
            if 'sampling_fct' not in dict_specific_param_range:
                # No sampling function, try to automatically define one
                if dict_specific_param_range['sampling_type'] == 'uniform':
                    # functool.partial is used to freeze the dict_parameters to the one at execution time. Work-around for immutable + functional programming madness.
                    dict_specific_param_range['sampling_fct'] = functools.partial(lambda params: params['low'] + np.random.rand()*(params['high'] - params['low']), dict_specific_param_range)
                elif dict_specific_param_range['sampling_type'] == 'randint':
                    dict_specific_param_range['sampling_fct'] = functools.partial(lambda params: np.random.randint(params['low'], params['high']), dict_specific_param_range)
                else:
                    raise ValueError('Sampling_type unknown and no sampling_fct provided... Cannot sample parameter %s' % param_name)




if __name__ == '__main__':

    # Simple usage
    cwd = os.getcwd()
    submit_pbs = SubmitPBS(working_directory=os.path.join(cwd, 'simple_pbs_test'), debug=True)
    # submit_pbs.make_script('echo "it works"')
    submit_pbs.submit_job('echo "it works"')

    # This is how you should generate random parameters and scripts
    num_samples = 10
    pbs_submission_infos = dict(description='Getting the full memory curves for all the parameters compatible with the experimental Fisher information', command='echo "Testing PBS"', other_options=dict(option1='1', option2=2), walltime='10:00:00', memory='2gb', simul_out_dir=os.path.join(cwd, 'testing_pbs'))
    dict_parameters_range = dict(param1=dict(sampling_type='uniform', low=0.01, high=15.0, dtype=float), param2=dict(sampling_type='uniform', low=0.01, high=0.8, dtype=float))
    filtering_function = lambda parameters, dict_parameters_range, other_options: parameters['param1'] > 1.0

    submit_pbs = SubmitPBS(pbs_submission_infos=pbs_submission_infos, debug=True)
    submit_pbs.generate_submit_constrained_parameters_random(dict_parameters_range, num_samples=num_samples, filtering_function=filtering_function, filtering_function_parameters=None, pbs_submission_infos=pbs_submission_infos, submit_jobs=False)




