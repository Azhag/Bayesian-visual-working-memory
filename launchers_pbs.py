#!/usr/bin/env python
# encoding: utf-8
"""
launchers_pbs.py


Created by Loic Matthey on 2013-05-21
Copyright (c) 2013 . All rights reserved.
"""

import os
import imp

from dataio import *
from datapbs import *
from submitpbs import *


def launcher_do_generate_submit_pbs_from_param_files(args):
    '''
        Generate a series of parameters to be run with PBS.

        Quite general, takes its parameters from a provided parameter_files file.
        (assume this is a .py file, which will be imported dynamically)

        If this parameter_file defines a filtering function, uses it.
    '''

    print "launcher_do_generate_submit_pbs_from_param_files, generating parameters..."


    all_parameters = vars(args)
    
    # Load the parameters from the specific file, fancyyyyy
    assert 'parameters_filename' in all_parameters and len(all_parameters['parameters_filename'])>0, "Parameters_filename is not set properly..."
    parameters_file = imp.load_source('params', all_parameters['parameters_filename'])

    ##### Now generate the parameters combinations and submit everything to PBS
    submit_pbs = SubmitPBS(pbs_submission_infos=parameters_file.pbs_submission_infos, debug=True)
    constrained_parameters = submit_pbs.generate_submit_constrained_parameters_from_module_parameters(parameters_file)

    dataio = DataIO(output_folder=all_parameters['output_directory'], label=os.path.splitext(all_parameters['parameters_filename'])[0])
    variables_to_save = ['constrained_parameters', 'all_parameters']
    dataio.save_variables(variables_to_save, locals())

    return locals()


def launcher_do_reload_constrained_parameters(dataset_infos):
    '''
        Reload outputs run with the automatic parameter generator for PBS

        Should handle random sampling of the parameter space.
    '''

    # Reload everything
    data_pbs = DataPBS(dataset_infos=dataset_infos, debug=True)

    # Reload the outputs of launcher used.
    # To be adapted, now have a file directly
    # launcher_variables = np.load(dataset_infos['launcher_file']).item()

    # Do the plots
    if dataset_infos['post_processing'] is not None:
        try:
            # Duck typing to check if we have a list of post_processings
            iterator = iter(dataset_infos['post_processing'])
        except TypeError:
            # Not a list... just call it
            dataset_infos['post_processing'](data_pbs, launcher_variables)
        else:
            for post_process in iterator:
                # Call each one one after the other
                post_process(data_pbs, launcher_variables)

    return locals()


