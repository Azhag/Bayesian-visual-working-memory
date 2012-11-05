#!/usr/bin/env python
# encoding: utf-8
"""
datapbs.py


Created by Loic Matthey on 2012-10-29
Copyright (c) 2012 . All rights reserved.
"""

import glob
import re
import numpy as np
import argparse


class DataPBS:
    '''
        Class reloading data created from PBS runs.

        Quite general, only requires a dataset_information dictionary.

        The data will be reloaded in appropriately sized ndarrays directly.
    '''

    def __init__(self, dataset_infos=None, debug=True):

        self.debug = debug

        self.loaded_data = None
        self.dict_arrays = None

        if dataset_infos:
            self.dataset_infos = dataset_infos

            # Load everything
            self.loaded_data = self.load_data(dataset_infos)

            # Convert to ndarray
            self.dict_arrays = self.construct_multiple_numpyarrays(self.loaded_data, self.dataset_infos['variables_to_load'], self.dataset_infos['parameters'])



    def load_data(self, dataset_infos):
        '''
            Load multiple files from a directory. Calls subroutines, depending on where the parameter values are stored.

            dataset_infos['loading_type'] should be defined.
        '''

        if dataset_infos['loading_type'] == 'regexp':
            return self.load_data_fromregexp(dataset_infos)
        elif dataset_infos['loading_type'] == 'args':
            return self.load_data_fromargs(dataset_infos)
        else:
            raise ValueError('loading_type unknown: %s' % dataset_infos['loading_type'])



    def load_data_fromregexp(self, dataset_infos):
        '''
            Load multiple files from a directory, where their filename indicates the parameter values used for the dataset.
                Assumes that all parameters are float.

            Returns the following:
                - Dictionary of uniques parameter values
                - Dictionary of the parameters values for each dataset (list in the same order as the datasets)
                - List of all datasets
                - Dictionary associating parameter values to their index in the unique lists.

            Can then be loaded into numpy arrays.

            Takes a description dictionary as input. Example format and keys:
            dict(
                label='Samples and sigmax effect on power-law fits',
                files='Data/effect_num_sample_on_powerlaw/multiple_memory_curve-samples*rcscale*sigmax*.npy',
                regexp='^[a-zA-Z_\/0-9]*-samples(?P<samples>[0-9]*)rcscale(?P<rcscale>[0-9.]*)sigmax(?P<sigmax>[0-9.]*)-[0-9a-z\-]*.npy',
                parameters=('samples', 'rcscale', 'sigmax')
                )
        '''

        all_output_files = glob.glob(dataset_infos['files'])

        assert len(all_output_files) > 0, "Wrong regular expression"

        # We have to load each dataset, but also associate them with their parameter values.
        #  let's try and be general:
        #   - Store the datasets in a big list.
        #   - Store the associated parameter values in lists (same indexing), in a dictionary indexed by the parameters.
        datasets_list = []
        parameters_complete = dict()
        parameters_uniques = dict()
        args_list = []

        for curr_file in all_output_files:
            
            # Do a nice regular expression to catch the parameters and remove the useless random unique string
            # (advanced, uses named groups now)
            matched = re.search(dataset_infos['regexp'], curr_file)
            
            if not matched:
                print curr_file
                print dataset_infos['regexp']
                raise ValueError('No match. Wrong regular expression?')

            curr_params = matched.groupdict()

            # Check if all the appropriate parameters were found
            # assert set(dataset_infos['parameters']) <= set(curr_params), "Couldn't extract the desired parameters from the filename"
            if not (set(dataset_infos['parameters']) <= set(curr_params)):
                print set(dataset_infos['parameters'])
                print set(curr_params)
                raise ValueError("Couldn't extract the desired parameters from the filename")


            # Load the data
            curr_dataset = np.load(curr_file).item()
            datasets_list.append(curr_dataset)

            # Save the arguments of each dataset
            args_list.append(curr_dataset['args'])

            # Fill the parameter dictionary
            for param in dataset_infos['parameters']:
                # Just append the parameter value of the current dataset to the appropriate list
                # warning: need to use the exact same string in the regexp and in the parameter names list
                if param in parameters_complete:
                    parameters_complete[param].append(float(curr_params[param]))
                else:
                    parameters_complete[param] = [float(curr_params[param])]

            if self.debug:
                print curr_file, curr_params


        
        # Extract the unique parameter values
        for key, val in parameters_complete.items():
            parameters_uniques[key] = np.unique(val)
        
        # Construct an indirection dictionary to give parameter index based on its value
        parameters_indirections = dict()
        for param in dataset_infos['parameters']:
            parameters_indirections[param] = dict()
            for i, par_val in enumerate(parameters_uniques[param]):
                parameters_indirections[param][par_val] = i

        return dict(parameters_uniques=parameters_uniques, parameters_complete=parameters_complete, datasets_list=datasets_list, parameters_indirections=parameters_indirections, args_list=args_list)



    def load_data_fromargs(self, dataset_infos):
        '''
            Load multiple files from a directory, where the parameter values used for the simulation are stored in the 'args' variable.
                Assumes that all parameters are float.

            Returns the following:
                - Dictionary of uniques parameter values
                - Dictionary of the parameters values for each dataset (list in the same order as the datasets)
                - List of all datasets
                - Dictionary associating parameter values to their index in the unique lists.

            Can then be loaded into numpy arrays.

            Takes a description dictionary as input. Example format and keys:
            dict(
                label='Samples and sigmax effect on power-law fits',
                files='Data/effect_num_sample_on_powerlaw/multiple_memory_curve-samples*rcscale*sigmax*.npy',
                parameters=('samples', 'rcscale', 'sigmax')
                )
        '''

        all_output_files = glob.glob(dataset_infos['files'])

        assert len(all_output_files) > 0, "Wrong regular expression"

        # We have to load each dataset, but also associate them with their parameter values.
        #  let's try and be general:
        #   - Store the datasets in a big list.
        #   - Store the associated parameter values in lists (same indexing), in a dictionary indexed by the parameters.
        datasets_list = []
        parameters_complete = dict()
        parameters_uniques = dict()
        args_list = []

        for curr_file in all_output_files:
            
            # Load the data
            curr_dataset = np.load(curr_file).item()
            datasets_list.append(curr_dataset)

            # Find out the parameter values
            if 'args' in curr_dataset:
                curr_args = curr_dataset['args']

                # Convert it to a dictionary to be able to generically access parameters...
                if type(curr_args) is argparse.Namespace:
                    curr_args = vars(curr_args)

                assert type(curr_args) is dict, "The args variable should be a dictionary now."
            else:
                raise ValueError('No args variable in this dataset, something is wrong. %s' % curr_file)


            # Check if all the appropriate parameters were found
            # assert set(dataset_infos['parameters']) <= set(curr_params), "Couldn't extract the desired parameters from the filename"
            if not (set(dataset_infos['parameters']) <= set(curr_args.keys())):
                print set(dataset_infos['parameters'])
                print set(curr_args)
                raise ValueError("Couldn't extract the desired parameters from the dataset's args variable")


            # Save the arguments of each dataset
            args_list.append(curr_args)

            # Fill the parameter dictionary
            for param_name in dataset_infos['parameters']:
                # Just append the parameter value of the current dataset to the appropriate list
                # warning: need to use the exact same string in the regexp and in the parameter names list
                if param_name in parameters_complete:
                    if np.isscalar(curr_args[param_name]):
                        # Scalar value, just append it
                        parameters_complete[param_name].append(curr_args[param_name])
                    else:
                        # Non-scalar, assume its a list and extend...
                        parameters_complete[param_name].extend(curr_args[param_name])
                else:
                    # First time we see a parameter value of this parameter
                    parameters_complete[param_name] = []
                    if np.isscalar(curr_args[param_name]):
                        parameters_complete[param_name].append(curr_args[param_name])
                    else:
                        parameters_complete[param_name].extend(curr_args[param_name])

            if self.debug:
                print curr_file, ', '.join(["%s %.2f" % (param, curr_args[param]) for param in dataset_infos['parameters']])


        # Extract the unique parameter values
        for key, val in parameters_complete.items():
            parameters_uniques[key] = np.unique(val)
        
        # Construct an indirection dictionary to give parameter index based on its value
        parameters_indirections = dict()
        for param in dataset_infos['parameters']:
            parameters_indirections[param] = dict()
            for i, par_val in enumerate(parameters_uniques[param]):
                parameters_indirections[param][par_val] = i

        return dict(parameters_uniques=parameters_uniques, parameters_complete=parameters_complete, datasets_list=datasets_list, parameters_indirections=parameters_indirections, args_list=args_list)



    def construct_numpyarray_specified_output_from_datasetlists(self, loaded_data, output_variable_desired, list_parameters):
        '''
            Construct a big numpy array out of a series of datasets, extracting a specified output variable of each dataset
             (usually, the results of the simulations, let's say)
            Looks only at a list of parameters, which can be of any size. Doesn't require any fixed dimensions per say (yeah I'm happy)

            Input:
                - the name of the output variable to extract from each dataset
                - Several dictionaries, created by load_data_fromregexp (or another function)

            Output:
                - A numpy array of variable size (parameters sizes found in dataset x output shape)
        '''

        # Reload some variables, to lighten the notation
        parameters_uniques = loaded_data['parameters_uniques']
        parameters_complete = loaded_data['parameters_complete']
        datasets_list = loaded_data['datasets_list']
        parameters_indirections = loaded_data['parameters_indirections']

        # Assume that we will store the whole desired variable for each parameter setting.
        # Discover the shape
        results_shape = (1, )
        for dataset in datasets_list:
            if output_variable_desired in dataset:
                results_shape = dataset[output_variable_desired].shape
                break

        # The indices will go in the same order as the descriptive parameters list
        fullarray_shape = [parameters_uniques[param].size for param in list_parameters]

        # Don't forget to make space for the actual results...
        fullarray_shape.extend(results_shape)

        if self.debug:
            print '%s dimensions: %s' % (output_variable_desired, fullarray_shape)
        
        # Initialize with NaN.
        results_array = np.ones(fullarray_shape)*np.nan

        # Keep the array of existing indices
        indices_array = []
        # Get the array of how many repeats were actually finished
        completed_repeats_array = []

        for i, dataset in enumerate(datasets_list):
            # Now put the data at the appropriate position
            #   We construct a variable size index (depends on how many parameters we have),
            #    which will look in the indirection dictionary
            curr_dataposition = tuple([parameters_indirections[param][parameters_complete[param][i]] for param in list_parameters])

            if output_variable_desired in dataset:
                if not curr_dataposition in indices_array:
                    if dataset[output_variable_desired].shape == results_shape:
                        # Save the dataset at the proper position
                        results_array[curr_dataposition] = dataset[output_variable_desired]
                        indices_array.append(curr_dataposition)

                        if 'repet_i' in dataset:
                            # For newer simulations, we keep the current repetition index. This allows to remove unfinished runs.
                            completed_repeats_array.append(dataset['repet_i'])
                        else:
                            # If nothing, assumed all are complete, and put the last index of the results (should be repetitions in last dimension anyway)
                            completed_repeats_array.append(fullarray_shape[-1])
                    else:
                        # Something is wrong with the result shapes... Just put as much as possible.
                        smallest_sizes = tuple([slice(None, min(results_shape[i], dataset[output_variable_desired].shape[i])) for i in range(len(results_shape))])
                        results_array[curr_dataposition+smallest_sizes] = dataset[output_variable_desired][smallest_sizes]
                else:
                    # Duplicate entry
                    print "duplicate for %s" % curr_dataposition
            else:
                print curr_dataposition, " not in dataset"

        # and we're good
        return dict(results=results_array, indices=np.array(indices_array), repeats_completed=np.array(completed_repeats_array))



    def construct_multiple_numpyarrays(self, loaded_data, list_output_variables, list_parameters):
        '''
            Constructs several numpy arrays, for each output variable given.

            Returns everything in a big dictionary, with the output variables as keys.

            (calls construct_numpyarray_specified_output_from_datasetlists)
        '''

        all_results_arrays = dict()

        for output_variable in list_output_variables:
            # Load each variable into a numpy array
            all_results_arrays[output_variable] = self.construct_numpyarray_specified_output_from_datasetlists(loaded_data, output_variable, list_parameters)

        return all_results_arrays



if __name__ == '__main__':
    dataset_infos = dict(label='New PBS runs, different loading method. Uses the 2D fisher information as a constraint between sigma and rcscale. Also checks the ratio between recall precision and FI curve.',
                    files='Data/constraint/allfi_N200samples300/allfi_*-launcher_do_fisher_information_param_search_pbs-*.npy',
                    loading_type='args',
                    parameters=('rc_scale', 'sigmax'),
                    variables_to_load=('FI_rc_curv', 'FI_rc_precision', 'FI_rc_theo'),
                    variables_description=('FI curve', 'FI recall precision', 'FI theo'),
                    )

    data_pbs = DataPBS(dataset_infos=dataset_infos, debug=False)


