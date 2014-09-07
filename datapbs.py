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
import progress

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

        if dataset_infos is not None:
            self.dataset_infos = dataset_infos

            # Load everything
            self.loaded_data = self.load_data()

            # Convert to ndarray
            if dataset_infos.get('construct_numpyarrays', True):
                self.dict_arrays = self.construct_multiple_numpyarrays()



    def load_data(self):
        '''
            Load multiple files from a directory. Calls subroutines, depending on where the parameter values are stored.

            self.dataset_infos['loading_type'] should be defined.
        '''

        if self.dataset_infos['loading_type'] == 'regexp':
            return self.load_data_fromregexp()
        elif self.dataset_infos['loading_type'] == 'args':
            return self.load_data_fromargs()
        else:
            raise ValueError('loading_type unknown: %s' % self.dataset_infos['loading_type'])


    def load_data_fromregexp(self):
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

        all_output_files = glob.glob(self.dataset_infos['files'])

        assert len(all_output_files) > 0, "No files founds. Wrong glob?"

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
            matched = re.search(self.dataset_infos['regexp'], curr_file)

            if not matched:
                print curr_file
                print self.dataset_infos['regexp']
                raise ValueError('No match. Wrong regular expression?')

            curr_params = matched.groupdict()

            # Check if all the appropriate parameters were found
            # assert set(self.dataset_infos['parameters']) <= set(curr_params), "Couldn't extract the desired parameters from the filename"
            if not (set(self.dataset_infos['parameters']) <= set(curr_params)):
                print set(self.dataset_infos['parameters'])
                print set(curr_params)
                raise ValueError("Couldn't extract the desired parameters from the filename")


            # Load the data
            curr_dataset = np.load(curr_file).item()
            datasets_list.append(curr_dataset)

            # Save the arguments of each dataset
            args_list.append(curr_dataset['args'])

            # Fill the parameter dictionary
            for param in self.dataset_infos['parameters']:
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
        for param in self.dataset_infos['parameters']:
            parameters_indirections[param] = dict()
            for i, par_val in enumerate(parameters_uniques[param]):
                parameters_indirections[param][par_val] = i

        return dict(parameters_uniques=parameters_uniques, parameters_complete=parameters_complete, datasets_list=datasets_list, parameters_indirections=parameters_indirections, args_list=args_list)



    def load_data_fromargs(self):
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

        all_output_files = glob.glob(self.dataset_infos['files'])

        assert len(all_output_files) > 0, "Wrong regular expression: " + self.dataset_infos['files']

        # We have to load each dataset, but also associate them with their parameter values.
        #  let's try and be general:
        #   - Store the datasets in a big list.
        #   - Store the associated parameter values in lists (same indexing), in a dictionary indexed by the parameters.
        datasets_list = []
        parameters_complete = dict()
        parameters_uniques = dict()
        parameters_dataset_index = dict()
        args_list = []

        load_progress = progress.Progress(len(all_output_files))

        for curr_file_i, curr_file in enumerate(all_output_files):

            # Load the data
            try:
                curr_dataset = np.load(curr_file).item()
            except IOError:
                # Failed to load, possibly as file is incomplete, skip.
                continue

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
            # assert set(self.dataset_infos['parameters']) <= set(curr_params), "Couldn't extract the desired parameters from the filename"
            if not (set(self.dataset_infos['parameters']) <= set(curr_args.keys())):
                print set(self.dataset_infos['parameters'])
                print set(curr_args)
                raise ValueError("Couldn't extract the desired parameters from the dataset's args variable")


            # Save the arguments of each dataset
            args_list.append(curr_args)

            # Fill the parameter dictionary
            for param_name in self.dataset_infos['parameters']:
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

            # Create a dictionary of parameters -> datasets index indirection
            param_index = tuple([curr_args[key] for key in self.dataset_infos['parameters']])
            if param_index in parameters_dataset_index:
                parameters_dataset_index[param_index].append(curr_file_i)
            else:
                parameters_dataset_index[param_index] = [curr_file_i]

            # Check number of dataset per parameters, indicating if multiple runs exist.
            nb_datasets_per_parameters = np.max([len(val) for key, val in parameters_dataset_index.items()])

            if self.debug:
                print curr_file
                print "%.2f%%, %s left - %s" % (load_progress.percentage(), load_progress.time_remaining_str(), load_progress.eta_str())
                print ', '.join(["%s %.2f" % (param, curr_args[param]) for param in self.dataset_infos['parameters']])

            load_progress.increment()


        # Extract the unique parameter values
        for key, val in parameters_complete.items():
            parameters_uniques[key] = np.unique(val)

        # Construct an indirection dictionary to give parameter index based on its value
        parameters_indirections = dict()
        for param in self.dataset_infos['parameters']:
            parameters_indirections[param] = dict()
            for i, par_val in enumerate(parameters_uniques[param]):
                parameters_indirections[param][par_val] = i

        return dict(parameters_uniques=parameters_uniques, parameters_complete=parameters_complete, datasets_list=datasets_list, parameters_indirections=parameters_indirections, args_list=args_list, parameters_dataset_index=parameters_dataset_index, nb_datasets_per_parameters=nb_datasets_per_parameters)


    def construct_numpyarray_specified_output_from_datasetlists(self, output_variable_desired):
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

        result_arrays_dict = dict()


        # Discover the result shape
        discovered_shapes_dict = self.discover_result_shape_numpyarray(output_variable_desired)

        ### Build a big N-D array, assuming grid-like coverage of features
        if self.dataset_infos.get('construct_multidimension_npyarr', True):
            results_array_multidim_dict = self.construct_numpyarray_specific_multidimensional(output_variable_desired, discovered_shapes_dict)

            result_arrays_dict.update(results_array_multidim_dict)


        ## Build the flat array. Well suited for random sampling
        if self.dataset_infos.get('construct_flat_npyarr', True):
            results_array_flat_dict = self.construct_numpyarray_specific_flat(output_variable_desired, discovered_shapes_dict)

            result_arrays_dict.update(results_array_flat_dict)

        return result_arrays_dict


    def discover_result_shape_numpyarray(self, output_variable_desired):
        '''
            Go over all datasets for a given output variable, discover the shape of the result.
        '''

        datasets_list = self.loaded_data['datasets_list']
        concatenate_multiple_datasets = self.dataset_infos.get('concatenate_multiple_datasets', False)

        if concatenate_multiple_datasets:
            # This is used to accommodate multiple datasets per parameters values, so that the dimensionality of the results_ arrays is increased
            nb_datasets_per_parameters = self.loaded_data.get('nb_datasets_per_parameters', 1)
        else:
            nb_datasets_per_parameters = 1

        # Assume that we will store the whole desired variable for each parameter setting.
        # Discover the shape
        curr_results_shape = (1, )
        results_shape = None
        for dataset in datasets_list:
            if output_variable_desired in dataset:
                if np.isscalar(dataset[output_variable_desired]):
                    curr_results_shape = (1, )
                else:
                    curr_results_shape = dataset[output_variable_desired].shape

                # Now keep track of the biggest results_shape found (tricky but simplest hack possible)
                if results_shape is None or np.any(curr_results_shape > results_shape):
                    results_shape = curr_results_shape
                    initial_results_shape = results_shape

                    if nb_datasets_per_parameters > 1:
                        # Found the shape, but now need to take into account number of repeats due to multiple datasets per parameter value
                        results_shape = list(results_shape)
                        results_shape[-1] *= nb_datasets_per_parameters
                        results_shape = tuple(results_shape)

                    print "Found new results_shape:", results_shape

        if results_shape is None:
            # Results_shape was never set, which means that this output_variable was in no dataset. This is not normal and most likely a human error when setting output_variables.
            # Just stop here.
            raise ValueError('Output variable %s was found in no dataset, are you sure?' % output_variable_desired)

        return dict(results_shape=results_shape, initial_results_shape=initial_results_shape)


    def construct_numpyarray_specific_flat(self, output_variable_desired, discovered_shapes_dict):
        '''
            Construct a flat result list of results.

            Good for random sweeps.

            Keeps ordering between parameters_list and results_list fixed, obviously.
        '''

        # Shortcuts
        parameters_complete = self.loaded_data['parameters_complete']
        datasets_list = self.loaded_data['datasets_list']
        parameters_indirections = self.loaded_data['parameters_indirections']
        list_parameters = self.dataset_infos['parameters']
        results_shape = discovered_shapes_dict['results_shape']
        initial_results_shape = discovered_shapes_dict['initial_results_shape']

        if self.debug:
            print '%s flat dimensions: %s of %s' % (output_variable_desired, len(datasets_list), results_shape)

        # Keep the results in a flat array
        results_flat = []
        parameters_flat = []

        # Get the array of how many repeats were actually finished
        completed_repeats_array = []


        for i, dataset in enumerate(datasets_list):
            # Now put the data at the appropriate position
            #   We construct a variable size index (depends on how many parameters we have),
            #    which will look in the indirection dictionary
            curr_dataposition = tuple([parameters_indirections[param][parameters_complete[param][i]] for param in list_parameters])

            if output_variable_desired in dataset:

                # Save the dataset
                if dataset[output_variable_desired].shape == initial_results_shape or np.isscalar(dataset[output_variable_desired]):

                    results_flat.append(dataset[output_variable_desired])

                else:
                    # Something is wrong with the result shapes... Just put as much as possible.
                    smallest_sizes = tuple([slice(None, min(results_shape[j], dataset[output_variable_desired].shape[j])) for j in xrange(len(results_shape))])
                    results_flat.append(dataset[output_variable_desired][smallest_sizes])

                # Keep the current parameters
                parameters_flat.append(np.array([parameters_complete[param][i] for param in list_parameters]))

                # For newer simulations, repet_i is the current repetition index. This allows to remove unfinished runs.
                # If not set, assume all are complete, and put the last index of the results (should be repetitions in last dimension anyway)
                completed_repeats_array.append(dataset.get('repet_i', results_shape[-1]))

            else:
                print curr_dataposition, " not in dataset. Output variable %s not found." % output_variable_desired

        return dict(repeats_completed=np.array(completed_repeats_array), results_flat=results_flat, parameters_flat=parameters_flat)


    def construct_numpyarray_specific_multidimensional(self, output_variable_desired, discovered_shapes_dict):
        '''
            Construct a multidimensional result numpy array.
            One dimension per parameter.

            Assumes that the number of parameter values per parameter is small, or you'll get huge matrices...
        '''

        # Shortcuts
        parameters_uniques = self.loaded_data['parameters_uniques']
        parameters_complete = self.loaded_data['parameters_complete']
        datasets_list = self.loaded_data['datasets_list']
        parameters_indirections = self.loaded_data['parameters_indirections']
        list_parameters = self.dataset_infos['parameters']
        concatenate_multiple_datasets = self.dataset_infos.get('concatenate_multiple_datasets', False)
        results_shape = discovered_shapes_dict['results_shape']
        initial_results_shape = discovered_shapes_dict['initial_results_shape']

        # The indices will go in the same order as the descriptive parameters list
        fullarray_shape = [parameters_uniques[param].size for param in list_parameters]

        # Don't forget to make space for the actual results...
        fullarray_shape.extend(results_shape)

        if self.debug:
            print '%s dimensions: %s' % (output_variable_desired, fullarray_shape)

        ## Check if it is going to be too large and will segfault...
        # assume 64 bit float (8 bytes), convert to Gb, see if larger than 4Go (process limit)
        memory_usage_array = np.prod(fullarray_shape)*8./1.e9
        if memory_usage_array > 4.:
            # Too large, just stop...
            print "Array is too large, will use %.2f Go. Cancelling..." % memory_usage_array
            return dict()

        # Initialize with NaN.
        results_array = np.ones(fullarray_shape)*np.nan

        # Keep the array of existing indices
        indices_array = []

        # Count how many datasets per parameters values were seen in a nice array (param_1*param2*..*paramk of ints)
        datasets_seen_array = np.zeros(tuple([parameters_uniques[param].size for param in list_parameters]), dtype=int)

        for i, dataset in enumerate(datasets_list):
            # Now put the data at the appropriate position
            #   We construct a variable size index (depends on how many parameters we have),
            #    which will look in the indirection dictionary
            curr_dataposition = tuple([parameters_indirections[param][parameters_complete[param][i]] for param in list_parameters])

            if output_variable_desired in dataset:

                # For multiple datasets per parameters, index the good position
                dataset_seens_repeat_i = datasets_seen_array[curr_dataposition]

                if dataset_seens_repeat_i == 0 or concatenate_multiple_datasets:
                    # First time seeing this parameter combination!
                    if dataset[output_variable_desired].shape == initial_results_shape or np.isscalar(dataset[output_variable_desired]):

                        # Save the dataset at the proper position
                        results_array[curr_dataposition][..., dataset_seens_repeat_i*initial_results_shape[-1]:(dataset_seens_repeat_i+1)*initial_results_shape[-1]] = dataset[output_variable_desired]

                    else:
                        # Something is wrong with the result shapes... Just put as much as possible.
                        smallest_sizes = tuple([slice(None, min(results_shape[j], dataset[output_variable_desired].shape[j])) for j in xrange(len(results_shape))])
                        results_array[curr_dataposition+smallest_sizes] = dataset[output_variable_desired][smallest_sizes]

                    # Indices in the array
                    indices_array.append(curr_dataposition)

                    datasets_seen_array[curr_dataposition] += 1

                else:
                    print "Duplicate parameters not allowed, discarded ", curr_dataposition
                    raise ValueError('weird')
            else:
                print curr_dataposition, " not in dataset. Output variable %s not found." % output_variable_desired

        return dict(results=results_array, datasets_seen_array=datasets_seen_array, indices=np.array(indices_array))


    def construct_multiple_numpyarrays(self):
        '''
            Constructs several numpy arrays, for each output variable given.

            Returns everything in a big dictionary, with the output variables as keys.

            (calls construct_numpyarray_specified_output_from_datasetlists)
        '''

        all_results_arrays = dict()

        for output_variable in self.dataset_infos['variables_to_load']:
            # Load each variable into a numpy array
            all_results_arrays[output_variable] = self.construct_numpyarray_specified_output_from_datasetlists(output_variable)

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



