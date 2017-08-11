'''
    Small class system to simplify the process of loading Experimental datasets
'''

import numpy as np
import utils
import experimentalloader
# import bottleneck as bn
import em_circularmixture
import em_circularmixture_parametrickappa
import em_circularmixture_parametrickappa_doublepowerlaw

em_circular_mixture_to_use = em_circularmixture

class ExperimentalLoaderGorgo11Sequential(experimentalloader.ExperimentalLoader):
    """docstring for ExperimentalLoaderGorgo11Sequential"""
    def __init__(self, dataset_description):
        super(self.__class__, self).__init__(dataset_description)


    def preprocess(self, parameters):
        '''
        For sequential datasets, need to convert to radians and correct the probe indexing.
        '''

        # Convert everything to radians, spanning a -np.pi/2:np.pi
        if parameters.get('convert_radians', True):  # pylint: disable=E0602
            self.convert_wrap()

        # Correct the probe field, Matlab format for indices...
        # if parameters.get('correct_probe', True) and 'probe' in self.dataset:  #pylint: disable=E0602
            # self.dataset['probe'] = self.dataset['probe'].astype(int)
            # self.dataset['probe'] -= 1


        self.dataset['n_items'] = self.dataset['n_items'].astype(int)
        self.dataset['subject'] = self.dataset['subject'].astype(int)

        self.dataset['Ntot'] = self.dataset['probe'].size

        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size


        # Will remove delayed trials
        self.dataset['masked'] = self.dataset['delayed'] == 1

        # Reconstruct the colour information_
        self.reconstruct_colours()

        # Compute additional errors, between the response and all items
        self.compute_all_errors()

        # Create arrays per subject
        self.create_subject_arrays()

        # Fit the mixture model, and save the responsibilities per datapoint.
        if parameters.get('fit_mixture_model', False):
            self.fit_mixture_model_cached(
                caching_save_filename=parameters.get(
                    'mixture_model_cache', None),
                saved_keys=['em_fits', 'em_fits_nitems_mean_arrays',
                            'em_fits_nitems_trecall',
                            'em_fits_nitems_trecall_arrays',
                            'em_fits_nitems_trecall_mean',
                            'em_fits_nitems_trecall_mean_arrays',
                            'em_fits_subjects_nitems',
                            'em_fits_subjects_nitems_arrays',
                            'em_fits_subjects_nitems_trecall',
                            'em_fits_subjects_nitems_trecall_arrays'])

        ## Save item in a nice format for the model fit
        # self.generate_data_to_fit()
        self.generate_data_subject_split()
        self.generate_data_to_fit()


        if parameters.get('fit_mixture_model', False):
            self.fit_collapsed_mixture_model_cached(caching_save_filename=parameters.get('collapsed_mixture_model_cache', None), saved_keys=['collapsed_em_fits_subjects_nitems', 'collapsed_em_fits_nitems', 'collapsed_em_fits_subjects_trecall', 'collapsed_em_fits_trecall', 'collapsed_em_fits_doublepowerlaw', 'collapsed_em_fits_doublepowerlaw_subjects', 'collapsed_em_fits_doublepowerlaw_array'])

        # Perform Vtest for circular uniformity
        self.compute_vtest()

        # Do per subject and nitems, get average histogram
        # self.compute_average_histograms()

    def reconstruct_colours(self):
        ''' Will recreate angular colour probes
        '''
        self.dataset['item_colour_id'] = self.dataset['item_colour'][:]

        # Create linearly spaced "colors"
        nb_colours = np.nanmax(self.dataset['item_colour'])

        # Handle np.nan with an indexing trick
        angular_colours = np.r_[np.nan, np.linspace(-np.pi, np.pi, nb_colours, endpoint=False)]
        self.dataset['item_colour'] = angular_colours[
            np.ma.masked_invalid(
                self.dataset['item_colour']).filled(fill_value=0.0).astype(int)]

    def extract_target_nontargets_columns(self, data, probe):
        '''
            Given an array NxK, where K is the number of items,
            should return the column corresponding to the target/probe, and the others columns
            When probe != 0, this is a bit annoying
        '''

        indices_columns = np.arange(data.shape[1])
        target_data = data[:, indices_columns == probe-1].flatten()
        nontarget_data = data[:, indices_columns != probe-1]

        return target_data, nontarget_data



    def create_subject_arrays(self, double_precision=True):
        '''
            Create arrays with errors per subject and per num_target
            also create an array with the precision per subject and num_target directly
        '''

        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        self.dataset['errors_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['sizes_subject_nitems_trecall'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_bays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_theo'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_theo_nochance'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))
        self.dataset['precision_subject_nitems_trecall_bays_notreatment'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size))

        self.dataset['response_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_angle_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_colour_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['target_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['nontargets_subject_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)


        self.dataset['errors_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_all_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['errors_nontarget_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['response_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['item_angle_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['target_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['nontargets_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)

        self.dataset['precision_nitems_trecall_bays'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_theo_nochance'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))
        self.dataset['precision_nitems_trecall_bays_notreatment'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size))



        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    ids_filtered = (
                        (self.dataset['subject'] == subject) &
                        (self.dataset['n_items'] == n_items) &
                        (self.dataset['probe'] == trecall) &
                        (~self.dataset['masked'])
                    ).flatten()

                    # Invert the order of storage, 0 -> last item probed, 1 -> second to last item probe, etc...
                    trecall_i = n_items - trecall

                    # Get the errors
                    self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['errors_all'][ids_filtered]

                    self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['errors_nontarget_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.extract_target_nontargets_columns(self.dataset['errors_all_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], trecall)


                    # Get the responses and correct item angles
                    # TODO (lmatthey) trecall here is inverted, should really fix it somehow...
                    self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['response'][ids_filtered].flatten()
                    self.dataset['item_angle_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['item_angle'][ids_filtered]
                    self.dataset['item_colour_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['item_colour'][ids_filtered]

                    # Save target item and nontargets as well
                    self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.extract_target_nontargets_columns(self.dataset['item_angle'][ids_filtered], trecall)

                    # Get the number of samples per conditions
                    self.dataset['sizes_subject_nitems_trecall'][subject_i, n_items_i, trecall_i] = self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i].size

                    # Compute the precision
                    self.dataset['precision_subject_nitems_trecall_bays'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
                    self.dataset['precision_subject_nitems_trecall_theo'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
                    self.dataset['precision_subject_nitems_trecall_theo_nochance'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
                    self.dataset['precision_subject_nitems_trecall_bays_notreatment'][subject_i, n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=True)


        # Store all/average subjects data
        for n_items_i, n_items in enumerate(unique_n_items):
            for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                self.dataset['errors_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['errors_all_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_all_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['errors_nontarget_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['errors_nontarget_subject_nitems_trecall'][:, n_items_i, trecall_i]))

                # Responses, target, nontarget
                self.dataset['response_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['response_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['target_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['target_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['nontargets_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['nontargets_subject_nitems_trecall'][:, n_items_i, trecall_i]))
                self.dataset['item_angle_nitems_trecall'][n_items_i, trecall_i] = np.array(utils.flatten_list(self.dataset['item_angle_subject_nitems_trecall'][:, n_items_i, trecall_i]))


                # Precision over all subjects errors (not average of precisions)
                self.dataset['precision_nitems_trecall_bays'][n_items_i, trecall_i] = self.compute_precision(self.dataset['errors_nitems_trecall'][n_items_i, trecall_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)

                # self.dataset['precision_nitems_trecall_bays'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays'], axis=0)
                self.dataset['precision_nitems_trecall_theo'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo'], axis=0)
                self.dataset['precision_nitems_trecall_theo_nochance'] = np.mean(self.dataset['precision_subject_nitems_trecall_theo_nochance'], axis=0)
                self.dataset['precision_nitems_trecall_bays_notreatment'] = np.mean(self.dataset['precision_subject_nitems_trecall_bays_notreatment'], axis=0)



    def compute_vtest(self):
        unique_n_items = np.unique(self.dataset['n_items'])
        self.dataset['vtest_nitems_trecall'] = np.empty((
            unique_n_items.size, unique_n_items.size))*np.nan
        for n_items_i, n_items in enumerate(unique_n_items):
            for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                curr_errors = utils.dropnan(
                    self.dataset['errors_nontarget_nitems_trecall'
                                 ][n_items - 1, trecall - 1]).flatten()
                if curr_errors.size > 0:
                    (self.dataset['vtest_nitems_trecall'][n_items_i, trecall_i]
                     ) = utils.V_test(curr_errors)['pvalue']


    def fit_mixture_model(self):
        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        # Initialize empty arrays
        em_fits_keys = ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_nontargets_sum', 'mixt_random', 'train_LL', 'K', 'aic', 'bic']

        self.dataset['em_fits'] = dict()
        for k in em_fits_keys:
            self.dataset['em_fits'][k] = np.nan*np.empty(self.dataset['probe'].size)

        self.dataset['em_fits']['resp_target'] = np.nan*np.empty(self.dataset['probe'].size)
        self.dataset['em_fits']['resp_nontarget'] = np.nan*np.empty(self.dataset['probe'].size)
        self.dataset['em_fits']['resp_random'] = np.nan*np.empty(self.dataset['probe'].size)

        self.dataset['em_fits_subjects_nitems_trecall'] = np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['em_fits_nitems_trecall'] = np.empty((unique_n_items.size, unique_n_items.size), dtype=np.object)
        self.dataset['em_fits_subjects_nitems'] = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)

        # for subject_i, subject in enumerate(unique_subjects):
        #     self.dataset['em_fits_subjects_nitems_trecall'][subject] = dict()
        #     for n_items_i, n_items in enumerate(unique_n_items):
        #         self.dataset['em_fits_subjects_nitems_trecall'][subject][n_items] = dict()

        self.dataset['em_fits_nitems_trecall_mean'] = dict(mean=dict(), std=dict(), values=dict())

        # Compute mixture model fits per n_items, subject and trecall
        for n_items_i, n_items in enumerate(unique_n_items):
            for subject_i, subject in enumerate(unique_subjects):
                for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                    ids_filtered = (
                        (self.dataset['subject'] == subject) &
                        (self.dataset['n_items'] == n_items) &
                        (self.dataset['probe'] == trecall) &
                        (not self.dataset.get('masked', False))).flatten()
                    # Invert the order of storage, 0 -> last item probed, 1 -> second to last item probe, etc...
                    # trecall_i = n_items - trecall

                    print "Fit mixture model, %d items, subject %d, trecall %d, %d datapoints (%d)" % (n_items, subject, trecall, np.sum(ids_filtered), self.dataset['sizes_subject_nitems_trecall'][subject_i, n_items_i, trecall_i])

                    params_fit = em_circular_mixture_to_use.fit(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i])
                    params_fit['mixt_nontargets_sum'] = np.sum(params_fit['mixt_nontargets'])
                    # print self.dataset['response'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 0].shape, self.dataset['item_angle'][ids_filtered, 1:].shape

                    # cross_valid_outputs = em_circularmixture.cross_validation_kfold(self.dataset['response'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 0], self.dataset['item_angle'][ids_filtered, 1:], K=10, shuffle=True, debug=False)
                    # params_fit = cross_valid_outputs['best_fit']
                    resp = em_circular_mixture_to_use.compute_responsibilities(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, trecall_i], params_fit)

                    for k, v in params_fit.iteritems():
                        self.dataset['em_fits'][k][ids_filtered] = v

                    # params_fit['responsibilities'] = resp

                    self.dataset['em_fits']['resp_target'][ids_filtered] = resp['target']
                    self.dataset['em_fits']['resp_nontarget'][ids_filtered] = np.sum(resp['nontargets'], axis=1)
                    self.dataset['em_fits']['resp_random'][ids_filtered] = resp['random']

                    self.dataset['em_fits_subjects_nitems_trecall'][subject_i, n_items_i, trecall_i] = params_fit

                # Do not look at trecall (weird but whatever)
                params_fit = em_circular_mixture_to_use.fit(np.array(utils.flatten_list(self.dataset['response_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])), np.array(utils.flatten_list(self.dataset['target_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])), np.array(utils.flatten_list(self.dataset['nontargets_subject_nitems_trecall'][subject_i, n_items_i, :n_items_i+1])))

                self.dataset['em_fits_subjects_nitems'][subject_i, n_items_i] = params_fit


        for n_items_i, n_items in enumerate(unique_n_items):
            for k in ['mean', 'std', 'values']:
                self.dataset['em_fits_nitems_trecall_mean'][k][n_items] = dict()

            for trecall_i, trecall in enumerate(np.arange(1, n_items + 1)):
                for k in ['mean', 'std', 'values']:
                    self.dataset['em_fits_nitems_trecall_mean'][k][n_items][trecall] = dict()

                ## Now compute mean/std em_fits per n_items, trecall
                # Refit the model mixing all subjects together (not sure how we could get sem, 1-held?)
                params_fit = em_circular_mixture_to_use.fit(self.dataset['response_nitems_trecall'][n_items_i, trecall_i], self.dataset['target_nitems_trecall'][n_items_i, trecall_i], self.dataset['nontargets_nitems_trecall'][n_items_i, trecall_i])
                self.dataset['em_fits_nitems_trecall'][n_items_i, trecall_i] = params_fit

                # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
                for key in em_fits_keys:
                    fits_persubjects = [self.dataset['em_fits_subjects_nitems_trecall'][subject_i, n_items_i, trecall_i][key] for subject in np.unique(unique_subjects)]

                    self.dataset['em_fits_nitems_trecall_mean']['mean'][n_items][trecall][key] = np.mean(fits_persubjects)
                    self.dataset['em_fits_nitems_trecall_mean']['std'][n_items][trecall][key] = np.std(fits_persubjects)
                    self.dataset['em_fits_nitems_trecall_mean']['values'][n_items][trecall][key] = fits_persubjects

        ## Construct array versions of the em_fits_nitems mixture proportions, for convenience
        self.construct_arrays_em_fits()



    def construct_arrays_em_fits(self):
        unique_subjects = np.unique(self.dataset['subject'])
        unique_n_items = np.unique(self.dataset['n_items'])

        # Check if mixt_nontargets in array or not
        if 'mixt_nontargets_sum' in self.dataset['em_fits_nitems_trecall_mean']['mean'].values()[0]:
            emkeys = ['kappa', 'mixt_target', 'mixt_nontargets_sum', 'mixt_random', 'train_LL']
        else:
            emkeys = ['kappa', 'mixt_target', 'mixt_nontargets', 'mixt_random', 'train_LL']


        if 'em_fits_nitems_trecall_mean_arrays' not in self.dataset:
            self.dataset['em_fits_nitems_trecall_mean_arrays'] = dict()

            self.dataset['em_fits_subjects_nitems_trecall_arrays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_arrays'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_subjects_nitems_arrays'] = np.nan*np.empty((unique_subjects.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_mean_arrays'] = dict(mean=np.nan*np.empty((unique_n_items.size, len(emkeys))), std=np.nan*np.empty((unique_n_items.size, len(emkeys))), sem=np.nan*np.empty((unique_n_items.size, len(emkeys))))

            unique_n_items = np.unique(self.dataset['n_items'])

            self.dataset['em_fits_nitems_trecall_mean_arrays']['mean'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_mean_arrays']['std'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))
            self.dataset['em_fits_nitems_trecall_mean_arrays']['sem'] = np.nan*np.empty((unique_n_items.size, unique_n_items.size, len(emkeys)))

            for n_items_i, n_items in enumerate(unique_n_items):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['mean'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall_mean']['mean'][n_items][trecall][em_key] for em_key in emkeys])
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['std'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall_mean']['std'][n_items][trecall][em_key] for em_key in emkeys])
                    self.dataset['em_fits_nitems_trecall_mean_arrays']['sem'][n_items_i, trecall_i] = self.dataset['em_fits_nitems_trecall_mean_arrays']['std'][n_items_i, trecall_i]/np.sqrt(self.dataset['subject_size'])

                    self.dataset['em_fits_nitems_trecall_arrays'][n_items_i, trecall_i] = np.array([self.dataset['em_fits_nitems_trecall'][n_items_i, trecall_i][em_key] for em_key in emkeys])

                for subject_i, subject in enumerate(unique_subjects):
                    self.dataset['em_fits_subjects_nitems_arrays'][subject_i, n_items_i] = np.array([self.dataset['em_fits_subjects_nitems'][subject_i, n_items_i][em_key] for em_key in emkeys])


            # get some mean/std for nitems sequentially, averaging over subjects, not taking trecall into account
            self.dataset['em_fits_nitems_mean_arrays']['mean'] = np.mean(self.dataset['em_fits_subjects_nitems_arrays'], axis=0)
            self.dataset['em_fits_nitems_mean_arrays']['std'] = np.std(self.dataset['em_fits_subjects_nitems_arrays'], axis=0)
            self.dataset['em_fits_nitems_mean_arrays']['sem'] = self.dataset['em_fits_nitems_mean_arrays']['std']/np.sqrt(self.dataset['subject_size'])



    def generate_data_subject_split(self):
        '''
            Split the data to get per-subject fits

            Fix trecall so that trecall=0 last queried. trecall=1 second to last, etc...
        '''

        self.dataset['data_subject_split'] = {}
        self.dataset['data_subject_split']['nitems_space'] = np.unique(self.dataset['n_items'])
        self.dataset['data_subject_split']['subjects_space'] = np.unique(self.dataset['subject'])
        self.dataset['data_subject_split']['data_subject_nitems_trecall'] = dict()
        self.dataset['data_subject_split']['data_subject'] = dict()
        self.dataset['data_subject_split']['data_subject_largest'] = dict()
        self.dataset['data_subject_split']['subject_smallestN'] = dict()
        self.dataset['data_subject_split']['subject_largestN'] = dict()
        self.dataset['data_subject_split']['N_smallest'] = np.inf

        for subject_i, subject in enumerate(self.dataset['data_subject_split']['subjects_space']):

            self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject] = dict()

            # Find the smallest number of samples for later
            self.dataset['data_subject_split']['subject_smallestN'][subject] = np.inf

            # Create dict(subject) -> dict(nitems_space, response, target, nontargets)
            for n_items_i, n_items in enumerate(self.dataset['data_subject_split']['nitems_space']):

                self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items] = dict()

                for trecall in np.arange(1, n_items+1):

                    # Inverting indexing of trecall, to be more logical
                    trecall_i = n_items - trecall
                    # print "Splitting data up: subject %d, %d items, trecall %d, %d datapoints" % (subject, n_items, trecall, self.dataset['sizes_subject_nitems_trecall'][subject_i, n_items_i, trecall_i])

                    # Create dict(subject) -> dict(n_items) -> dict(trecall) -> dict(nitems_space, response, target, nontargets, N)
                    # Fix the trecall indexing along the way!
                    N = self.dataset['sizes_subject_nitems_trecall'][subject_i][n_items_i][trecall_i]
                    responses = self.dataset['response_subject_nitems_trecall'][subject_i][n_items_i][trecall_i]
                    targets = self.dataset['target_subject_nitems_trecall'][subject_i][n_items_i][trecall_i]
                    nontargets = self.dataset['nontargets_subject_nitems_trecall'][subject_i][n_items_i][trecall_i][..., :(n_items - 1)]
                    # stimuli in a form ready for DataGenerator
                    item_features = np.empty((N, n_items, 2))
                    item_features[..., 0] = self.dataset[
                        'item_angle_subject_nitems_trecall'][
                        subject_i, n_items_i, trecall_i][:, :n_items]
                    item_features[..., 1] = self.dataset[
                        'item_colour_subject_nitems_trecall'][
                        subject_i, n_items_i, trecall_i][:, :n_items]

                    (self.dataset[
                        'data_subject_split'][
                        'data_subject_nitems_trecall'][
                        subject][
                        n_items][
                        trecall]) = dict(
                            N=N,
                            responses=responses[:],
                            targets=targets[:],
                            nontargets=nontargets[:],
                            item_features=item_features)


        # Find the smallest number of samples for later
        self.dataset['data_subject_split']['subject_smallestN'] = np.nanmin(np.nanmin(self.dataset['sizes_subject_nitems_trecall'], axis=-1), axis=-1)
        self.dataset['data_subject_split']['N_smallest'] = int(min(
            np.min(self.dataset['data_subject_split']['subject_smallestN']),
            self.dataset['data_subject_split']['N_smallest']))

        # Now redo a run through the data, but store everything per subject, in a matrix with TxTxN' (T objects, T recall times, N_small_sub datapoints).
        # To be precise, only Tr <= T is there.
        for subject_i, subject in enumerate(self.dataset['data_subject_split']['subjects_space']):

            self.dataset['data_subject_split']['data_subject'][subject] = dict(
                # Responses: TxTxN
                responses=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_smallestN'][subject_i])),
                # Targets: TxTxN
                targets=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_smallestN'][subject_i])),
                # Nontargets: TxTxNx(Tmax-1)
                nontargets=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_smallestN'][subject_i], self.dataset['data_subject_split']['nitems_space'].max()-1))
            )

            for n_items_i, n_items in enumerate(self.dataset['data_subject_split']['nitems_space']):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    self.dataset['data_subject_split']['data_subject'][subject]['responses'][n_items_i, trecall_i] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['responses'][:self.dataset['data_subject_split']['subject_smallestN'][subject_i]]

                    self.dataset['data_subject_split']['data_subject'][subject]['targets'][n_items_i, trecall_i] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['targets'][:self.dataset['data_subject_split']['subject_smallestN'][subject_i]]

                    self.dataset['data_subject_split']['data_subject'][subject]['nontargets'][n_items_i, trecall_i, :, :(n_items - 1)] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['nontargets'][:self.dataset['data_subject_split']['subject_smallestN'][subject_i]]

        # Do the same, but try to keep as much of the data as possible
        self.dataset['data_subject_split']['subject_largestN'] = np.nanmax(np.nanmax(self.dataset['sizes_subject_nitems_trecall'], axis=-1), axis=-1)

        for subject_i, subject in enumerate(self.dataset['data_subject_split']['subjects_space']):

            self.dataset['data_subject_split']['data_subject_largest'][subject] = dict(
                # Responses: TxTxN
                responses=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_largestN'][subject_i])),
                # Targets: TxTxN
                targets=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_largestN'][subject_i])),
                # Nontargets: TxTxNx(Tmax-1)
                nontargets=np.nan*np.empty((self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['nitems_space'].size, self.dataset['data_subject_split']['subject_largestN'][subject_i], self.dataset['data_subject_split']['nitems_space'].max()-1))
            )

            for n_items_i, n_items in enumerate(self.dataset['data_subject_split']['nitems_space']):
                for trecall_i, trecall in enumerate(np.arange(1, n_items+1)):
                    # Need to recorrect trecall for this one...
                    curr_size = self.dataset['sizes_subject_nitems_trecall'][subject_i][n_items_i][n_items - trecall]

                    self.dataset['data_subject_split']['data_subject_largest'][subject]['responses'][n_items_i, trecall_i, :curr_size] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['responses']
                    self.dataset['data_subject_split']['data_subject_largest'][subject]['targets'][n_items_i, trecall_i, :curr_size] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['targets']
                    self.dataset['data_subject_split']['data_subject_largest'][subject]['nontargets'][n_items_i, trecall_i, :curr_size, :(n_items - 1)] = self.dataset['data_subject_split']['data_subject_nitems_trecall'][subject][n_items][trecall]['nontargets']


    def fit_collapsed_mixture_model(self):
        '''
            Fit the new Collapsed Mixture Model, using data created
            just above in generate_data_subject_split.

            Do:
             * One fit per subject/nitems, using trecall as T_space
             * One fit per subject/trecall, using nitems as T_space
             * One fit per subject, using the double-powerlaw on nitems/trecall

        '''
        Tmax = self.dataset['data_subject_split']['nitems_space'].max()
        Tnum = self.dataset['data_subject_split']['nitems_space'].size


        self.dataset['collapsed_em_fits_subjects_nitems'] = dict()
        self.dataset['collapsed_em_fits_nitems'] = dict()

        self.dataset['collapsed_em_fits_subjects_trecall'] = dict()
        self.dataset['collapsed_em_fits_trecall'] = dict()

        self.dataset['collapsed_em_fits_doublepowerlaw_subjects'] = dict()
        self.dataset['collapsed_em_fits_doublepowerlaw'] = dict()
        self.dataset['collapsed_em_fits_doublepowerlaw_array'] = np.nan*np.empty((Tnum, Tnum, 4))


        for subject, subject_data_dict in self.dataset['data_subject_split']['data_subject'].iteritems():
            print 'Fitting Collapsed Mixture model for subject %d' % subject

            if True:
                # Use trecall as T_space, bit weird
                for n_items_i, n_items in enumerate(self.dataset['data_subject_split']['nitems_space']):

                    print '%d nitems, using trecall as T_space' % n_items

                    params_fit = em_circularmixture_parametrickappa.fit(np.arange(1, n_items+1), subject_data_dict['responses'][n_items_i, :(n_items)], subject_data_dict['targets'][n_items_i, :(n_items)], subject_data_dict['nontargets'][n_items_i, :(n_items), :, :(n_items - 1)], debug=False)

                    self.dataset['collapsed_em_fits_subjects_nitems'].setdefault(subject, dict())[n_items] = params_fit

                # Use nitems as T_space, as a function of trecall (be careful)
                for trecall_i, trecall in enumerate(self.dataset['data_subject_split']['nitems_space']):

                    print 'trecall %d, using n_items as T_space' % trecall

                    params_fit = em_circularmixture_parametrickappa.fit(np.arange(trecall, Tmax+1), subject_data_dict['responses'][trecall_i:, trecall_i], subject_data_dict['targets'][trecall_i:, trecall_i], subject_data_dict['nontargets'][trecall_i:, trecall_i], debug=False)

                    self.dataset['collapsed_em_fits_subjects_trecall'].setdefault(subject, dict())[trecall] = params_fit

            # Now do the correct fit, with double powerlaw on nitems+trecall
            print 'Double powerlaw fit'

            params_fit_double = (
                em_circularmixture_parametrickappa_doublepowerlaw.fit(
                    self.dataset['data_subject_split']['nitems_space'],
                    subject_data_dict['responses'],
                    subject_data_dict['targets'],
                    subject_data_dict['nontargets'],
                    debug=False))
            self.dataset['collapsed_em_fits_doublepowerlaw_subjects'][subject] = params_fit_double


        if True:
            ## Now compute mean/std collapsed_em_fits_nitems
            self.dataset['collapsed_em_fits_nitems']['mean'] = dict()
            self.dataset['collapsed_em_fits_nitems']['std'] = dict()
            self.dataset['collapsed_em_fits_nitems']['sem'] = dict()
            self.dataset['collapsed_em_fits_nitems']['values'] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for n_items_i, n_items in enumerate(self.dataset['data_subject_split']['nitems_space']):
                for key in emfits_keys:
                    values_allsubjects = [self.dataset['collapsed_em_fits_subjects_nitems'][subject][n_items][key] for subject in self.dataset['data_subject_split']['subjects_space']]

                    self.dataset['collapsed_em_fits_nitems']['mean'].setdefault(n_items, dict())[key] = np.mean(values_allsubjects, axis=0)
                    self.dataset['collapsed_em_fits_nitems']['std'].setdefault(n_items, dict())[key] = np.std(values_allsubjects, axis=0)
                    self.dataset['collapsed_em_fits_nitems']['sem'].setdefault(n_items, dict())[key] = self.dataset['collapsed_em_fits_nitems']['std'][n_items][key]/np.sqrt(self.dataset['data_subject_split']['subjects_space'].size)
                    self.dataset['collapsed_em_fits_nitems']['values'].setdefault(n_items, dict())[key] = values_allsubjects

            ## Same for the other ones
            self.dataset['collapsed_em_fits_trecall']['mean'] = dict()
            self.dataset['collapsed_em_fits_trecall']['std'] = dict()
            self.dataset['collapsed_em_fits_trecall']['sem'] = dict()
            self.dataset['collapsed_em_fits_trecall']['values'] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for trecall_i, trecall in enumerate(self.dataset['data_subject_split']['nitems_space']):
                for key in emfits_keys:
                    values_allsubjects = [self.dataset['collapsed_em_fits_subjects_trecall'][subject][trecall][key] for subject in self.dataset['data_subject_split']['subjects_space']]

                    self.dataset['collapsed_em_fits_trecall']['mean'].setdefault(trecall, dict())[key] = np.mean(values_allsubjects, axis=0)
                    self.dataset['collapsed_em_fits_trecall']['std'].setdefault(trecall, dict())[key] = np.std(values_allsubjects, axis=0)
                    self.dataset['collapsed_em_fits_trecall']['sem'].setdefault(trecall, dict())[key] = self.dataset['collapsed_em_fits_trecall']['std'][trecall][key]/np.sqrt(self.dataset['data_subject_split']['subjects_space'].size)
                    self.dataset['collapsed_em_fits_trecall']['values'].setdefault(trecall, dict())[key] = values_allsubjects

        # Collapsed full double powerlaw model across subjects
        self.dataset['collapsed_em_fits_doublepowerlaw']['mean'] = dict()
        self.dataset['collapsed_em_fits_doublepowerlaw']['std'] = dict()
        self.dataset['collapsed_em_fits_doublepowerlaw']['sem'] = dict()
        self.dataset['collapsed_em_fits_doublepowerlaw']['values'] = dict()

        # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
        emfits_keys = params_fit_double.keys()
        for key in emfits_keys:
            values_allsubjects = [self.dataset['collapsed_em_fits_doublepowerlaw_subjects'][subject][key] for subject in self.dataset['data_subject_split']['subjects_space']]

            self.dataset['collapsed_em_fits_doublepowerlaw']['mean'][key] = np.mean(values_allsubjects, axis=0)
            self.dataset['collapsed_em_fits_doublepowerlaw']['std'][key] = np.std(values_allsubjects, axis=0)
            self.dataset['collapsed_em_fits_doublepowerlaw']['sem'][key] = self.dataset['collapsed_em_fits_doublepowerlaw']['std'][key]/np.sqrt(self.dataset['data_subject_split']['subjects_space'].size)
            self.dataset['collapsed_em_fits_doublepowerlaw']['values'][key] = values_allsubjects

        # Construct some easy arrays to compare the fit to the dataset
        self.dataset['collapsed_em_fits_doublepowerlaw_array'][..., 0] = self.dataset['collapsed_em_fits_doublepowerlaw']['mean']['kappa']
        self.dataset['collapsed_em_fits_doublepowerlaw_array'][..., 1] = self.dataset['collapsed_em_fits_doublepowerlaw']['mean']['mixt_target_tr']
        self.dataset['collapsed_em_fits_doublepowerlaw_array'][..., 2] = self.dataset['collapsed_em_fits_doublepowerlaw']['mean']['mixt_nontargets_tr']
        self.dataset['collapsed_em_fits_doublepowerlaw_array'][..., 3] = self.dataset['collapsed_em_fits_doublepowerlaw']['mean']['mixt_random_tr']



