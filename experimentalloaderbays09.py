'''
    Small class system to simplify the process of loading Experimental datasets
'''

import numpy as np

import utils

from experimentalloader import ExperimentalLoader

class ExperimentalLoaderBays09(ExperimentalLoader):
    """
        Bays2009 dataset
    """
    def __init__(self, dataset_description):
        super(self.__class__, self).__init__(dataset_description)


    def preprocess(self, parameters):
        '''
        The Bays2009 dataset is completely different...
        Some preprocessing is already done, so just do the plots we care about
        '''

        # Make some aliases
        self.dataset['n_items'] = self.dataset['N'].astype(int)
        self.dataset['n_items_size'] = np.unique(self.dataset['n_items']).size
        self.dataset['subject'] = self.dataset['subject'].astype(int)
        self.dataset['subject_size'] = np.unique(self.dataset['subject']).size
        self.dataset['error'] = self.dataset['E']
        self.dataset['response'] = self.dataset['Y']
        self.dataset['item_angle'] = self.dataset['X']
        self.dataset['item_colour'] = self.dataset['A'] - np.pi
        self.dataset['probe'] = np.zeros(self.dataset['response'].shape, dtype=int)
        self.dataset['errors_nitems'] = np.empty(self.dataset['n_items_size'], dtype=np.object)
        self.dataset['errors_nontarget_nitems'] = np.empty(self.dataset['n_items_size'], dtype=np.object)
        self.dataset['errors_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size']), dtype=np.object)
        self.dataset['errors_nontarget_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size']), dtype=np.object)
        self.dataset['vtest_nitems'] = np.empty(self.dataset['n_items_size'])*np.nan
        self.dataset['precision_subject_nitems_bays'] = np.nan*np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))
        self.dataset['precision_subject_nitems_theo'] = np.nan*np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))
        self.dataset['precision_subject_nitems_theo_nochance'] = np.nan*np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))
        self.dataset['precision_subject_nitems_bays_notreatment'] = np.nan*np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))



        # Fit mixture model
        if parameters.get('fit_mixture_model', False):
            self.fit_mixture_model_cached(caching_save_filename=parameters.get('mixture_model_cache', None))


        # Compute errors and Vtests
        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            for subject_i, subject in enumerate(np.unique(self.dataset['subject'])):
                # Data per subject
                ids_filtered = (self.dataset['subject'] == subject).flatten() & (self.dataset['n_items'] == n_items).flatten()

                self.dataset['errors_subject_nitems'][subject_i, n_items_i] = self.dataset['error'][ids_filtered, 0]
                self.dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i] = self.dataset['error'][ids_filtered, 1:n_items]

                # Precisions
                # Compute the precision
                self.dataset['precision_subject_nitems_bays'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=True)
                self.dataset['precision_subject_nitems_theo'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=False, use_wrong_precision=False)
                self.dataset['precision_subject_nitems_theo_nochance'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=True, correct_orientation=False, use_wrong_precision=False)
                self.dataset['precision_subject_nitems_bays_notreatment'][subject_i, n_items_i] = self.compute_precision(self.dataset['errors_subject_nitems'][subject_i, n_items_i], remove_chance_level=False, correct_orientation=False, use_wrong_precision=True)

            # Data collapsed accross subjects
            ids_filtered = (self.dataset['n_items'] == n_items).flatten()

            self.dataset['errors_nitems'][n_items_i] = self.dataset['error'][ids_filtered, 0]
            self.dataset['errors_nontarget_nitems'][n_items_i] = self.dataset['error'][ids_filtered, 1:n_items]

            if n_items > 1:
                self.dataset['vtest_nitems'][n_items_i] = utils.V_test(utils.dropnan(self.dataset['errors_nontarget_nitems'][n_items_i]).flatten())['pvalue']

        # Save item in a nice format for the model fit
        self.generate_data_to_fit()

        # Save data in a better format to fit the new collapsed mixture model
        self.generate_data_subject_split()

        # Fit the new Collapsed mixture model
        if parameters.get('fit_mixture_model', False):
            self.fit_collapsed_mixture_model_cached(caching_save_filename=parameters.get('collapsed_mixture_model_cache', None))

        # Perform Bootstrap analysis if required
        if parameters.get('should_compute_bootstrap', False):
            self.compute_bootstrap_cached(
                caching_save_filename=parameters.get('bootstrap_cache', None),
                nb_bootstrap_samples=parameters.get('nb_bootstrap_samples', 1000))

        # Do per subject and nitems, get average histogram
        self.compute_average_histograms()


