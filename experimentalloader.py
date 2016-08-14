'''
    Small class system to simplify the process of loading Experimental datasets
'''

import numpy as np
import scipy.io as sio
# import matplotlib.patches as plt_patches
# import matplotlib.gridspec as plt_grid
import os
import os.path
import cPickle as pickle
# import bottleneck as bn
import em_circularmixture_allitems_uniquekappa as em_circmixtmodel
import em_circularmixture_parametrickappa as em_circmixtmodel_parametric

import utils


class ExperimentalLoader(object):
    """
        Loads an experiment.

        Will define a few functions, should be overriden.
    """
    def __init__(self, dataset_description):
        self.load_dataset(dataset_description)

    def preprocess(self, parameters):
        raise NotImplementedError('Should be overriden')

    def load_dataset(self, dataset_description={}):
        '''
            Load dataset file
        '''
        # Add datadir
        self.datadir = dataset_description.get('datadir', '')
        self.filename = os.path.join(self.datadir, dataset_description['filename'])

        # Load everything
        self.dataset = sio.loadmat(self.filename, mat_dtype=True)

        # Set its name
        self.dataset['name'] = dataset_description['name']

        # Specific operations, for different types of datasets
        self.preprocess(dataset_description['parameters'])

        return self.dataset


    def convert_wrap(self, keys_to_convert=['item_angle', 'probe_angle', 'response', 'error', 'err'], multiply_factor=2., max_angle=np.pi):
        '''
            Takes a dataset and a list of keys. Each data associated with these keys will be converted to radian,
                and wrapped in a [-max_angle, max_angle] interval
        '''
        for key in keys_to_convert:
            if key in self.dataset:
                self.dataset[key] = utils.wrap_angles(np.deg2rad(multiply_factor*self.dataset[key]), bound=max_angle)


    def compute_all_errors(self):
        '''
            Will compute the error between the response and all possible items
        '''

        # Get the difference between angles
        # Should also wrap it around
        self.dataset['errors_all'] = utils.wrap_angles(self.dataset['item_angle'] - self.dataset['response'], bound=np.pi)



    def compute_precision(self, errors, remove_chance_level=True, correct_orientation=False, use_wrong_precision=True):
        '''
            Compute the precision (1./circ_std**2). Remove the chance level if desired.
        '''

        # if correct_orientation:
        #     # Correct for the fact that bars are modelled in [0, pi] and not [0, 2pi]
        #     errors = errors.copy()*2.0

        # Circular standard deviation estimate
        error_std_dev_error = utils.angle_circular_std_dev(errors)

        # Precision
        if use_wrong_precision:
            precision = 1./error_std_dev_error
        else:
            precision = 1./error_std_dev_error**2.

        if remove_chance_level:
            # Remove the chance level
            precision -= utils.compute_precision_chance(errors.size)

        if correct_orientation:
            # The obtained precision is for half angles, correct it
            precision *= 2.

        return precision


    def fit_mixture_model_cached(self, caching_save_filename=None, saved_keys=['em_fits', 'em_fits_nitems', 'em_fits_subjects_nitems', 'em_fits_nitems_arrays', 'em_fits_subjects_nitems_arrays']):
        '''
            Fit the mixture model onto classical responses/item_angle values

            If caching_save_filename is not None:
            - Will try to open the file provided and use 'em_fits', 'em_fits_subjects_nitems' and 'em_fits_nitems' instead of computing them.
            - If file does not exist, compute and save it.
        '''

        should_fit_model = True
        save_caching_file = False

        if caching_save_filename is not None:
            caching_save_filename = os.path.join(self.datadir, caching_save_filename)

            if os.path.exists(caching_save_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_save_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        self.dataset.update(cached_data)
                        should_fit_model = False
                        print "reloaded mixture model from cache", caching_save_filename

                except:
                    print "Error while loading ", caching_save_filename, "falling back to computing the EM fits"
            else:
                # No file, create it after everything is computed
                save_caching_file = True


        if should_fit_model:
            self.fit_mixture_model()

        if save_caching_file:
            try:
                with open(caching_save_filename, 'w') as filecache_out:
                    data_em = dict((key, self.dataset[key]) for key in saved_keys)

                    pickle.dump(data_em, filecache_out, protocol=2)

            except IOError:
                print "Error writing out to caching file ", caching_save_filename


    def fit_mixture_model(self):
        N = self.dataset['probe'].size

        # Initialize empty arrays and dicts
        self.dataset['em_fits'] = dict(kappa=np.empty(N),
                                       mixt_target=np.empty(N),
                                       mixt_nontarget=np.empty(N),
                                       mixt_random=np.empty(N),
                                       resp_target=np.empty(N),
                                       resp_nontarget=np.empty(N),
                                       resp_random=np.empty(N),
                                       train_LL=np.empty(N),
                                       test_LL=np.empty(N)
                                       )
        for key in self.dataset['em_fits']:
            self.dataset['em_fits'][key].fill(np.nan)
        self.dataset['target'] = np.empty(N)
        self.dataset['em_fits_subjects_nitems'] = dict()
        for subject in np.unique(self.dataset['subject']):
            self.dataset['em_fits_subjects_nitems'][subject] = dict()
        self.dataset['em_fits_nitems'] = dict(mean=dict(), std=dict(), values=dict())

        # Compute mixture model fits per n_items and per subject
        for n_items in np.unique(self.dataset['n_items']):
            for subject in np.unique(self.dataset['subject']):
                ids_filter = (self.dataset['subject'] == subject).flatten() & \
                             (self.dataset['n_items'] == n_items).flatten()
                print "Fit mixture model, %d items, subject %d, %d datapoints" % (subject, n_items, np.sum(ids_filter))

                self.dataset['target'][ids_filter] = self.dataset['item_angle'][ids_filter, 0]

                params_fit = em_circmixtmodel.fit(
                    self.dataset['response'][ids_filter, 0],
                    self.dataset['item_angle'][ids_filter, 0],
                    self.dataset['item_angle'][ids_filter, 1:]
                )
                params_fit['mixt_nontargets_sum'] = np.sum(
                    params_fit['mixt_nontargets']
                )

                resp = em_circmixtmodel.compute_responsibilities(
                    self.dataset['response'][ids_filter, 0],
                    self.dataset['item_angle'][ids_filter, 0],
                    self.dataset['item_angle'][ids_filter, 1:],
                    params_fit
                )

                self.dataset['em_fits']['kappa'][ids_filter] = \
                    params_fit['kappa']
                self.dataset['em_fits']['mixt_target'][ids_filter] = \
                    params_fit['mixt_target']
                self.dataset['em_fits']['mixt_nontarget'][ids_filter] = \
                    params_fit['mixt_nontargets_sum']
                self.dataset['em_fits']['mixt_random'][ids_filter] = \
                    params_fit['mixt_random']
                self.dataset['em_fits']['resp_target'][ids_filter] = \
                    resp['target']
                self.dataset['em_fits']['resp_nontarget'][ids_filter] = \
                    np.sum(resp['nontargets'], axis=1)
                self.dataset['em_fits']['resp_random'][ids_filter] = \
                    resp['random']
                self.dataset['em_fits']['train_LL'][ids_filter] = \
                    params_fit['train_LL']

                self.dataset['em_fits_subjects_nitems'][subject][n_items] = params_fit


            ## Now compute mean/std em_fits per n_items
            self.dataset['em_fits_nitems']['mean'][n_items] = dict()
            self.dataset['em_fits_nitems']['std'][n_items] = dict()
            self.dataset['em_fits_nitems']['values'][n_items] = dict()

            # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
            emfits_keys = params_fit.keys()
            for key in emfits_keys:
                values_allsubjects = [self.dataset['em_fits_subjects_nitems'][subject][n_items][key] for subject in np.unique(self.dataset['subject'])]

                self.dataset['em_fits_nitems']['mean'][n_items][key] = np.mean(values_allsubjects)
                self.dataset['em_fits_nitems']['std'][n_items][key] = np.std(values_allsubjects)
                self.dataset['em_fits_nitems']['values'][n_items][key] = values_allsubjects

        ## Construct array versions of the em_fits_nitems mixture proportions, for convenience
        self.construct_arrays_em_fits()



    def construct_arrays_em_fits(self):
        fits_keys = ['kappa', 'mixt_target', 'mixt_nontargets_sum',
                     'mixt_random']

        if 'em_fits_nitems_arrays' not in self.dataset:
            self.dataset['em_fits_nitems_arrays'] = dict()

            # Check if mixt_nontargets is array or not
            if 'mixt_nontargets_sum' in self.dataset['em_fits_nitems']['mean'].values()[0]:
                self.dataset['em_fits_nitems_arrays']['mean'] = np.array(
                    [[self.dataset['em_fits_nitems']['mean'][item][em_key]
                        for item in np.unique(self.dataset['n_items'])]
                        for em_key in fits_keys]
                )
                self.dataset['em_fits_nitems_arrays']['std'] = np.array(
                    [[self.dataset['em_fits_nitems']['std'][item][em_key]
                        for item in np.unique(self.dataset['n_items'])]
                        for em_key in fits_keys])
            else:
                self.dataset['em_fits_nitems_arrays']['mean'] = np.array(
                    [[self.dataset['em_fits_nitems']['mean'][item][em_key]
                        for item in np.unique(self.dataset['n_items'])]
                        for em_key in fits_keys])
                self.dataset['em_fits_nitems_arrays']['std'] = np.array(
                    [[self.dataset['em_fits_nitems']['std'][item][em_key]
                        for item in np.unique(self.dataset['n_items'])]
                        for em_key in fits_keys])

        if 'sem' not in self.dataset['em_fits_nitems_arrays']:
            self.dataset['em_fits_nitems_arrays']['sem'] = self.dataset['em_fits_nitems_arrays']['std']/np.sqrt(self.dataset['subject_size'])

        if 'em_fits_subjects_nitems_arrays' not in self.dataset:
            self.dataset['em_fits_subjects_nitems_arrays'] = \
                np.empty((self.dataset['subject_size'],
                          self.dataset['n_items_size'],
                          len(fits_keys)
                          ))

            for subject_i, subject in enumerate(np.unique(self.dataset['subject'])):
                for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
                    self.dataset['em_fits_subjects_nitems_arrays'][subject_i, n_items_i] = \
                        np.array([self.dataset['em_fits_subjects_nitems']
                                              [subject][n_items][key]
                                  for key in fits_keys
                                  ])



    def compute_bootstrap_cached(self,
                                 caching_save_filename=None,
                                 nb_bootstrap_samples=1000):
        '''
            Compute bootstrap estimates per subject/nitems.

            If caching_save_filename is not None:
            - Will try to open the file provided and use 'bootstrap_subject_nitems', 'bootstrap_nitems' and 'bootstrap_nitems_pval' instead of computing them.
            - If file does not exist, compute and save it.
        '''

        should_compute_bootstrap = True
        save_caching_file = False

        if caching_save_filename is not None:
            caching_save_filename = os.path.join(self.datadir, caching_save_filename)

            if os.path.exists(caching_save_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_save_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        self.dataset.update(cached_data)
                        should_compute_bootstrap = False

                except IOError:
                    print "Error while loading ", caching_save_filename, "falling back to computing the EM fits"
            else:
                # No file, create it after everything is computed
                save_caching_file = True


        if should_compute_bootstrap:
            self.compute_bootstrap(nb_bootstrap_samples=1000)


        if save_caching_file:
            try:
                with open(caching_save_filename, 'w') as filecache_out:
                    cached_data = dict((key, self.dataset[key]) for key in ['bootstrap_subject_nitems', 'bootstrap_nitems', 'bootstrap_nitems_pval', 'bootstrap_subject_nitems_pval'])

                    pickle.dump(cached_data, filecache_out, protocol=2)

            except IOError:
                print "Error writing out to caching file ", caching_save_filename


    def compute_bootstrap(self, nb_bootstrap_samples=1000):
        print "Computing bootstrap..."

        self.dataset['bootstrap_nitems_pval'] = np.nan*np.empty(self.dataset['n_items_size'])
        self.dataset['bootstrap_nitems'] = np.empty(self.dataset['n_items_size'], dtype=np.object)
        self.dataset['bootstrap_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size']), dtype=np.object)
        self.dataset['bootstrap_subject_nitems_pval'] = np.nan*np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))


        for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
            if n_items > 1:
                for subject_i, subject in enumerate(np.unique(self.dataset['subject'])):
                    print "Nitems %d, subject %d" % (n_items, subject)

                    # Bootstrap per subject and nitems
                    ids_filter = (self.dataset['subject'] == subject).flatten() & (self.dataset['n_items'] == n_items).flatten()

                    # Compute bootstrap if required

                    bootstrap = em_circmixtmodel.bootstrap_nontarget_stat(self.dataset['response'][ids_filter, 0], self.dataset['item_angle'][ids_filter, 0], self.dataset['item_angle'][ids_filter, 1:n_items], nb_bootstrap_samples=nb_bootstrap_samples)
                    self.dataset['bootstrap_subject_nitems'][subject_i, n_items_i] = bootstrap
                    self.dataset['bootstrap_subject_nitems_pval'][subject_i, n_items_i] = bootstrap['p_value']

                    print self.dataset['bootstrap_subject_nitems_pval'][:, n_items_i]

                print "Nitems %d, all subjects" % (n_items)

                # Data collapsed accross subjects
                ids_filter = (self.dataset['n_items'] == n_items).flatten()

                bootstrap = em_circmixtmodel.bootstrap_nontarget_stat(self.dataset['response'][ids_filter, 0], self.dataset['item_angle'][ids_filter, 0], self.dataset['item_angle'][ids_filter, 1:n_items], nb_bootstrap_samples=nb_bootstrap_samples)
                self.dataset['bootstrap_nitems'][n_items_i] = bootstrap
                self.dataset['bootstrap_nitems_pval'][n_items_i] = bootstrap['p_value']

                print self.dataset['bootstrap_nitems_pval']


    def generate_data_to_fit(self):
        self.dataset['data_to_fit'] = {}
        self.dataset['data_to_fit']['n_items'] = np.unique(self.dataset['n_items'])
        self.dataset['data_to_fit']['N_smallest'] = np.inf

        for n_items in self.dataset['data_to_fit']['n_items']:
            ids_n_items = (self.dataset['n_items'] == n_items).flatten()

            if n_items not in self.dataset['data_to_fit']:
                self.dataset['data_to_fit'][n_items] = {}
                self.dataset['data_to_fit'][n_items]['N'] = np.sum(ids_n_items)
                self.dataset['data_to_fit'][n_items]['probe'] = np.unique(self.dataset['probe'][ids_n_items])
                self.dataset['data_to_fit'][n_items]['item_features'] = np.empty((self.dataset['data_to_fit'][n_items]['N'], n_items, 2))
                self.dataset['data_to_fit'][n_items]['response'] = np.empty((self.dataset['data_to_fit'][n_items]['N'], 1))
                self.dataset['data_to_fit']['N_smallest'] = min(self.dataset['data_to_fit']['N_smallest'], self.dataset['data_to_fit'][n_items]['N'])

            self.dataset['data_to_fit'][n_items]['item_features'][..., 0] = self.dataset['item_angle'][ids_n_items, :n_items]
            self.dataset['data_to_fit'][n_items]['item_features'][..., 1] = self.dataset['item_colour'][ids_n_items, :n_items]
            self.dataset['data_to_fit'][n_items]['response'] = self.dataset['response'][ids_n_items].flatten()


    def generate_data_subject_split(self):
        '''
            Split the data to get per-subject fits:

             - response, target, nontargets per subject and per n_item
             - nitems_space, response, target, nontargets per subject
        '''

        self.dataset['data_subject_split'] = {}
        self.dataset['data_subject_split']['nitems_space'] = np.unique(self.dataset['n_items'])
        self.dataset['data_subject_split']['subjects_space'] = np.unique(self.dataset['subject'])
        self.dataset['data_subject_split']['data_subject_nitems'] = dict()
        self.dataset['data_subject_split']['data_subject'] = dict()
        self.dataset['data_subject_split']['subject_smallestN'] = dict()

        for subject in np.unique(self.dataset['data_subject_split']['subjects_space']):

            # Find the smallest number of samples for later
            self.dataset['data_subject_split']['subject_smallestN'][subject] = np.inf

            # Create dict(subject) -> dict(nitems_space, response, target, nontargets)
            for n_items in np.unique(self.dataset['data_subject_split']['nitems_space']):

                ids_filtered = (self.dataset['subject'] == subject).flatten() & (self.dataset['n_items'] == n_items).flatten()
                # print "Splitting data up: subject %d, %d items, %d datapoints" % (subject, n_items, np.sum(ids_filtered))

                # Create dict(subject) -> dict(n_items) -> dict(nitems_space, response, target, nontargets, N)
                self.dataset['data_subject_split']['data_subject_nitems'].setdefault(subject, dict())[n_items] = \
                    dict(N=np.sum(ids_filtered),
                         response=self.dataset['response'][ids_filtered, 0],
                         target=self.dataset['item_angle'][ids_filtered, 0],
                         nontargets=self.dataset['item_angle'][ids_filtered, 1:n_items],
                         item_features=self.dataset['item_angle'][ids_filtered, :n_items]
                         )

                # Find the smallest number of samples for later
                self.dataset['data_subject_split']['subject_smallestN'][subject] = min(self.dataset['data_subject_split']['subject_smallestN'][subject], np.sum(ids_filtered))

        # Now redo a run through the data, but store everything per subject, in a matrix with TxN (T objects, N datapoints).
        for subject in np.unique(self.dataset['data_subject_split']['subjects_space']):

            self.dataset['data_subject_split']['data_subject'][subject] = dict(
                # Responses: TxN
                responses=np.nan*np.empty(
                    (self.dataset['data_subject_split']['nitems_space'].size,
                     self.dataset['data_subject_split']['subject_smallestN'][subject])),
                # Targets: TxN
                targets=np.nan*np.empty(
                    (self.dataset['data_subject_split']['nitems_space'].size,
                     self.dataset['data_subject_split']['subject_smallestN'][subject])),
                # Nontargets: TxNx(Tmax-1)
                nontargets=np.nan*np.empty(
                    (self.dataset['data_subject_split']['nitems_space'].size,
                     self.dataset['data_subject_split']['subject_smallestN'][subject],
                     self.dataset['data_subject_split']['nitems_space'].max()-1))
            )

            for n_items_i, n_items in enumerate(np.unique(self.dataset['data_subject_split']['nitems_space'])):

                ids_filtered = (self.dataset['subject'] == subject).flatten() & (self.dataset['n_items'] == n_items).flatten()

                # Assign data to:
                # dict(subject) -> dict(responses TxN, targets TxN, nontargets TxNx(T-1) )
                self.dataset['data_subject_split']['data_subject'][subject]['responses'][n_items_i] = self.dataset['response'][ids_filtered, 0][:self.dataset['data_subject_split']['subject_smallestN'][subject]]
                self.dataset['data_subject_split']['data_subject'][subject]['targets'][n_items_i] = self.dataset['item_angle'][ids_filtered, 0][:self.dataset['data_subject_split']['subject_smallestN'][subject]]
                self.dataset['data_subject_split']['data_subject'][subject]['nontargets'][n_items_i, :, :(n_items-1)] = self.dataset['item_angle'][ids_filtered, 1:n_items][:self.dataset['data_subject_split']['subject_smallestN'][subject]]


    def fit_collapsed_mixture_model_cached(self, caching_save_filename=None, saved_keys=['collapsed_em_fits_subjects', 'collapsed_em_fits']):

        should_fit_model = True
        save_caching_file = False

        if caching_save_filename is not None:
            caching_save_filename = os.path.join(self.datadir, caching_save_filename)

            if os.path.exists(caching_save_filename):
                # Got file, open it and try to use its contents
                try:
                    with open(caching_save_filename, 'r') as file_in:
                        # Load and assign values
                        cached_data = pickle.load(file_in)
                        self.dataset.update(cached_data)
                        should_fit_model = False
                        print "reloaded collapsed mixture model from cache", caching_save_filename

                except:
                    print "Error while loading ", caching_save_filename, "falling back to computing the Collapsed EM fits"
            else:
                # No file, create it after everything is computed
                save_caching_file = True


        if should_fit_model:
            self.fit_collapsed_mixture_model()

        if save_caching_file:
            try:
                with open(caching_save_filename, 'w') as filecache_out:
                    data_em = dict((key, self.dataset[key]) for key in saved_keys)

                    pickle.dump(data_em, filecache_out, protocol=2)

            except IOError:
                print "Error writing out to caching file ", caching_save_filename


    def fit_collapsed_mixture_model(self):
        '''
            Fit the new Collapsed Mixture Model, using data created
            just above in generate_data_subject_split.

            One fit per subject, obtain parametric estimates of kappa.

        '''

        self.dataset['collapsed_em_fits_subjects'] = dict()
        self.dataset['collapsed_em_fits'] = dict()

        for subject, subject_data_dict in self.dataset['data_subject_split']['data_subject'].iteritems():
            print 'Fitting Collapsed Mixture model for subject %d' % subject

            # Bug here, fit is not using the good dimensionality for the number of Nontarget angles...
            params_fit = em_circmixtmodel_parametric.fit(
                self.dataset['data_subject_split']['nitems_space'],
                subject_data_dict['responses'],
                subject_data_dict['targets'],
                subject_data_dict['nontargets'],
                debug=False
            )

            self.dataset['collapsed_em_fits_subjects'][subject] = params_fit

        ## Now compute mean/std collapsed_em_fits

        self.dataset['collapsed_em_fits']['mean'] = dict()
        self.dataset['collapsed_em_fits']['std'] = dict()
        self.dataset['collapsed_em_fits']['sem'] = dict()
        self.dataset['collapsed_em_fits']['values'] = dict()

        # Need to extract the values for a subject/nitems pair, for all keys of em_fits. Annoying dictionary indexing needed
        emfits_keys = params_fit.keys()
        for key in emfits_keys:
            values_allsubjects = [self.dataset['collapsed_em_fits_subjects'][subject][key] for subject in self.dataset['data_subject_split']['subjects_space']]

            self.dataset['collapsed_em_fits']['mean'][key] = np.mean(values_allsubjects, axis=0)
            self.dataset['collapsed_em_fits']['std'][key] = np.std(values_allsubjects, axis=0)
            self.dataset['collapsed_em_fits']['sem'][key] = self.dataset['collapsed_em_fits']['std'][key]/np.sqrt(self.dataset['data_subject_split']['subjects_space'].size)
            self.dataset['collapsed_em_fits']['values'][key] = values_allsubjects


    def compute_average_histograms(self):
        '''
            Do per subject and nitems, get average histogram
        '''

        angle_space = np.linspace(-np.pi, np.pi, 51)

        self.dataset['hist_cnts_target_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size'], angle_space.size - 1))*np.nan
        self.dataset['hist_cnts_nontarget_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size'], angle_space.size - 1))*np.nan
        self.dataset['pvalue_nontarget_subject_nitems'] = np.empty((self.dataset['subject_size'], self.dataset['n_items_size']))*np.nan

        for subject_i, subject in enumerate(np.unique(self.dataset['subject'])):
            for n_items_i, n_items in enumerate(np.unique(self.dataset['n_items'])):
                self.dataset['hist_cnts_target_subject_nitems'][subject_i, n_items_i], x, bins = utils.histogram_binspace(utils.dropnan(self.dataset['errors_subject_nitems'][subject_i, n_items_i]), bins=angle_space, norm='density')

                self.dataset['hist_cnts_nontarget_subject_nitems'][subject_i, n_items_i], x, bins = utils.histogram_binspace(utils.dropnan(self.dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i]), bins=angle_space, norm='density')

                if n_items > 1:
                    self.dataset['pvalue_nontarget_subject_nitems'][subject_i, n_items_i] = utils.V_test(utils.dropnan(self.dataset['errors_nontarget_subject_nitems'][subject_i, n_items_i]).flatten())['pvalue']

        self.dataset['hist_cnts_target_nitems_stats'] = dict(mean=np.mean(self.dataset['hist_cnts_target_subject_nitems'], axis=0), std=np.std(self.dataset['hist_cnts_target_subject_nitems'], axis=0), sem=np.std(self.dataset['hist_cnts_target_subject_nitems'], axis=0)/np.sqrt(self.dataset['subject_size']))

        self.dataset['hist_cnts_nontarget_nitems_stats'] = dict(mean=np.mean(self.dataset['hist_cnts_nontarget_subject_nitems'], axis=0), std=np.std(self.dataset['hist_cnts_nontarget_subject_nitems'], axis=0), sem=np.std(self.dataset['hist_cnts_nontarget_subject_nitems'], axis=0)/np.sqrt(self.dataset['subject_size']))






