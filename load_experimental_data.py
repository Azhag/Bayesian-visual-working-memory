##
# Load experimental data in python, because Matlab sucks ass.
##
import numpy as np
import scipy.io as sio
import pylab as plt
import os
# import bottleneck as bn
import em_circularmixture
import pandas as pd

def wrap(angles, max_angle = np.pi):
    '''
        Wrap angles in a -max_angle:max_angle space
    '''

    angles = np.mod(angles + max_angle, 2*max_angle) - max_angle
    
    return angles


def convert_wrap(dataset, keys_to_convert = ['item_angle', 'probe_angle', 'response', 'error', 'err'], max_angle=np.pi/2.):
    '''
        Takes a dataset and a list of keys. Each data associated with these keys will be converted to radian, 
            and wrapped in a [-max_angle, max_angle] interval
    '''
    for key in keys_to_convert:
        if key in dataset:
            dataset[key] = wrap(np.deg2rad(dataset[key]), max_angle = max_angle)


def preprocess_simultaneous(dataset, parameters):
    '''
        For simultaneous datasets, there is no 'probe' key, the first item in 'item_angle' is the target one.
        The 'error' key is missing and called 'err', which we will correct as well.
    '''

    # Extract parameters
    # params_lists: List( (param_name, default_value) )
    params_list = [('convert_radians', True), ('correct_probe', True)]
    for curr_param in params_list:
        if curr_param[0] in parameters:
            # Set to provided value
            exec(curr_param[0] +" = parameters['" + curr_param[0] + "']")
        else:
            # Default value
            exec(curr_param[0] +" = " + str(curr_param[1]))

    # Rename the error field
    dataset['error'] = dataset['err']
    del dataset['err']

    # Assign probe field correctly
    dataset['probe'] = np.zeros(dataset['error'].shape, dtype= int)

    # Convert everything to radians, spanning a -np.pi/2:np.pi
    if convert_radians:
        convert_wrap(dataset)

    # Compute additional errors, between the response and all items
    compute_all_errors(dataset)

    # Create arrays per subject
    create_subject_arrays(dataset)

    # Fit the mixture model, and save the responsibilities per datapoint.
    # em_circularmixture.fit()




def preprocess_sequential(dataset, parameters):
    '''
        For sequential datasets, need to convert to radians and correct the probe indexing.
    '''
    # Extract parameters
    # params_lists: List( (param_name, default_value) )
    params_list = [('convert_radians', True), ('correct_probe', True)]
    for curr_param in params_list:
        if curr_param[0] in parameters:
            # Set to provided value
            exec(curr_param[0] + " = parameters['" + curr_param[0] + "']")
        else:
            # Default value
            exec(curr_param[0] + " = " + str(curr_param[1]))

    
    # Convert everything to radians, spanning a -np.pi/2:np.pi
    if convert_radians:
        convert_wrap(dataset)

    # Correct the probe field, Matlab format for indices...
    if correct_probe and 'probe' in dataset:
        dataset['probe'] = dataset['probe'].astype(int)
        dataset['probe'] -= 1

    # Compute additional errors, between the response and all items
    compute_all_errors(dataset)

    # Create arrays per subject
    create_subject_arrays(dataset)

    # Fit the mixture model, and save the responsibilities per datapoint.
    # em_circularmixture.fit()



def preprocess_doublerecall(dataset, parameters):
    '''
        This is the dataset where both colour and orientation can be recalled.
        - There are two groups of subjects, either with 6 or 3 items shown (no intermediates...). Stored in 'n_items'
        - Subjects recalled either colour or orientation, per blocks. Stored in 'cond'
        - Subject report their confidence, which is cool.

        Things to change:
        - 'item_location' really contains 'item_angle'...
        - item_location and probe_location should be wrapped back into -pi:pi.
        - Should compute the errors.
    '''

    # Make some aliases
    dataset['item_angle'] = dataset['item_location']
    dataset['probe_angle'] = dataset['probe_location']
    dataset['probe'] = np.ones(dataset['probe_angle'].shape, dtype= int)

    # Wrap everything around
    dataset['item_angle'] = wrap(dataset['item_angle'], np.pi/2.)
    dataset['probe_angle'] = wrap(dataset['probe_angle'], np.pi/2.)
    dataset['item_colour'] = wrap(dataset['item_colour'], np.pi/2.)
    dataset['probe_colour'] = wrap(dataset['probe_colour'], np.pi/2.)

    # Remove wrong trials
    reject_ids = (dataset['reject'] == 1.0).flatten()
    for key in dataset:
        if type(dataset[key]) == np.ndarray and dataset[key].shape[0] == reject_ids.size and key in ('probe_colour', 'probe_angle', 'item_angle', 'item_colour'):
            dataset[key][reject_ids] = np.nan

    # Compute the errors
    dataset['errors_angle_all'] = wrap(dataset['item_angle'] - dataset['probe_angle'], np.pi/2.)
    dataset['errors_colour_all'] = wrap(dataset['item_colour'] - dataset['probe_colour'], np.pi/2.)
    dataset['error_angle'] = dataset['errors_angle_all'][:, 0]
    dataset['error_colour'] = dataset['errors_colour_all'][:, 0]
    dataset['error'] = np.where(~np.isnan(dataset['error_angle']), dataset['error_angle'], dataset['error_colour'])

    
    # Fit the mixture model
    dataset['em_fits'] = dict(kappa=np.empty(dataset['probe_angle'].size), mixt_target=np.empty(dataset['probe_angle'].size), mixt_nontarget=np.empty(dataset['probe_angle'].size), mixt_random=np.empty(dataset['probe_angle'].size), resp_target=np.empty(dataset['probe_angle'].size), resp_nontarget=np.empty(dataset['probe_angle'].size), resp_random=np.empty(dataset['probe_angle'].size))
    for key in dataset['em_fits']:
        dataset['em_fits'][key].fill(np.nan)
    
    # Angles trials
    ids_angle = (dataset['cond'] ==  2.0).flatten()
    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        ids_n_items = (dataset['n_items'] == n_items).flatten()
        ids_filtered = ids_angle & ids_n_items

        kappa, mixt_target, mixt_nontarget, mixt_random, resp_ik = em_circularmixture.fit(dataset['probe_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 0], dataset['item_angle'][ids_filtered, 1:])
        dataset['em_fits']['kappa'][ids_filtered] = kappa
        dataset['em_fits']['mixt_target'][ids_filtered] = mixt_target
        dataset['em_fits']['mixt_nontarget'][ids_filtered] = mixt_nontarget
        dataset['em_fits']['mixt_random'][ids_filtered] = mixt_random
        dataset['em_fits']['resp_target'][ids_filtered] = resp_ik[:, 0]
        dataset['em_fits']['resp_nontarget'][ids_filtered] = resp_ik[:, 1]
        dataset['em_fits']['resp_random'][ids_filtered] = resp_ik[:, 2]

    # Colour trials
    ids_colour = (dataset['cond'] ==  1.0).flatten()
    for n_items_i, n_items in enumerate(np.unique(dataset['n_items'])):
        ids_n_items = (dataset['n_items'] == n_items).flatten()
        ids_filtered = ids_colour & ids_n_items

        kappa, mixt_target, mixt_nontarget, mixt_random, resp_ik = em_circularmixture.fit(dataset['probe_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 0], dataset['item_colour'][ids_filtered, 1:])
        dataset['em_fits']['kappa'][ids_filtered] = kappa
        dataset['em_fits']['mixt_target'][ids_filtered] = mixt_target
        dataset['em_fits']['mixt_nontarget'][ids_filtered] = mixt_nontarget
        dataset['em_fits']['mixt_random'][ids_filtered] = mixt_random
        dataset['em_fits']['resp_target'][ids_filtered] = resp_ik[:, 0]
        dataset['em_fits']['resp_nontarget'][ids_filtered] = resp_ik[:, 1]
        dataset['em_fits']['resp_random'][ids_filtered] = resp_ik[:, 2]

    # Try with Pandas for some advanced plotting
    dataset_filtered = dict((k, dataset[k].flatten()) for k in ('n_items', 'trial', 'subject', 'reject', 'rating', 'probe_colour', 'probe_angle', 'cond', 'error', 'error_angle', 'error_colour'))
    dataset_filtered.update(dataset['em_fits'])
    dataset['panda'] = pd.DataFrame(dataset_filtered)



def load_dataset(filename = '', specific_preprocess = lambda x, p: x, parameters={}):
    '''
        Load datasets. 
        Supports
            - Simultaneous and sequential Gorgoraptis_2011 data: Exp1 and Exp2.
            - Dual recall data DualRecall_Bays.
    '''

    # Load everything
    dataset = sio.loadmat(filename, mat_dtype=True)

    # Specific operations, for different types of datasets
    specific_preprocess(dataset, parameters)

    
    return dataset


def load_multiple_datasets(datasets_descr = []):
    '''
        Takes a list of datasets descriptors, and loads them up.
    '''

    datasets = []
    for datasets_descr in datasets_descr:
        datasets.append( load_dataset(filename=datasets_descr['filename'], specific_preprocess=datasets_descr['preprocess'], parameters=datasets_descr['parameters']))

    return datasets


def reconstruct_colours_exp1(datasets=('Data/ad.mat', 'Data/gb.mat', 'Data/kf.mat', 'Data/md.mat', 'Data/sf.mat', 'Data/sw.mat', 'Data/wd.mat', 'Data/zb.mat')):
    '''
        The colour is missing from the simultaneous experiment dataset
        Reconstruct it.
    '''

    all_colours = []
    all_preangles = []
    all_targets = []
    for dataset_fn in datasets:
        print dataset_fn
        curr_data = sio.loadmat(dataset_fn, mat_dtype=True)

        all_colours.append(curr_data['probe_colour'])
        all_preangles.append(wrap(np.deg2rad(curr_data['probe_pre_angle']), max_angle=np.pi/2.))
        all_targets.append(wrap(np.deg2rad(curr_data['item_angle'][:, 0]), max_angle=np.pi/2.))

    print "Data loaded"

    all_colours = np.array(all_colours)
    all_targets = np.array(all_targets)
    all_preangles = np.array(all_preangles)

    # Ordering in original data
    order_subjects = [0, 4, 1, 5, 2, 6, 3, 7]
    all_colours = all_colours[order_subjects]
    all_targets = all_targets[order_subjects]
    all_preangles = all_preangles[order_subjects]

    # Flatten everything
    all_colours_ = []
    all_targets_ = []
    all_preangles_ = []
    for curr_colour in all_colours:
        all_colours_.extend(np.squeeze(curr_colour).tolist())
    print "Colours flat"
    for curr_target in all_targets:
        all_targets_.extend(np.squeeze(curr_target).tolist())
    print "Targets flat"
    for curr_preangle in all_preangles:
        all_preangles_.extend(np.squeeze(curr_preangle).tolist())
    print "Preangles flat"


    return (np.array(all_colours_), np.array(all_targets_), np.array(all_preangles_))



def compute_all_errors(dataset = {}):
    '''
        Will compute the error between the response and all possible items
    '''

    # Get the difference between angles
    # Should also wrap it around
    dataset['errors_all'] = wrap(dataset['item_angle'] - dataset['response'], max_angle=np.pi/2.)

    # Sanity check, verify that errors computed are the same as precomputed ones.
    assert all(np.abs(dataset['errors_all'][np.arange(dataset['probe'].size), dataset['probe'][:, 0]] - dataset['error'][:, 0]) < 10**-6), "Errors computed are different than errors given in the data"


def create_subject_arrays(dataset = {}):
    '''
        Create arrays with errors per subject and per num_target
        also create an array with the precision per subject and num_target directly
    '''

    unique_subjects = np.unique(dataset['subject'])
    unique_n_items = np.unique(dataset['n_items'])

    errors_subject_nitems = np.empty((unique_subjects.size, unique_n_items.size), dtype=np.object)
    precision_subject_nitems = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_theo = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_nochance = np.zeros((unique_subjects.size, unique_n_items.size))
    precision_subject_nitems_raw = np.zeros((unique_subjects.size, unique_n_items.size))

    for n_items_i, n_items in enumerate(unique_n_items):
        for subject_i, subject in enumerate(unique_subjects):
            # Get the responses and errors
            errors_subject_nitems[subject_i, n_items_i] = dataset['error'][(dataset['subject']==subject) & (dataset['n_items'] == n_items)]

            # Compute the precision
            precision_subject_nitems[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=True)
            precision_subject_nitems_theo[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=False, correct_orientation=True, use_wrong_precision=False)
            precision_subject_nitems_nochance[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=True, correct_orientation=True, use_wrong_precision=False)
            precision_subject_nitems_raw[subject_i, n_items_i] = compute_precision(errors_subject_nitems[subject_i, n_items_i], remove_chance_level=False)

    dataset['errors_subject_nitems'] = errors_subject_nitems
    dataset['precision_subject_nitems_bays'] = precision_subject_nitems
    dataset['precision_subject_nitems_theo'] = precision_subject_nitems_theo
    dataset['precision_subject_nitems_theo_nochance'] = precision_subject_nitems_nochance
    dataset['precision_subject_nitems_bays_chance'] = precision_subject_nitems_raw


def check_bias_all(dataset):
    '''
        Get an histogram of the errors between the response and all non targets
            If biased towards 0-values, should indicate misbinding errors.

        (if you do this with respect to all targets, it's retarded and should always be biased)
    '''

    # Remove all the probes, by setting them to nan
    all_errors = dataset['errors_all'].copy()
    all_errors[np.arange(dataset['probe'].size), dataset['probe'][:, 0]] = np.nan

    # Now filter all nans
    all_errors = all_errors[~np.isnan(all_errors)]

    # Some plots
    plt.figure()
    plt.hist(all_errors, bins=20)


def check_bias_bestnontarget(dataset):
    '''
        Get an histogram of errors between response and best nontarget.
        Should be more biased towards 0 than the overall average
    '''

    # all_errors = dataset['errors_all']
    all_errors = dataset['errors_all'][(dataset['n_items'] > 1)[:, 0]]  # filter trials with only no nontargets
    probe_indices = dataset['probe'][(dataset['n_items'] > 1)[:, 0], 0]

    ## Remove all the probes, by setting them to nan
    # all_errors[np.arange(probe_indices.size), probe_indices] = np.nan
    
    # Keep only the best non target for each trial
    # Use Bottleneck
    # min_indices = bn.nanargmin(np.abs(all_errors), axis=1)
    # all_errors = all_errors[np.arange(probe_indices.size), min_indices]

    ## More efficient, use masked arrays
    masked_errors = np.ma.masked_invalid(all_errors)
    # mask the probed item
    masked_errors[np.arange(probe_indices.size), probe_indices] = np.ma.masked

    # Get the best non targets
    all_errors = all_errors[np.arange(probe_indices.size), np.ma.argmin(np.abs(masked_errors), axis=1)]
    
    ## Superslow    
    # all_errors = np.array([all_errors[i, np.nanargmin(np.abs(all_errors[i]))] for i in xrange(all_errors.shape[0]) if not np.isnan(np.nanargmin(np.abs(all_errors[i])))])

    # Some plots
    plt.figure()
    plt.hist(all_errors, bins=20)


def compute_precision(errors, remove_chance_level=True, correct_orientation=True, use_wrong_precision=True):
    '''
        Compute the precision (1./circ_std**2). Remove the chance level if desired.
    '''

    if correct_orientation:
        # Correct for the fact that bars are modelled in [0, pi] and not [0, 2pi]
        errors = errors.copy()*2.0

    avg_error = np.mean(np.abs(errors), axis=0)
        
    # Angle population vector
    error_mean_vector = np.mean(np.exp(1j*errors), axis=0)
        
    # Population mean
    error_mean_error = np.angle(error_mean_vector)
        
    # Circular standard deviation estimate
    error_std_dev_error = np.sqrt(-2.*np.log(np.abs(error_mean_vector)))

    # Precision
    if use_wrong_precision:
        precision = 1./error_std_dev_error
    else:
        precision = 1./error_std_dev_error**2.

    if remove_chance_level:
        # Expected precision under uniform distribution
        x = np.logspace(-2, 2, 100)

        precision_uniform = np.trapz(errors.size/(np.sqrt(x)*np.exp(x+errors.size*np.exp(-x))), x)

        # Remove the chance level
        precision -= precision_uniform

    if correct_orientation:
        # The obtained precision is for half angles, correct it
        precision *= 2.

    return precision


def check_oblique_effect(data, nb_bins=100):
    '''
        Humans are more precise for vertical and horizontal bars than diagonal orientations.

        Check if present.
    '''

    # Construct the list of (target angles, errors), see if there is some structure in that
    errors_per_angle = np.array(zip(data['item_angle'][np.arange(data['probe'].size), data['probe'][:, 0]], data['error'][:, 0]))

    plt.figure()
    plt.plot(errors_per_angle[:, 0], errors_per_angle[:, 1], 'x')

    plt.figure()
    plt.plot(errors_per_angle[:, 0], np.abs(errors_per_angle[:, 1]), 'x')

    discrete_x = np.linspace(-np.pi/2., np.pi/2., nb_bins)
    avg_error = np.zeros(discrete_x.shape)

    for x_i in np.arange(discrete_x.size):
        if x_i < discrete_x.size - 1:
            # Check what data comes in the current interval x[x_i, x_i+1]
            avg_error[x_i] = np.mean(errors_per_angle[np.logical_and(errors_per_angle[:, 0] > discrete_x[x_i], errors_per_angle[:, 0] < discrete_x[x_i+1]), 1])

    plt.figure()
    plt.plot(discrete_x, avg_error)

    plt.figure()
    plt.plot(discrete_x, avg_error**2.)

    plt.figure()
    plt.plot(discrete_x, np.abs(avg_error))


def plots_doublerecall(dataset):
    '''
        Create plots for the double recall dataset
    '''

    dataset_pd = dataset['panda']

    dataset_pd['error_abs'] = dataset_pd.error.abs()
    
    # dataset_pd[ids_filtered][ids_targets_responses].boxplot('error_angle_abs', by='rating')
    # dataset_pd[ids_filtered][ids_nontargets_responses].boxplot('error_angle_abs', by='rating')
    dataset_pd.boxplot(column=['error_abs'], by=['cond', 'n_items', 'rating'])
    
    for i in dataset_pd.subject.unique():
        dataset_pd[dataset_pd.subject == i].boxplot(column=['error_angle'], by=['n_items', 'rating'])

    dataset_grouped_nona_rating = dataset_pd[dataset_pd.n_items == 3.0].dropna(subset=['error']).groupby(['rating'])
    f, axes = plt.subplots(dataset_pd.rating.nunique(), 3)
    i=0
    for name, group in dataset_grouped_nona_rating:
        print name

        # Print Responsibility target density estimation
        # group.resp_target.plot(kind='kde', ax=axes[i, 0])
        group.resp_target.hist(ax=axes[i, 0], color='b', bins=20)
        axes[i, 0].text(0.9*axes[i, 0].axis()[1], 0.8*axes[i, 0].axis()[3], "T" + str(name), fontweight='bold')

        # Print Responsibility nontarget density estimation
        # group.resp_nontarget.plot(kind='kde', ax=axes[i, 1])
        group.resp_nontarget.hist(ax=axes[i, 1], color='g', bins=20)
        axes[i, 1].text(0.9*axes[i, 1].axis()[1], 0.8*axes[i, 1].axis()[3], "NT" + str(name), fontweight='bold')

        # Print Responsibility random density estimation
        # group.resp_random.plot(kind='kde', ax=axes[i, 1])
        group.resp_random.hist(ax=axes[i, 2], color='r', bins=20)
        axes[i, 2].text(0.9*axes[i, 2].axis()[1], 0.8*axes[i, 2].axis()[3], "R" + str(name), fontweight='bold')

        i+=1

    # Add condition names
    dataset_pd['cond_name'] = np.array(['Colour', 'Angle'])[np.array(dataset_pd['cond']-1, dtype=int)]

    dataset_grouped_nona_conditems = dataset_pd.dropna(subset=['error']).groupby(['cond_name', 'n_items'])
    dataset_grouped_nona_conditems.boxplot(column=['mixt_target', 'mixt_nontarget', 'mixt_random'])
    dataset_grouped_nona_conditems.boxplot(column=['resp_target', 'resp_nontarget', 'resp_random'])
    

if __name__ == '__main__':
    ## Load data
    data_dir = '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data/'

    # keys:
    # 'probe', 'delayed', 'item_colour', 'probe_colour', 'item_angle', 'error', 'probe_angle', 'n_items', 'response', 'subject']
    (data_sequen, data_simult, data_dualrecall) = load_multiple_datasets([dict(filename=os.path.join(data_dir, 'Gorgoraptis_2011', 'Exp1.mat'), preprocess=preprocess_sequential, parameters={}), dict(filename=os.path.join(data_dir, 'Gorgoraptis_2011', 'Exp2.mat'), preprocess=preprocess_simultaneous, parameters={}), dict(filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'), preprocess=preprocess_doublerecall, parameters={})])
    

    # Check for bias towards 0 for the error between response and all items
    check_bias_all(data_simult)

    # Check for bias for the best non-probe
    check_bias_bestnontarget(data_simult)

    # check_bias_all(data_sequen)
    check_bias_bestnontarget(data_sequen)

    # print data_simult['precision_subject_nitems_bays']
    # print data_simult['precision_subject_nitems_theo']

    # prec_exp = np.mean(data_simult['precision_subject_nitems_bays'], axis=0)
    # prec_theo = np.mean(data_simult['precision_subject_nitems_theo'], axis=0)
    # fi_fromexp = prec_exp**2./4.
    # fi_fromtheo = prec_theo**2./4.
    # print "Precision experim", prec_exp
    # print "FI from exp", fi_fromexp
    # print "Precision no chance level removed", prec_theo
    # print "FI no chance", fi_fromtheo
    
    check_oblique_effect(data_simult, nb_bins=50)

    # np.save('processed_experimental_230613.npy', dict(data_simult=data_simult, data_sequen=data_sequen))
    
    plt.show()





