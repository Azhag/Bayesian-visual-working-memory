'''
     Load experimental data in python, because Matlab sucks ass.
'''

import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# import matplotlib.patches as plt_patches
# import matplotlib.gridspec as plt_grid
import os
import os.path
import cPickle as pickle
# import bottleneck as bn
import em_circularmixture
import em_circularmixture_allitems_uniquekappa
import pandas as pd

import dataio as DataIO

import utils

from plots_experimental_data import *
from experimentalloaderbays09 import *
from experimentalloaderdualrecall import *
from experimentalloadergorgo11seq import *
from experimentalloadergorgo11sim import *

######


def load_data_simult(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Gorgoraptis_2011 dataset.
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    expLoader = ExperimentalLoaderGorgo11Simultaneous(dict(name='gorgo11', filename='Exp2_withcolours.mat', datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), parameters=dict(fit_mixture_model=fit_mixture_model, mixture_model_cache='em_simult.pickle', collapsed_mixture_model_cache='collapsed_em_simult.pickle')))

    return expLoader.dataset


def load_data_gorgo11(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Gorgo11 simultaneous dataset.
    '''

    return load_data_simult(data_dir, fit_mixture_model)


def load_data_gorgo11_sequential(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Gorgo11 sequential dataset.
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    expLoader = ExperimentalLoaderGorgo11Sequential(dict(name='gorgo11seq', filename='Exp1.mat', datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), parameters=dict(fit_mixture_model=fit_mixture_model, mixture_model_cache='em_gorgo_seq.pickle', collapsed_mixture_model_cache='collapsed_em_gorgo_seq.pickle')))

    return expLoader.dataset


def load_data_bays09(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Bays2009 dataset.
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    expLoader = ExperimentalLoaderBays09(dict(name='bays09', filename='colour_data.mat', datadir=os.path.join(data_dir, 'Bays2009'), parameters=dict(fit_mixture_model=fit_mixture_model, mixture_model_cache='em_bays_allitems.pickle', collapsed_mixture_model_cache='collapsed_em_bays.pickle')))

    return expLoader.dataset



def load_data_dualrecall(data_dir='../../experimental_data/', fit_mixture_model=False):
    '''
        Convenience function, automatically load the Double recall dataset (unpublished).
    '''

    if data_dir == '../../experimental_data/':
        experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
        data_dir = os.path.normpath(os.path.join(experim_datadir, data_dir))

    expLoader = ExperimentalLoaderDualRecall(dict(name='dualrecall', filename='rate_data.mat', datadir=os.path.join(data_dir, 'DualRecall_Bays'), parameters=dict(fit_mixture_model=fit_mixture_model, mixture_model_cache='em_dualrecall_allitems.pickle', collapsed_mixture_model_cache='collapsed_em_dualrecall.pickle')))

    return expLoader.dataset

def load_data(experiment_id='bays09', data_dir='../../experimental_data/', fit_mixture_model=True):
    '''
        Load the appropriate dataset given an experiment_id.
    '''
    if experiment_id == 'bays09':
        return load_data_bays09(data_dir=data_dir, fit_mixture_model=fit_mixture_model)
    elif experiment_id == 'gorgo11':
        return load_data_gorgo11(data_dir=data_dir, fit_mixture_model=fit_mixture_model)
    elif experiment_id == 'gorgo11_sequential':
        return load_data_gorgo11_sequential(data_dir=data_dir, fit_mixture_model=fit_mixture_model)
    elif experiment_id == 'dualrecall':
        return load_data_dualrecall(data_dir=data_dir, fit_mixture_model=fit_mixture_model)
    else:
        raise ValueError('Experiment_id %s unknown.' % experiment_id)


if __name__ == '__main__':
    ## Load data
    experim_datadir = os.environ.get('WORKDIR_DROP', os.path.split(utils.__file__)[0])
    data_dir = os.path.normpath(os.path.join(experim_datadir, '../../experimental_data/'))
    # data_dir = '/Users/loicmatthey/Dropbox/UCL/1-phd/Work/Visual_working_memory/experimental_data/'

    # data_dir = os.path.normpath(os.path.join(experim_datadir, '../experimental_data/'))

    print sys.argv

    if True or (len(sys.argv) > 1 and sys.argv[1]):
        # keys:
        # 'probe', 'delayed', 'item_colour', 'probe_colour', 'item_angle', 'error', 'probe_angle', 'n_items', 'response', 'subject']
        # (data_sequen, data_simult, data_dualrecall) = load_multiple_datasets([dict(filename='Exp1.mat', parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'))), dict(filename='Exp2_withcolours.mat',  parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), fit_mixture_model=True)), dict(filename=os.path.join(data_dir, 'DualRecall_Bays', 'rate_data.mat'),  parameters=dict(fit_mixture_model=True))])
        # (data_simult,) = load_multiple_datasets([dict(name='Gorgo_simult', filename='Exp2_withcolours.mat',  parameters=dict(datadir=os.path.join(data_dir, 'Gorgoraptis_2011'), fit_mixture_model=True, mixture_model_cache='em_simult.pickle'))])
        # (data_bays2009, ) = load_multiple_datasets([dict(name='Bays2009', filename='colour_data.mat', parameters=dict(datadir=os.path.join(data_dir, 'Bays2009'), fit_mixture_model=True, mixture_model_cache='em_bays.pickle', should_compute_bootstrap=True, bootstrap_cache='bootstrap_1000samples.pickle'))])

        # data_bays2009 = load_data_bays09(data_dir=data_dir, fit_mixture_model=True)
        # data_gorgo11 = load_data_gorgo11(data_dir=data_dir, fit_mixture_model=True)
        # data_dualrecall = load_data_dualrecall(data_dir=data_dir, fit_mixture_model=True)
        # data_gorgo11_sequ = load_data_gorgo11_sequential(data_dir=data_dir, fit_mixture_model=True)

        data_bays09 = load_data('bays09', data_dir=data_dir)



    # Check for bias towards 0 for the error between response and all items
    # check_bias_all(data_simult)

    # Check for bias for the best non-probe
    # check_bias_bestnontarget(data_simult)

    # check_bias_all(data_sequen)
    # check_bias_bestnontarget(data_sequen)

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

    # plots_check_oblique_effect(data_simult, nb_bins=50)

    # np.save('processed_experimental_230613.npy', dict(data_simult=data_simult, data_sequen=data_sequen))

    # plots_dualrecall(data_dualrecall)

    plt.rcParams['font.size'] = 16
    dataio = None

    # dataio = DataIO.DataIO(label='experiments_bays2009')
    # plots_check_bias_nontarget(data_simult, dataio=dataio)
    # plots_check_bias_bestnontarget(data_simult, dataio=dataio)
    # plots_check_bias_nontarget_randomized(data_simult, dataio=dataio)
    # plots_bays2009(data_bays2009, dataio=dataio)

    # dataio = DataIO.DataIO(label='experiments_gorgo11')
    # plots_gorgo11(data_gorgo11, dataio)

    # plots_precision(data_gorgo11, dataio)
    # plots_precision(data_bays2009, dataio)

    # dataio = DataIO.DataIO(label='experiments_bays2009')
    # plot_bias_close_feature(data_bays2009, dataio)

    # dataio = DataIO.DataIO(label='experiments_gorgo11')
    # plot_bias_close_feature(data_gorgo11, dataio)

    # plot_compare_bic_collapsed_mixture_model(data_bays2009, dataio)
    # plot_compare_bic_collapsed_mixture_model(data_gorgo11, dataio)

    if False:
        for subj in data_bays2009['data_subject_split']['subjects_space'][:5]:
            for nitems_i, nitems in enumerate(data_bays2009['data_subject_split']['nitems_space']):
                utils.scatter_marginals(data_bays2009['data_subject_split']['data_subject'][subj]['targets'][nitems_i], data_bays2009['data_subject_split']['data_subject'][subj]['responses'][nitems_i], title='Subject %d, %d items' % (subj, nitems))

    # dataio = DataIO.DataIO(label='experiments_gorgo11_seq')
    # plots_gorgo11_sequential(data_gorgo11_sequ, dataio)
    # plots_gorgo11_sequential_collapsed(data_gorgo11_sequ, dataio)

    # plot_compare_bic_collapsed_mixture_model_sequential(data_gorgo11_sequ, dataio)

    plt.show()





