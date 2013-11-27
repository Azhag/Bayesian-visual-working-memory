"""
    PBS reloader for hierarchicalrandomnetwork parameter sweep.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *


from experimentlauncher import *
from dataio import *
import inspect


# # Commit @360c93c +

parameters_entryscript=dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')


def plots_3dvolume_hierarchical_M_sparsity_sigmaweights(data_pbs, generator_module=None):
    '''
        Reload 3D volume runs from PBS and plot them

    '''

    print "Order parameters: ", generator_module.dict_parameters_range.keys() 

    results_precision_M_sparsity_sigmaweights = np.squeeze(nanmean(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))
    
    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    sparsity_space = data_pbs.loaded_data['parameters_uniques']['sparsity']
    sigma_weights_space = data_pbs.loaded_data['parameters_uniques']['sigma_weights']


    print M_space
    print sparsity_space
    print sigma_weights_space
    print results_precision_M_sparsity_sigmaweights.shape
    # print results_precision_M_sparsity_sigmaweights
    
    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    savefigs = True

    # Create one pcolor_2d per T
    T = results_precision_M_sparsity_sigmaweights.shape[-1]

    for sparsity_i, sparsity in enumerate(sparsity_space):
        for t in xrange(T):
            pcolor_2d_data(results_precision_M_sparsity_sigmaweights[sparsity_i, ..., t], x=sigma_weights_space, y=M_space, xlabel='sigma', ylabel='M', title='Sparsity %.2f, T %d' % (sparsity, t+1), colorbar=True, log_scale=False, ticks_interpolate=None)

            if savefigs:
                dataio.save_current_figure('results_{label}_sparsity%.2ffixedT%d.pdf' % (sparsity, t+1))

    dataio.save_variables(['results_precision_M_sparsity_sigmaweights', 'M_space', 'sparsity_space', 'sigma_weights_space'], locals())
    plt.show()




generator_script='generator_sweep_feature_code_210513.py'
generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)

dataset_infos = dict(label='HierarchicalRandomNetwork initial characterisation. Feature layer one. 3D volume (sparsity, sigma_weights, M). Precision.', 
                     files="%s/%s-*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label']),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=generator_module.dict_parameters_range.keys(),
                     variables_to_load=['results_precision_M_T'],
                     variables_description=['Precision volume'],
                     post_processing=plots_3dvolume_hierarchical_M_sparsity_sigmaweights,
                     save_output_filename='3dvolume_hierarchical_M_sparsity_sigmaweights'
                     )




if __name__ == '__main__':
    
    this_file = inspect.getfile(inspect.currentframe())
    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]


