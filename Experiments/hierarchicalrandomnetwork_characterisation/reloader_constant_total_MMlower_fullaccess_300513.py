"""
    ExperimentDescriptor for hierarchicalrandomnetwork parameter sweep.
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


def plots_3dvolume_hierarchical_M_Mlayerone(data_pbs, generator_module=None):
    '''
        Reload 3D volume runs from PBS and plot them

    '''
    
    print "Order parameters: ", generator_module.dict_parameters_range.keys() 

    results_precision_constant_M_Mlower = np.squeeze(nanmean(data_pbs.dict_arrays['results_precision_M_T']['results'], axis=-1))
    
    M_space = data_pbs.loaded_data['parameters_uniques']['M']
    M_layer_one_space = data_pbs.loaded_data['parameters_uniques']['M_layer_one']
    
    print M_space
    print M_layer_one_space
    print results_precision_constant_M_Mlower.shape
    # print results_precision_constant_M_Mlower
    
    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    savefigs = True


    if generator_module.v2:
        # This version allows for M + M_lower <= 200
        # Create one pcolor_2d per T
        T = results_precision_constant_M_Mlower.shape[-1]

        for t in xrange(T):
            pcolor_2d_data(results_precision_constant_M_Mlower[..., t], x=M_space, y=M_layer_one_space, xlabel='M', ylabel='M_layer_one', title='T %d' % (t+1), colorbar=True, log_scale=False, ticks_interpolate=None)

            if savefigs:
                dataio.save_current_figure('results_{label}_fixedT%d.pdf' % (t+1))

    else:
        # This version forces M + M_lower = 200
        # Fewer plots

        T = results_precision_constant_M_Mlower.shape[-1]

        ratio_MMlower = M_space/200.
        pcolor_2d_data(results_precision_constant_M_Mlower[np.arange(M_space.size), np.arange(-M_layer_one_space.size, 0)[::-1]], log_scale=True, x=ratio_MMlower, y=np.arange(1, T+1), xlabel='M/(M+M_layer_one)', ylabel='T')

        if savefigs:
            dataio.save_current_figure('results_{label}_global.pdf')


    variables_to_save = ['results_precision_constant_M_Mlower', 'M_space', 'M_layer_one_space']
    dataio.save_variables(variables_to_save, locals())
    plt.show()




generator_script='generator_constant_total_MMlower_fullaccess_300513.py'

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Hierarchical network. Assume we want to allocate a fixed number of neurons between the two layers. Do that by constraining the sum of M and M_layer_one to be some constant. Corrected logic so that whole population is accessible now', 
                     files="%s/%s-*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label']),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['M', 'M_layer_one'],
                     variables_to_load=['results_precision_M_T'],
                     variables_description=['Precision volume'],
                     post_processing=plots_3dvolume_hierarchical_M_Mlayerone,
                     save_output_filename='3dvolume_hierarchical_constant_M_Mlayerone'
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

