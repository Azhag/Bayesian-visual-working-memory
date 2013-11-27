"""
    ExperimentDescriptor for Fitting experiments in a mixed population code
"""

import os
import numpy as np
from experimentlauncher import *
import inspect

# Commit @db3fc53 +

import os
import re
import numpy as np
from experimentlauncher import *
from dataio import *
from utils import *

import inspect

# Commit @db3fc53 +


def plots_logposterior_mixed_tworcscale(data_pbs, generator_module=None):
    '''
        Reload 2D volume runs from PBS and plot them

    '''

    #### SETUP
    #
    savefigs = True
    plot_all_pcolor = False
    plot_global_max = True
    #
    #### /SETUP

    print "Order parameters: ", generator_module.dict_parameters_range.keys()

    result_log_posterior_mean = np.squeeze(data_pbs.dict_arrays['result_log_posterior_mean']['results'])
    result_log_posterior_std = np.squeeze(data_pbs.dict_arrays['result_log_posterior_std']['results'])

    ratio_space = data_pbs.loaded_data['parameters_uniques']['ratio_conj']
    sigmax_space = data_pbs.loaded_data['parameters_uniques']['sigmax']

    # FROM launcher_fitexperiment.launcher_do_fitexperiment_mixed_tworcscale
    rcscale_space = np.linspace(0.01, 10, 15)    # kappa conj
    rcscale2_space = np.linspace(0.01, 30., 17)   # kappa feat

    print ratio_space
    print sigmax_space
    print result_log_posterior_mean.shape, result_log_posterior_std.shape

    dataio = DataIO(output_folder=generator_module.pbs_submission_infos['simul_out_dir'] + '/outputs/', label='global_' + dataset_infos['save_output_filename'])

    plt.rcParams['font.size'] = 16

    if plot_all_pcolor:
        for ratio_conj_i, ratio_conj in enumerate(ratio_space):
            for sigmax_i, sigmax in enumerate(sigmax_space):

                # Pcolor for this ratio and sigma
                pcolor_2d_data(result_log_posterior_mean[ratio_conj_i, sigmax_i], x=rcscale_space, y=rcscale2_space, xlabel="kappa conj", ylabel="kappa feature", title="Loglikelihood, ratio %.2f, sigmax %.2f" % (ratio_conj, sigmax), ticks_interpolate=5)

                # ax.get_figure().canvas.draw()

                if savefigs:
                    dataio.save_current_figure('pcolorall_fitexp_loglike_ratio%.2fsigmax%.2f_{label}_global_{unique_id}.pdf' % (ratio_conj, sigmax))

    if plot_global_max:
        # Pcolor for the max (rcscale, rcscale2) per condition
        # Coverage is bad though...

        max_result_log_posterior_mean = np.nanmax(np.nanmax(result_log_posterior_mean, axis = -1), axis = -1)

        pcolor_2d_data(max_result_log_posterior_mean, x=ratio_space, y=sigmax_space, xlabel="Ratio conj", ylabel="Sigmax", title="Loglikelihood, best rcscale, rcscale2 each", ticks_interpolate=5)

        if savefigs:
            dataio.save_current_figure('fitexp_loglike_globalmax_rcscalercscale2_{label}_global_{unique_id}.pdf')



    all_args = data_pbs.loaded_data['args_list']
    variables_to_save = ['result_log_posterior_mean', 'result_log_posterior_std', 'ratio_space', 'sigmax_space', 'all_args']

    if savefigs:
        dataio.save_variables(variables_to_save, locals())


    plt.show()

    return locals()


this_file = os.path.basename(inspect.getfile(inspect.currentframe()))

parameters_entryscript = dict(action_to_do='launcher_do_reload_constrained_parameters', output_directory='.')

generator_script = 'generator' + re.split("^reloader", this_file)[-1]

generator_module = imp.load_source(os.path.splitext(generator_script)[0], generator_script)
dataset_infos = dict(label='Fitting of experimental data. Dualrecall dataset, fitting for 3 items. We submit varying ratio_conj and sigmax, and vary rcscale and rcscale2 on each node. Should plot as a 2D surface',
                     files="%s/%s*.npy" % (generator_module.pbs_submission_infos['simul_out_dir'], generator_module.pbs_submission_infos['other_options']['label'].split('{')[0]),
                     launcher_module=generator_module,
                     loading_type='args',
                     parameters=['ratio_conj', 'sigmax'],
                     variables_to_load=['result_log_posterior_mean', 'result_log_posterior_std'],
                     variables_description=['Log posterior mean of experimental data', 'Log posterior std dev'],
                     post_processing=plots_logposterior_mixed_tworcscale,
                     save_output_filename='plots_fitexp_mixed_autoset'
                     )



if __name__ == '__main__':

    print "Running ", this_file

    arguments_dict=dict(parameters_filename=this_file)
    arguments_dict.update(parameters_entryscript)
    experiment_launcher = ExperimentLauncher(run=True, arguments_dict=arguments_dict)

    variables_to_reinstantiate = ['data_gen', 'sampler', 'stat_meas', 'random_network', 'args', 'constrained_parameters', 'data_pbs', 'dataio', 'post_processing_outputs', 'fit_exp']

    if 'variables_to_save' in experiment_launcher.all_vars:
        # Also reinstantiate the variables we saved
        variables_to_reinstantiate.extend(experiment_launcher.all_vars['variables_to_save'])

    for var_reinst in variables_to_reinstantiate:
        if var_reinst in experiment_launcher.all_vars:
            vars()[var_reinst] = experiment_launcher.all_vars[var_reinst]

    for var_reinst in post_processing_outputs:
        vars()[var_reinst] = post_processing_outputs[var_reinst]

