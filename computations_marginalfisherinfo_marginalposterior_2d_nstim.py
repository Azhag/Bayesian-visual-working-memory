#!/usr/bin/env python
# encoding: utf-8
"""
computations_marginalfisherinfo_marginalposterior_2d_nstim.py

Created by Loic Matthey on 2013-02-23.
Copyright (c) 2012 Gatsby Unit. All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scsp
import sys
import functools

from utils import *
# from statisticsmeasurer import *
# from randomfactorialnetwork import *
# from datagenerator import *
# from slicesampler import *
# from gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork import *
from dataio import *
import progress
import tables as tb


def main(to_plot = []):
    ####
    #   2D N stimuli
    ####

    n_items = 2
    use_pytables = True

    N     = int(13*13)
    kappa = 3.0
    sigma = 0.5
    amplitude = 1.0
    # min_distance = 0.01
    min_distance = 0.1

    dataio = DataIO(label='compute_fimarg_2dnstim', calling_function='')
    additional_comment = ''

    N_sqrt = int(np.sqrt(N))
    coverage_1D = np.linspace(-np.pi, np.pi, N_sqrt, endpoint=False)
    pref_angles = np.array(cross(2*[coverage_1D.tolist()]))

    def population_code_response_2D(theta1, theta2, pref_angles, N=10, kappa=0.1, amplitude=1.0):
        return amplitude*np.exp(kappa*np.cos(theta1 - pref_angles[:, 0]) + kappa*np.cos(theta2 - pref_angles[:, 1]))/(4.*np.pi**2.*scsp.i0(kappa)**2.)

    ## Estimate likelihood
    num_points = 100
    # num_points_space = np.arange(50, 1000, 200)
    # effects_num_points = []

    # all_angles = np.linspace(0., 2.*np.pi, num_points, endpoint=False)
    all_angles_1D = np.linspace(-np.pi, np.pi, num_points, endpoint=True)


    def enforce_distance(theta1, theta2, min_distance=0.1):
        return np.abs(wrap_angles(theta1 - theta2)) > min_distance

    def show_population_output(data):
        N_sqrt = int(np.sqrt(data.size))
        plt.figure()
        plt.imshow(data.reshape((N_sqrt, N_sqrt)).T, origin='left', interpolation='nearest')
        plt.show()


    #### Compute Theo Inverse Fisher Info

    if 1 in to_plot:
        ### Loop over min_distance and kappa
        # min_distance_space = np.linspace(0.0, 1.5, 10)
        # min_distance_space = np.array([min_distance])
        # min_distance_space = np.array([0.001])
        # min_distance_space = np.array([0.8])
        # kappa_space = np.linspace(0.05, 30., 40.)
        # kappa_space = np.array([kappa])

        # sigma_space = np.array([sigma])
        # sigma_space = np.array([0.1, 0.25, 0.5])
        # sigma_space = np.linspace(0.1, 1.0, 11)

        # dim1_size = min_distance_space.size
        # dim2_size = sigma_space.size

        item1_theta1 = 0.0
        item1_theta2 = 0.0


        deriv_mu = np.zeros((n_items*2, N))

        print "FI 1 obj"

        FI_2d_1obj_search = np.zeros((2, 2))
        inv_FI_2d_1obj_search = np.zeros((2, 2))

        deriv_mu[0] = -kappa*np.sin(pref_angles[:, 0] - item1_theta1)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        deriv_mu[1] = -kappa*np.sin(pref_angles[:, 1] - item1_theta2)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)

        # FI for 1 object
        FI_2d_1obj_search[0, 0] = np.sum(deriv_mu[0]**2.)/sigma**2.
        FI_2d_1obj_search[0, 1] = np.sum(deriv_mu[0]*deriv_mu[1])/sigma**2.
        FI_2d_1obj_search[1, 1] = np.sum(deriv_mu[1]**2.)/sigma**2.
        FI_2d_1obj_search[1, 0] = FI_2d_1obj_search[0, 1]
        inv_FI_2d_1obj_search = np.linalg.inv(FI_2d_1obj_search)

        if n_items > 1:
            print "FI 2-4 obj"
            search_progress = progress.Progress(all_angles_1D.size**(2*(n_items - 1)))

            print search_progress.total_work

            FI_2d_2obj_search = np.zeros((all_angles_1D.size, all_angles_1D.size, 4, 4))
            inv_FI_2d_2obj_search = np.zeros((all_angles_1D.size, all_angles_1D.size, 4, 4))

            if n_items > 2:
                FI_2d_3obj_search_curr = np.zeros((6, 6))

                if use_pytables:
                    f_table = tb.openFile('inv_FI_3obj_search.hdf', 'w', title="Inv Fisher Info 2d 3 objects")
                    atom = tb.Atom.from_dtype(inv_FI_2d_2obj_search.dtype)
                    compression_filter = tb.Filters(complib='blosc', complevel=5)
                    # compression_filter = None
                    inv_FI_2d_3obj_search = f_table.createCArray(f_table.root, 'FI', atom, (all_angles_1D.size**4, 6, 6), filters=compression_filter)

                    print inv_FI_2d_3obj_search
                else:
                    inv_FI_2d_3obj_search = np.zeros((all_angles_1D.size**4, 6, 6))

            ### FI for 2-3 objects
            for i, item2_theta1 in enumerate(all_angles_1D):
                for j, item2_theta2 in enumerate(all_angles_1D):
                    if (enforce_distance(item1_theta1, item2_theta1, min_distance=min_distance) and enforce_distance(item1_theta2, item2_theta2, min_distance=min_distance)):

                        deriv_mu[2] = -kappa*np.sin(pref_angles[:, 0] - item2_theta1)*population_code_response_2D(item2_theta1, item2_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
                        deriv_mu[3] = -kappa*np.sin(pref_angles[:, 1] - item2_theta2)*population_code_response_2D(item2_theta1, item2_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                        # FI, 2 items, 2 features per item
                        # for ii in xrange(4):
                        #     for jj in xrange(4):
                        #         FI_2d_2obj_search[i, j, ii, jj] = np.sum(deriv_mu[ii]*deriv_mu[jj])/(2.*sigma**2.)
                        FI_2d_2obj_search[i, j] = np.dot(deriv_mu, deriv_mu.T)/(2.*sigma**2.)

                        inv_FI_2d_2obj_search[i, j] = np.linalg.inv(FI_2d_2obj_search[i, j])

                        ### FI for 3 items
                        if n_items > 2:
                            for k in xrange(all_angles_1D.size):
                                if search_progress.percentage() % 1.0 < 0.0001:
                                    print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())

                                item3_theta1 = all_angles_1D[k] +1e-6

                                for l in xrange(all_angles_1D.size):

                                    item3_theta2 = all_angles_1D[l]+1e-6

                                    if (enforce_distance(item1_theta1, item3_theta1, min_distance=min_distance) and enforce_distance(item1_theta2, item3_theta2, min_distance=min_distance) and enforce_distance(item2_theta1, item3_theta1, min_distance=min_distance) and enforce_distance(item2_theta2, item3_theta2, min_distance=min_distance)):
                                        # Only compute if items are sufficiently different

                                        deriv_mu[4] = -kappa*np.sin(pref_angles[:, 0] - item3_theta1)*population_code_response_2D(item3_theta1, item3_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
                                        deriv_mu[5] = -kappa*np.sin(pref_angles[:, 1] - item3_theta2)*population_code_response_2D(item3_theta1, item3_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)

                                        # FI, 2 items, 2 features per item
                                        # for ii in xrange(6):
                                        #     for jj in xrange(6):
                                        #         FI_2d_3obj_search_curr[ii, jj] = np.sum(deriv_mu[ii]*deriv_mu[jj])/(1.*sigma**2.)
                                        FI_2d_3obj_search_curr = np.dot(deriv_mu, deriv_mu.T)/(3.*sigma**2.)

                                        # print "bla"
                                        inv_FI_2d_3obj_search[i*all_angles_1D.size*all_angles_1D.size*all_angles_1D.size + j*all_angles_1D.size*all_angles_1D.size + k*all_angles_1D.size + l, :, :] = np.linalg.inv(FI_2d_3obj_search_curr)

                                    search_progress.increment()
                        else:
                            if search_progress.percentage() % 5.0 < 0.0001:
                                print "%.2f%%, %s left - %s" % (search_progress.percentage(), search_progress.time_remaining_str(), search_progress.eta_str())
                            search_progress.increment()


        # Compute marginal Fisher Info
        if n_items > 1:
            FI_2d_2obj_marginal = np.mean(np.mean(np.ma.masked_equal(FI_2d_2obj_search, 0.0), axis=0), axis=0)
            inv_FI_2d_2obj_marginal_bis = np.mean(np.mean(np.ma.masked_equal(inv_FI_2d_2obj_search, 0.0), axis=0), axis=0)
            inv_FI_2d_2obj_marginal = np.linalg.inv(FI_2d_2obj_marginal)

            # Some stats
            inv_FI_nbitems_effects = [inv_FI_2d_1obj_search[0, 0], inv_FI_2d_2obj_marginal[0, 0], inv_FI_2d_2obj_marginal_bis[0, 0]]
            N_effect_nolocal = [inv_FI_2d_1obj_search[0, 0], inv_FI_2d_2obj_marginal[0, 0]]
            N_effect_withlocal = [inv_FI_2d_1obj_search[0, 0], inv_FI_2d_2obj_marginal_bis[0, 0]]

            if n_items>2:
                # FI_2d_3obj_marginal = np.mean(np.mean(np.mean(np.mean(np.ma.masked_equal(FI_2d_3obj_search, 0.0), axis=0), axis=0), axis=0), axis=0)
                # inv_FI_2d_3obj_marginal = np.linalg.inv(FI_2d_3obj_marginal)
                inv_FI_2d_3obj_marginal_bis = np.mean(np.ma.masked_equal(inv_FI_2d_3obj_search, 0.0), axis=0)

                # inv_FI_nbitems_effects.extend([inv_FI_2d_3obj_marginal[0, 0], inv_FI_2d_3obj_marginal_bis[0, 0]])
                inv_FI_nbitems_effects.extend([inv_FI_2d_3obj_marginal_bis[0, 0]])
                N_effect_nolocal.append(inv_FI_2d_3obj_marginal_bis[0, 0])
                N_effect_withlocal.append(inv_FI_2d_3obj_marginal_bis[0, 0])


            # Some plots
            # pcolor_2d_data(inv_FI_2d_2obj_search[:, :, 0, 0], x=all_angles_1D, y=all_angles_1D, ticks_interpolate=5.)
            pcolor_2d_data(inv_FI_2d_2obj_search[:, :, 0, 0], x=all_angles_1D, y=all_angles_1D, ticks_interpolate=0)

            dataio.save_current_figure('inv_FI_2d_2obj_search_{label}_{unique_id}.pdf')

            plt.figure()
            plt.bar(np.arange(len(inv_FI_nbitems_effects)), inv_FI_nbitems_effects, width=0.5)
            plt.xticks(np.arange(len(inv_FI_nbitems_effects))+0.25, ['1 obj', '2 obj invmean', '2 obj meaninv', '3 obj invmean', '3 obj meaninv'])

            dataio.save_current_figure('bars_if_1_2_3_objects_{label}_{unique_id}.pdf')

            N_effect_withlocal = np.array(N_effect_withlocal)
            N_effect_nolocal = np.array(N_effect_nolocal)


            print N_effect_nolocal
            print N_effect_withlocal

            print "Local effect: ", N_effect_withlocal/N_effect_withlocal[0]

            dataio.save_variables(['N_effect_withlocal', 'N_effect_nolocal', 'inv_FI_2d_2obj_search', 'inv_FI_nbitems_effects', 'FI_2d_2obj_marginal', 'inv_FI_2d_2obj_marginal', 'inv_FI_2d_2obj_marginal_bis', 'FI_2d_3obj_marginal', 'inv_FI_2d_3obj_marginal', 'inv_FI_2d_3obj_marginal_bis', 'FI_2d_1obj_search', 'inv_FI_2d_1obj_search'], locals())

        # plt.figure()
        # plt.semilogy(min_distance_space, inv_FI_search- inv_FI_1_search)

        # plt.figure()
        # plt.semilogy(min_distance_space, inv_FI_search)
        # plt.semilogy(min_distance_space, inv_FI_1_search)

        # plt.figure()
        # plt.plot(min_distance_space, inv_FI_search)

        plt.rcParams['font.size'] = 18

        # plt.figure()
        # plt.semilogy(min_distance_space, (inv_FI_search- inv_FI_1_search)[:, 1:])
        # plt.xlabel('Minimum distance')
        # plt.ylabel('$\hat{I_F}^{-1} - {I_F^{(1)}}^{-1}$')

        if n_items > 2:
            f_table.close()

    if 2 in to_plot:
        ## Change computation style, now define the elements with functions, and provide the inputs at the end

        # 1 obj
        # def inv_FI_1obj(item1_theta1=0.0, item1_theta2=0.0):
        #     deriv_mu = np.zeros((2, N))
        #     deriv_mu[0] = -kappa*np.sin(pref_angles[:, 0] - item1_theta1)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     deriv_mu[1] = -kappa*np.sin(pref_angles[:, 1] - item1_theta2)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     FI_1obj = np.dot(deriv_mu, deriv_mu.T)/sigma**2.
        #     inv_FI_1obj = np.linalg.inv(FI_1obj)
        #     return inv_FI_1obj[0, 0]

        # def inv_FI_2obj(item2_theta1, item2_theta2, item1_theta1=0.0, item1_theta2=0.0):
        #     deriv_mu = np.zeros((4, N))
        #     deriv_mu[0] = -kappa*np.sin(pref_angles[:, 0] - item1_theta1)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     deriv_mu[1] = -kappa*np.sin(pref_angles[:, 1] - item1_theta2)*population_code_response_2D(item1_theta1, item1_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     deriv_mu[2] = -kappa*np.sin(pref_angles[:, 0] - item2_theta1)*population_code_response_2D(item2_theta1, item2_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     deriv_mu[3] = -kappa*np.sin(pref_angles[:, 1] - item2_theta2)*population_code_response_2D(item2_theta1, item2_theta2, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     FI_2obj = np.dot(deriv_mu, deriv_mu.T)/(2.*sigma**2.)
        #     inv_FI_2obj = np.linalg.inv(FI_2obj)
        #     return inv_FI_2obj[0, 0]

        # def inv_FI_nobj(items_thetas):
        #     deriv_mu = np.zeros((items_thetas.size, N))
        #     for i in xrange(items_thetas.size/2):
        #         deriv_mu[2*i] = -kappa*np.sin(pref_angles[:, 0] - items_thetas[2*i])*population_code_response_2D(items_thetas[2*i], items_thetas[2*i+1], pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #         deriv_mu[2*i+1] = -kappa*np.sin(pref_angles[:, 1] - items_thetas[2*i+1])*population_code_response_2D(items_thetas[2*i], items_thetas[2*i+1], pref_angles, N=N, kappa=kappa, amplitude=amplitude)
        #     FI_nobj = np.dot(deriv_mu, deriv_mu.T)/(items_thetas.size/2.*sigma**2.)
        #     # print FI_nobj
        #     inv_FI_nobj = np.linalg.inv(FI_nobj)
        #     return inv_FI_nobj[0, 0]


        def inv_FI_nobj_fixoneitem(items_thetas):
            K = len(items_thetas)
            deriv_mu = np.zeros((2*K+2, N))
            deriv_mu[0] = -kappa*np.sin(pref_angles[:, 0])*population_code_response_2D(0.0, 0.0, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
            deriv_mu[1] = -kappa*np.sin(pref_angles[:, 1])*population_code_response_2D(0.0, 0.0, pref_angles, N=N, kappa=kappa, amplitude=amplitude)
            for i in xrange(K):
                deriv_mu[2*i+2] = -kappa*np.sin(pref_angles[:, 0] - items_thetas[i][0])*population_code_response_2D(items_thetas[i][0], items_thetas[i][1], pref_angles, N=N, kappa=kappa, amplitude=amplitude)
                deriv_mu[2*i+3] = -kappa*np.sin(pref_angles[:, 1] - items_thetas[i][1])*population_code_response_2D(items_thetas[i][0], items_thetas[i][1], pref_angles, N=N, kappa=kappa, amplitude=amplitude)
            FI_nobj = np.dot(deriv_mu, deriv_mu.T)/((K+1.)*sigma**2.)
            # print FI_nobj
            inv_FI_nobj = np.linalg.inv(FI_nobj)
            return inv_FI_nobj[0, 0], FI_nobj[0, 0]


        def enforce_distance_set(new_item, other_items, min_distance=0.001):
            return all(enforce_distance(new_item[0], other_item[0], min_distance=min_distance) and enforce_distance(new_item[1], other_item[1], min_distance=min_distance) for other_item in other_items)

        # Use samples now
        n_samples = int(1e5)
        min_num_samples_std = int(3e3)
        min_distance = 0.1
        n_items = 5
        save_every = 3000

        plot_item2_effect = True

        def inv_FI_one_sample(n_items = 4, return_items=False):
            all_items = [np.array([0.0, 0.0])]
            inv_FI_allobj = np.zeros(n_items)
            FI_allobj = np.zeros(n_items)

            for item_i in xrange(n_items):
                inv_FI_allobj[item_i], FI_allobj[item_i] = inv_FI_nobj_fixoneitem(all_items[1:])

                # Add an extra item
                new_item = -np.pi + 2*np.pi*np.random.random(2)
                while not enforce_distance_set(new_item, all_items, min_distance):
                    new_item = -np.pi + 2*np.pi*np.random.random(2)
                all_items.append(new_item)

            if return_items:
                return inv_FI_allobj, FI_allobj, all_items[1]
            else:
                return inv_FI_allobj, FI_allobj

        inv_FI_allobj_cum = np.zeros(n_items)
        inv_FI_all_obj_var_cum = np.zeros(n_items)
        FI_all_obj_cum = np.zeros(n_items)
        FI_all_obj_var_cum = np.zeros(n_items)
        power_law_fits = np.zeros(2)

        if plot_item2_effect:
            # Also keep the 2nd item positions and all the Inv FI values
            item2_positions = []
            all_inv_FI_2obj = []

        # inv_FI_mean_gen = sum((inv_FI_one_sample() for i in xrange(n_samples)))/float(n_samples)
        search_progress = progress.Progress(n_samples)
        for i in xrange(n_samples):

            if i % 1000 == 0:
                sys.stdout.write("%.1f%%, %s \n %d %s %s %s\n" % (search_progress.percentage(), search_progress.time_remaining_str(), i, inv_FI_allobj_cum*n_samples/(i+1), np.sqrt(inv_FI_all_obj_var_cum*n_samples/(i+1)), power_law_fits))
                sys.stdout.flush()

            # Get sample of invFI and FI
            if plot_item2_effect:
                sample_invFI, sample_FI, new_pos_item2 = inv_FI_one_sample(n_items=n_items, return_items=True)
                item2_positions.append(new_pos_item2)
                all_inv_FI_2obj.append(sample_invFI[1])
            else:
                sample_invFI, sample_FI = inv_FI_one_sample(n_items=n_items)

            # Compute mean
            inv_FI_allobj_cum += sample_invFI/float(n_samples)
            FI_all_obj_cum += 1./sample_FI/float(n_samples)

            # Compute std (wrong but oh well...)
            if i > min_num_samples_std:
                inv_FI_all_obj_var_cum += (sample_invFI-inv_FI_allobj_cum*n_samples/(i+1.))**2./float(n_samples)
                FI_all_obj_var_cum += (1./sample_FI-FI_all_obj_cum*n_samples/(i+1.))**2./float(n_samples)

            # Saves some stuff
            if i % save_every == 0:
                # Estimate power law fit
                power_law_fits = fit_powerlaw(np.arange(1, n_items+1), inv_FI_allobj_cum*n_samples/(i+1))

                dataio.save_variables(['inv_FI_allobj_cum', 'inv_FI_all_obj_var_cum', 'power_law_fits', 'n_samples', 'i', 'min_distance', 'n_items'], locals())

                plt.figure(1)
                width=0.35
                rects1 = plt.bar(np.arange(n_items), (FI_all_obj_cum*n_samples/(i+1.)), width=width, yerr=np.sqrt(FI_all_obj_var_cum*n_samples/(i+1.))/np.sqrt(i+1), error_kw=dict(elinewidth=2, ecolor='black'), hold=False)
                rects2 = plt.bar(np.arange(n_items)+width, inv_FI_allobj_cum*n_samples/(i+1), width=width, yerr=np.sqrt(inv_FI_all_obj_var_cum*n_samples/(i+1))/np.sqrt(i+1), color='r', error_kw=dict(elinewidth=2, ecolor='black'))
                plt.xticks(np.arange(n_items)+width, ['%d obj' % it for it in xrange(1, n_items+1)])
                # plt.errorbar(np.arange(1, n_items+1), inv_FI_allobj_cum, yerr=inv_FI_all_obj_var_cum, fmt='.')
                plt.legend((rects1[0], rects2[0]), ('Without local effect', 'With local effect'), loc='best')
                plt.ylabel('Inverse marginal fisher information [$rad^{-2}$]')
                dataio.save_current_figure('bars_IF_%dobj_{label}_{unique_id}.pdf' % (n_items))

                if i > 10 and plot_item2_effect:
                    plt.figure(2)

                    min_distance_fct = lambda x, min_distance=0.1: np.abs(x) < min_distance
                    min_distance_fct_part = functools.partial(min_distance_fct, min_distance=min_distance)

                    contourf_interpolate_data(np.array(item2_positions), np.array(all_inv_FI_2obj), xlabel='$\phi_2$', ylabel='$\psi_2$', title='Inverse Fisher Information for $\phi_1 = \psi_1 = 0$, varying $\phi_2, \psi_2$', fignum=3, show_colorbar=False, show_scatter=False, mask_x_condition=min_distance_fct_part, mask_y_condition=min_distance_fct_part, interpolation_method='nearest')

                    dataio.save_current_figure('contourf_IF_2objeffect_{label}_{unique_id}.png')


            search_progress.increment()

        # inv_FI_mean_np = inv_FI_allsamples()



    plt.show()

    say_finished(additional_comment=additional_comment)

    return locals()


if __name__ == '__main__':

    all_vars = main(to_plot = [1, 2])

    for var_to_reinst in all_vars:
        vars()[var_to_reinst] = all_vars[var_to_reinst]





