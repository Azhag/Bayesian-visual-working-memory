#!/bin/sh
#

curr_dir=`pwd`
path_source=/nfs/home2/lmatthey/Documents/work/Visual_working_memory/code/git-bayesian-visual-working-memory
launcher=gibbs_sampler_continuous_fullcollapsed_randomfactorialnetwork.py
action='do_save_responses_simultaneous'
nb_inputs_experiments=1

nb_repeats=4
N=200
T=6
alpha=1.0
num_samples=50
select_num_samples=1

#for num_samples in 1 10 20 50; do
#    echo "num_samples: $num_samples"
#    for select_num_samples in 1 10 20 50; do
#        if [    $num_samples -lt  $select_num_samples ]; then
#            continue
#        fi
#        
#        echo "selection_num_Samples: $select_num_samples"
        for rc_scale in 0.1 0.25 0.5 1.0 2.0 4.0 5.0; do 
            echo "rc_scale: ${rc_scale}"
            for rc_scale2 in 0.1 0.25 0.5 1.0 2.0 4.0 5.0; do 
                echo "rc_scale2: ${rc_scale2}"
                for ratio_conj in 0.0 0.2 0.5 0.7 1.0; do 
                    echo "ratio_conj: ${ratio_conj}"
                    for sigmax in 0.8 1.0 1.2 1.5 2.0 3.0; do
                        echo "sigmax: ${sigmax}"

                file_suffix=fit_fullsamples_simult_mixed_sigmax_rcsclase_numsamples-sigmax${sigmax}rcscale${rc_scale}rcscale2${rc_scale2}ratio_conj${ratio_conj}T${T}alpha${alpha}N${N}numsamples${num_samples}selectnumsamples${select_num_samples}nbrepeats${nb_repeats}nbexperiments${nb_inputs_experiments}

        cat > script_train_${file_suffix} << EOF
#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l mem=2gb
#

export PATH="/nfs/home1/ucabjga/opt/epd-7.0-2-rh5-x86_64/bin/":$PATH
export OMP_NUM_THREADS=1

cd ${curr_dir}

hostn=\`hostname\`
echo "Job execution host: \$hostn" 

# If some nodes are messing with you, exclude them
if [ \$hostn = "behemoth_" -o \$hostn = "wood_" -o \$hostn = "zhora__" ]; then
    qsub script_train_${file_suffix}
    #id=\`echo \$PBS_JOBID | sed s/.eldon//g\`
    #rm *$id
    exit 0
fi;

nice python ${path_source}/${launcher} --action_to_do ${action} --code_type mixed --output_directory . --N ${N} --num_repetitions ${nb_repeats} --T ${T} --label ${file_suffix} --alpha ${alpha} --rc_scale ${rc_scale} --num_samples ${num_samples} --selection_num_samples ${select_num_samples} --sigmax ${sigmax} --rc_scale2 ${rc_scale2} --ratio_conj ${ratio_conj}

# Compress the results files
#tar -czf file_${file_suffix}.tar.gz *${file_suffix}-*
#rm *${file_suffix}-*
EOF
        
        #
        # Make the script executable (useful for testing)
        #
        chmod a+x script_train_${file_suffix}
    
        #
        # Submit the job 
        #
        # This command submits the same job multiple time, quite fast and convenient. But breaks qstat IDENTIFIER.
        # qsub script_train_${file_suffix} -t 1-${nb_inputs_experiments}
        qsub script_train_${file_suffix}
        #
                    done
                done
            done
        done
    #done
#done
