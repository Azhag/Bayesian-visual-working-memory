#!/bin/bash
#!
#! Example SLURM job script for Darwin (Sandy Bridge, ConnectX3)
#! Last updated: Fri Jan 24 12:12:18 GMT 2014
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
{pbs_options}
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=1
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p sandybridge

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by node number*walltime. Allocation is in entire nodes.
#! The --ntasks value refers to the number of tasks to be launched by SLURM only. This
#! usually equates to the number of MPI tasks launched. Reduce this from nodes*16 if
#! demanded by memory requirements, or if OMP_NUM_THREADS>1.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module load default-impi                   # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

#! Full path to application executable:
application="{cmd}"

#! Run options for the application:
options=

#! Work directory (i.e. where the job will run):
#workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
#                             # in which sbatch is run.
workdir="{working_dir}"


#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 16:
export OMP_NUM_THREADS=2

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${{numnodes}}*${{mpi_tasks_per_node}}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
#! Notes:
#! 1. These variables influence Intel MPI only.
#! 2. Domains are non-overlapping sets of cores which map 1-1 to MPI tasks.
#! 3. I_MPI_PIN_PROCESSOR_LIST is ignored if I_MPI_PIN_DOMAIN is set.
#! 4. If MPI tasks perform better when sharing caches/sockets, try I_MPI_PIN_ORDER=compact.


#! Uncomment one choice for CMD below (add mpirun/mpiexec options if necessary):

#! Choose this for a MPI code (possibly using OpenMP) using Intel MPI.
#CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code (possibly using OpenMP) using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"
echo "Filename: {filename}"
T="$(date +%s)"

if [ "$SLURM_JOB_NODELIST" ]; then
#! Create a machine file:
export NODEFILE=`generate_pbs_nodefile`
cat $NODEFILE | uniq > machine.file.$JOBID
echo -e "\nNodes allocated:\n================"
echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

echo "+++ Job completed +++"
T="$(($(date +%s)-T))"
printf "Elapsed time: %02d:%02d:%02d\n" "$((T/3600))" "$((T/60%60))" "$((T%60))"
