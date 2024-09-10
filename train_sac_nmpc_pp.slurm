#!/bin/sh

#SBATCH --job-name=train_drl
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-itk
#SBATCH --time=0-40:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=R-%x.%j.out


helpFunction()
{
   echo ""
   echo "Usage: $0 -bd base_dir -bs batch_size -lr learning_rate -ncpus ncpus -gs gradient_steps -tf train_freq -bfs buffer_size -en experiment_name"
   echo -e "\t-bd base directory for storing models, logs etc"
   echo -e "\t-bs batch size"
   echo -e "\t-lr learning rate"
   echo -e "\t-ncpus number of cpu cores to use"
   echo -e "\t-gs gradient steps in training"
   echo -e "\t-tf training frequency in number of steps"
   echo -e "\t-bfs replay buffer size"
   echo -e "\t-en experiment name"
   exit 1 # Exit script after printing help
}


BASE_DIR="/cluster/work/trymte/sac_rlmpc_dir"
BATCH_SIZE=128
LEARNING_RATE=0.0004
NCPUS=10 # Equal to number of training environments
TRAIN_FREQ=8
GRADIENT_STEPS=2
BUFFER_SIZE=30000
MAX_NUM_TRAIN_EPS=100 # Maximum number of loaded training episodes
MAX_NUM_EVAL_EPS=10 # Maximum number of loaded evaluation episodes
EXPERIMENT_NAME="sac_nmpc_1"
TIMESTEPS=10_000_000

setArgs(){
  while [ "${1:-}" != "" ]; do
    case "$1" in
      "-bd" | "--base_dir")
        shift
        BASE_DIR=$1
        ;;
      "-bs" | "--batch_size")
	shift
        BATCH_SIZE=$1
        ;;
      "-lr" | "--learning_rate")
	shift
	LEARNING_RATE=$1
        ;;
      "-ncpus" | "--ncpus")
	shift
	NCPUS=$1
	;;
      "-gs" | "--gradient_steps")
	shift
	GRADIENT_STEPS=$1
	;;
      "-tf" | "--train_freq")
	shift
	TRAIN_FREQ=$1
	;;
      "-bfs" | "--buffer_size")
	shift
	BUFFER_SIZE=$1
	;;
      "-en" | "--experiment_name")
	shift
	EXPERIMENT_NAME=$1
	;;
      "-t" | "--timesteps")
	shift
	TIMESTEPS=$1
	;;
      *)
	helpFunction
	;;
    esac
    shift
  done
}
setArgs "$@"

echo "BASE_DIR = $BASE_DIR"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LEARNING_RATE = $LEARNING_RATE"
echo "NCPUS = $NCPUS"
echo "GRADIENT_STEPS = $GRADIENT_STEPS"
echo "TRAIN_FREQ = $TRAIN_FREQ"
echo "BUFFER_SIZE = $BUFFER_SIZE"
echo "EXPERIMENT_NAME = $EXPERIMENT_NAME"
echo "TIMESTEPS = $TIMESTEPS"
echo "MAX_NUM_TRAIN_EPS = $MAX_NUM_TRAIN_EPS"
echo "MAX_NUM_EVAL_EPS = $MAX_NUM_EVAL_EPS"

echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

# loading required modules
module purge
module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load mpi4py/3.1.4-gompi-2023a
module load TensorFlow/2.13.0-foss-2023a
module load Rust/1.70.0-GCCcore-12.3.0
module load GDAL/3.7.1-foss-2023a
module load FFmpeg/6.0-GCCcore-12.3.0

# enable virtual env
source /cluster/home/trymte/Desktop/mpy3.11.3/bin/activate

# setting the matplotlib backend to be headless
export MPLBACKEND="Agg"

# setting the UCX logging level
export UCX_LOG_LEVEL=error

# debugging
module list
echo "which python: "
which python
echo -e "\n\n"

# using srun to launch the MPI job, --ntasks below must equal --nodes in #SBATCH above
srun --ntasks=1 python /cluster/home/trymte/Desktop/rlmpc/tests/test_train_sac_with_nmpc_pp_policy.py --base_dir=$BASE_DIR --learning_rate=$LEARNING_RATE --buffer_size=$BUFFER_SIZE --train_freq=$TRAIN_FREQ --gradient_steps=$GRADIENT_STEPS --timesteps=$TIMESTEPS --experiment_name=$EXPERIMENT_NAME --batch_size=$BATCH_SIZE --n_training_envs=$NCPUS --max_num_loaded_train_scen_episodes=$MAX_NUM_TRAIN_EPS --max_num_loaded_eval_scen_episodes=$MAX_NUM_EVAL_EPS

# running the analysis script sequentially after the srun command has finished


# fetching the job information
job_info=$(sacct --format=JobID,JobName,Elapsed,State,ExitCode --jobs=${SLURM_JOB_ID} --parsable2)

# constructing the email body
email_body=$(cat <<EOF
Slurm Job Summary:
$job_info

EOF
)

# sending the log file via email with the job information in the email body
echo "$email_body" | mailx -s "SLURM Job ${SLURM_JOB_NAME} Completed" -a R-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out trym.tengesdal@ntnu.no
