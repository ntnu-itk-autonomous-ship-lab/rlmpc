#!/bin/sh



#SBATCH --job-name=train_snmpc
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-itk
#SBATCH --time=10-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=idun/R-%x.%j.out

BASE_DIR="/cluster/work/trymte/sac_rlmpc_dir"
BATCH_SIZE=128
LEARNING_RATE=0.0002
NCPUS=16 # Equal to number of training environments
N_EVAL_ENVS=4
TRAIN_FREQ=16
GRADIENT_STEPS=4
BUFFER_SIZE=300000
DEVICE="cpu"
MAX_NUM_TRAIN_EPS=200 # Maximum number of loaded training episodes
MAX_NUM_EVAL_EPS=8 # Maximum number of loaded evaluation episodes
N_EVAL_EPS=8
EXPERIMENT_NAME="snmpc_db_high_${MAX_NUM_TRAIN_EPS}te_${MAX_NUM_EVAL_EPS}ee_${NCPUS}cpus"
TIMESTEPS=10_000_000
N_TIMESTEPS_PER_LEARN=20000
LOAD_MODEL_NAME=""

# show available resources
sinfo -o "%10P %5D %34N  %5c  %7m  %47f  %23G"


# echo "BASE_DIR = $BASE_DIR"
# echo "BATCH_SIZE = $BATCH_SIZE"
# echo "LEARNING_RATE = $LEARNING_RATE"
# echo "NCPUS = $NCPUS"
# echo "GRADIENT_STEPS = $GRADIENT_STEPS"
# echo "TRAIN_FREQ = $TRAIN_FREQ"
# echo "BUFFER_SIZE = $BUFFER_SIZE"
# echo "EXPERIMENT_NAME = $EXPERIMENT_NAME"
# echo "TIMESTEPS = $TIMESTEPS"
# echo "MAX_NUM_TRAIN_EPS = $MAX_NUM_TRAIN_EPS"
# echo "MAX_NUM_EVAL_EPS = $MAX_NUM_EVAL_EPS"

echo "Slurm run directory: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID is $SLURM_JOB_ID"
echo "Job running on nodes: $SLURM_JOB_NODELIST"
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
module load Clang/16.0.6-GCCcore-12.3.0

# enable virtual env
source /cluster/home/trymte/Desktop/mpy3.11.3/bin/activate

# setting the matplotlib backend to be headless
export MPLBACKEND="Agg"

# setting the UCX logging level
export UCX_LOG_LEVEL=error
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

# debugging
# module list
# echo "which python: "
# which python
# echo -e "\n\n"

# # Get all jobs
# squeue

# # get all jobs for user < only pending | only running > in <partition>
# squeue -u username <-t PENDING|-t RUNNING> <-p partition>

# Show detailed info on <jobid>
scontrol show jobid -dd $SLURM_JOB_ID

# # cancel specific <jobid>
# scancel < jobid >

# # cancel all <pending> jobs for <username>
# scancel <-t PENDING> -u <username>

# using srun to launch the MPI job, --ntasks below must equal --nodes in #SBATCH above
srun --ntasks=1 python /cluster/home/trymte/Desktop/rlmpc/tests/test_train_sac_with_nmpc_pp_policy.py --base_dir=$BASE_DIR --learning_rate=$LEARNING_RATE --buffer_size=$BUFFER_SIZE --train_freq=$TRAIN_FREQ --gradient_steps=$GRADIENT_STEPS --timesteps=$TIMESTEPS --experiment_name=$EXPERIMENT_NAME --batch_size=$BATCH_SIZE --n_training_envs=$NCPUS --max_num_loaded_train_scen_episodes=$MAX_NUM_TRAIN_EPS --max_num_loaded_eval_scen_episodes=$MAX_NUM_EVAL_EPS --n_eval_envs=$N_EVAL_ENVS --n_eval_episodes=$N_EVAL_EPS --n_timesteps_per_learn=$N_TIMESTEPS_PER_LEARN --load_model_name=$LOAD_MODEL_NAME --device=$DEVICE --load_critics --reset_num_timesteps


# fetching the job information
job_info=$(sacct --format=JobID,JobName,Elapsed,State,ExitCode --jobs=${SLURM_JOB_ID} --parsable2)

# constructing the email body
email_body=$(cat <<EOF
Slurm Job Summary:
$job_info

EOF
)

# sending the log file via email with the job information in the email body
# echo "$email_body" | mailx -s "SLURM Job ${SLURM_JOB_NAME} Completed" -a R-${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out trym.tengesdal@ntnu.no
