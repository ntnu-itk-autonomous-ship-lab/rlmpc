#!/bin/sh



#SBATCH --job-name=standard_tsac_nmpc_pp
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-itk
#SBATCH --time=20-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=250G
#SBATCH --output=idun/R-%x.%j.out


CURRENT_DATE_TIME="`date +%d%m%y`"
BASE_DIR="$HOME/sac_rlmpc_dir"
BATCH_SIZE=256
LEARNING_RATE=5e-6 #0.00002
TAU=0.005
SDE_SAMPLE_FREQ=20
NCPUS=20 # Equal to number of training environments
N_EVAL_ENVS=4
TRAIN_FREQ=8
EVAL_FREQ=10000
GRADIENT_STEPS=1
BUFFER_SIZE=200000
DEVICE="cpu"
MAX_NUM_TRAIN_EPS=50 # Maximum number of loaded training episodes
MAX_NUM_EVAL_EPS=10 # Maximum number of loaded evaluation episodes
N_EVAL_EPS=10
TIMESTEPS=10_000_000
N_TIMESTEPS_PER_LEARN=10000
LOAD_MODEL_PATH="" #"sac_gsde20_rd_050225_8e_6lr_0005tau_s1_jid21125646/models/sac_gsde20_rd_050225_8e_6lr_0005tau_s1_jid21125646_71680_steps"
SEED=1

EXPERIMENT_NAME="sac_gsde_50eps_rd_${CURRENT_DATE_TIME}_5e_6lr_0005tau_s${SEED}_jid${SLURM_JOB_ID}"

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
source $HOME/Desktop/mpy3.11.3/bin/activate

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
srun --ntasks=1 python $HOME/Desktop/rlmpc/run_examples/run_training_sac_standard_nmpc_pp.py --base_dir=$BASE_DIR --learning_rate=$LEARNING_RATE --buffer_size=$BUFFER_SIZE --train_freq=$TRAIN_FREQ --gradient_steps=$GRADIENT_STEPS --timesteps=$TIMESTEPS --experiment_name=$EXPERIMENT_NAME --batch_size=$BATCH_SIZE --n_training_envs=$NCPUS --max_num_loaded_train_scen_episodes=$MAX_NUM_TRAIN_EPS --max_num_loaded_eval_scen_episodes=$MAX_NUM_EVAL_EPS --n_eval_envs=$N_EVAL_ENVS --n_eval_episodes=$N_EVAL_EPS --n_timesteps_per_learn=$N_TIMESTEPS_PER_LEARN --eval_freq=$EVAL_FREQ --load_model_path=$LOAD_MODEL_PATH --device=$DEVICE --reset_num_timesteps --seed=$SEED --tau=$TAU --sde_sample_freq=$SDE_SAMPLE_FREQ


# fetching the job information
job_info=$(sacct --format=JobID,JobName,Elapsed,State,ExitCode --jobs=${SLURM_JOB_ID} --parsable2)

# constructing the email body
email_body=$(cat <<EOF
Slurm Job Summary:
$job_info

EOF
)
