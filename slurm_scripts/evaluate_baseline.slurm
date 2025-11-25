#!/bin/sh



#SBATCH --job-name=baseline_snmpc_eval
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-itk
#SBATCH --time=15-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=idun/R-%x.%j.out


CURRENT_DATE_TIME="`date +%H%M_%d%m%y`"
BASE_DIR="$HOME/sac_rlmpc_dir"
NCPUS=4 # Equal to number of training environments
MAX_NUM_EVAL_EPS=16 # Maximum number of loaded evaluation episodes
N_EVAL_EPS=16
LOAD_MODEL_NAME="" #"snmpc_nd_200te_32ee_seed1_jid20743023/models/snmpc_nd_200te_32ee_seed1_jid20743023_71680_steps"
SEED=1

EXPERIMENT_NAME="baseline_snmpc_eval_${CURRENT_DATE_TIME}_${MAX_NUM_EVAL_EPS}te_s${SEED}_jid${SLURM_JOB_ID}"

# show available resources
sinfo -o "%10P %5D %34N  %5c  %7m  %47f  %23G"

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

# # get all jobs for user < only pending | only running > in <partition>
# squeue -u username <-t PENDING|-t RUNNING> <-p partition>

# Show detailed info on <jobid>
scontrol show jobid -dd $SLURM_JOB_ID

# # cancel specific <jobid>
# scancel < jobid >

# # cancel all <pending> jobs for <username>
# scancel <-t PENDING> -u <username>

# using srun to launch the MPI job, --ntasks below must equal --nodes in #SBATCH above
srun --ntasks=1 python $HOME/Desktop/rlmpc/rlmpc/scripts/evaluate_agent.py --base_dir=$BASE_DIR --experiment_name=$EXPERIMENT_NAME --n_eval_episodes=$N_EVAL_EPS --seed=$SEED --n_cpus=$NCPUS


# fetching the job information
job_info=$(sacct --format=JobID,JobName,Elapsed,State,ExitCode --jobs=${SLURM_JOB_ID} --parsable2)

# constructing the email body
email_body=$(cat <<EOF
Slurm Job Summary:
$job_info

EOF
)
