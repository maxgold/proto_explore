#!/bin/bash
#declare -a METHODS=("proto" "diayn" "aps" "rnd")
#declare -a METHODS=("icm" "proto" "diayn" "icm_apt" "ind_apt" "aps" "smm" "rnd" "disagreement")
agent=$1
num_protos=$2
pred_dim=$3
proj_dim=$4
hidden_dim=$5
seed=$6
#sd_sched=$6
#sd_clip=$7
#read -r -d '' msg <<EOT
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pmm_$agent.$num_protos.$pred_dim.$proj_dim.$hidden_dim.$seed
#SBATCH --open-mode=append
#SBATCH --output=/scratch/nm1874/output/%j_%x.out
#SBATCH --error=/scratch/nm1874/output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nm1874@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 10:00:00

export PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nm1874/.mujoco/mujoco210/bin
export MJLIB_PATH=/ext3/mujoco210/bin/libmujoco210.so

singularity \
    exec --nv \
    --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    --overlay /vast/nm1874/dm_control_2022/proto.ext3:ro \
    /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate /ext3/proto
cd /vast/nm1874/dm_control_2022/proto_explore/url_benchmark/python pretrain.py agent=$agent domain=point_mass_maze batch_size=256 num_protos=$num_protos pred_dim=$pred_dim proj_dim=$proj_dim goal=False obs_type=pixels use_wandb=True const_init=True num_seed_frames=4000 replay_buffer_size=1000000 hidden_dim=$hidden_dim tmux_session=batch_greene seed=$seed"

EOT
#echo $msg

