#!/bin/bash
#declare -a METHODS=("proto" "diayn" "aps" "rnd")
#declare -a METHODS=("icm" "proto" "diayn" "icm_apt" "ind_apt" "aps" "smm" "rnd" "disagreement")
offset=$1
offline_model_step=$2
#read -r -d '' msg <<EOT
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pmm_$task
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
#SBATCH -t 2:00:00
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
cd /vast/nm1874/dm_control_2022/proto_explore/url_benchmark/
python pph_general.py agent=ddpg domain=point_mass_maze batch_size=256 num_protos=16 pred_dim=16 proj_dim=512 goal=False obs_type=pixels use_wandb=True \
num_seed_frames=2000 replay_buffer_size=1000000 hidden_dim=256 seed=0 gc_only=True load_encoder=True inv=True feature_dim_gc=50 encoder1=True sl=False \
use_critic_trunk=True offline_gc=True expert_buffer=False offline_model_step=$offline_model_step greene=True init_from_proto=True pretrained_feature_dim=16 \
eval_every_frames=100000 model_path=/exp_local/2023.03.06/144540_proto/optimizer_proto_1000000.pth goal_offset=$offset tmux_session=greene_batch frame_stack=1 \
feature_dim=50 egocentric=False camera_id=0 debug=True"
EOT
#echo $msg
