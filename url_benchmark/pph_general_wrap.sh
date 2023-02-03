#!/bin/bash
#declare -a METHODS=("proto" "diayn" "aps" "rnd")
#declare -a METHODS=("icm" "proto" "diayn" "icm_apt" "ind_apt" "aps" "smm" "rnd" "disagreement")
offset=$1
seed=$2
#sd_sched=$6
#sd_clip=$7
#read -r -d '' msg <<EOT
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=pmm_$task.$seed
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
python pphg_gc_only.py agent=proto_inv task_no_goal=point_mass_maze_reach_room_no_goal task=point_mass_maze_reach_custom_goal_room domain=point_mass_maze obs_type=pixels goal=True replay_buffer_gc=500000 num_train_frames=4000010 num_seed_frames=4000 replay_buffer_num_workers=4 num_protos=16 pred_dim=16 feature_dim=16 hidden_dim=256 batch_size=256 batch_size_gc=256 use_wandb=True eval_every_frames=200000 episode_length=500 lr=.0001 frame_stack=1 proto_goal_med=False proto_goal_intr=False proto_goal_random=True og=False combine_storage_gc=False switch_gc=100000 stddev_schedule=.3 stddev_clip=.2 stddev_schedule2=.3 stddev_clip2=.2 hybrid_gc=True hybrid_pct=.7 test1=True nstep1=1 nstep2=3 update_proto_opt=True update_proto_while_gc=True update_gc_while_proto=True domain=point_mass_maze normalize=False normalize2=True offline_gc=True goal_offset=$offset model_path=/exp_local/2023.01.26/094234_proto_encoder1/ greene=True tmux_session=batch seed=$seed"

EOT
#echo $msg

