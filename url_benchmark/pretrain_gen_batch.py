
import os


if __name__ == '__main__':
    run_dir = "run_all"
    greene_config_path= "/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/run_exps/greene"
    write_path = "/".join(os.path.abspath(__file__).split('/')[:-1] + ['run_exps/greene'])
    filename_base_str = "run_offline_{agent}_{bs}_{hpct}_lr{lr}.slurm"
    base_str = \
"""#!/bin/bash
#SBATCH --job-name=run_offline_{agent}_{bs}_{hpct}_lr{lr}
#SBATCH --open-mode=append
#SBATCH --output=/scratch/nm1874/output/%j_%x.out
#SBATCH --error=/scratch/nm1874/output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nm1874@nyu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
singularity exec --nv --bind /usr/share/glvnd/egl_vendor.d/10_nvidia.json --overlay /vast/nm1874/dm_control_2022/urlb.ext3:ro /scratch/work/public/singularity/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c \"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nm1874/.mujoco/mujoco210/bin
export MJLIB_PATH=/ext3/mujoco210/bin/libmujoco210.so
export PATH=/vast/nm1874/dm_control_2022/proto_explore/url_benchmark
source /ext3/env.sh
conda activate /ext3/urlb
python /vast/nm1874/dm_control_2022/proto_explore/url_benchmark/pretrain_pixel_proto_goal.py agent={agent} domain=point_mass_maze obs_type=pixels goal=True replay_buffer_gc=1000000 num_train_frames=2000010 num_seed_frames=4000 replay_buffer_num_workers=2 hidden_dim=1024 batch_size_gc={bs} batch_size=256 update_gc=2 sample_proto=True use_wandb=True tmux_session=batch hybrid_gc=True hybrid_pct={hpct} load_encoder=False load_proto=True lr={lr} sample_proto=False reward_euclid=False reward_scores=True eval_every_frames=100000 seed=1 curriculum=False
\"
"""
    batch_size = [512,1024, 2048]
    agents=['proto_goal_gc_encoder', 'proto_goal']
    hybrid_pct=[0,5,8]
    lrs = [.0001, .0005, .001]


    for bs in batch_size:
        for agent in agents:
            for hpct in hybrid_pct:
                for lr in lrs:
                    
                    arg_dict = {
                           'agent' : agent,
                           'bs': bs,
                           'hpct':hpct,
                           'lr':lr,
                           }
                    
                    out_str = base_str.format(**arg_dict)
                    with open(os.path.join(write_path, filename_base_str.format(**arg_dict)), 'w') as f:
                        f.write(out_str)

    run_strs = []                                             #
    for bs in batch_size:
        run_strs_agent = []
        for agent in agents:
            for hpct in hybrid_pct:
                for lr in lrs:
                    arg_dict = {
                           'agent' : agent,
                           'bs': bs,
                           'hpct':hpct,
                           'lr':lr,
                            }
                    
                    filename = filename_base_str.format(**arg_dict)
                    run_strs_agent.append("sbatch --array=0 {}/{}".format(greene_config_path, filename))

                with open(os.path.join(write_path,"run_{}_{}_{}.sh".format(agent, hpct, lr)), "w") as f:
                    f.write('#!/bin/bash\n')
                    for r in run_strs_agent:
                        line = "{}\n".format(r)
                        f.write(line)
                os.chmod(os.path.join(write_path,"run_{}_{}_{}.sh".format(agent, hpct, lr)), 0o775)

                run_strs.extend(run_strs_agent)

    with open(os.path.join(write_path,"run_all.sh"), "w") as f:
        f.write('#!/bin/bash\n')
        for r in run_strs:
            line = "{}\n".format(r)
            f.write(line)
    os.chmod(os.path.join(write_path,"run_all.sh"), 0o775)
