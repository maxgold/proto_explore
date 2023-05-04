import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
import os
import pandas as pd
# from eval_ops import *
import wandb


def heatmaps(state_visitation_gc, reward_matrix_gc, goal_state_matrix, state_visitation_proto, proto_goals_matrix, 
global_step, gc=False, proto=False, v_queue_ptr=None, v_queue=None):
    # this only works for 2D mazesf

    if gc:
        heatmap = state_visitation_gc

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(global_step)

        plt.savefig(f"./{global_step}_gc_heatmap.png")
        wandb.save(f"./{global_step}_gc_heatmap.png")

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(np.log(1 + reward_matrix_gc.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(global_step)

        plt.savefig(f"./{global_step}_gc_reward.png")
        wandb.save(f"./{global_step}_gc_reward.png")

        goal_matrix = goal_state_matrix
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))
        # labels = np.round(goal_matrix.T/goal_matrix.sum()*100, 1)
        sns.heatmap(np.log(1 + goal_matrix.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(global_step)

        plt.savefig(f"./{global_step}_goal_state_heatmap.png")
        wandb.save(f"./{global_step}_goal_state_heatmap.png")

    if proto:
        heatmap = state_visitation_proto

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(global_step)

        plt.savefig(f"./{global_step}_proto_heatmap.png")
        wandb.save(f"./{global_step}_proto_heatmap.png")

        heatmap = proto_goals_matrix

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(global_step)

        plt.savefig(f"./{global_step}_proto_goal_heatmap.png")
        wandb.save(f"./{global_step}_proto_goal_heatmap.png")

        # ########################################################
        # # exploration moving avg
        # indices = [5, 10, 20, 50]
        # sets = [mov_avg_5, mov_avg_10, mov_avg_20, mov_avg_50]

        # if global_step % 100000 == 0:

        #     plt.clf()
        #     fig, ax = plt.subplots(figsize=(15, 5))
        #     labels = ['mov_avg_5', 'mov_avg_10', 'mov_avg_20', 'mov_avg_50']

        #     for ix, x in enumerate(indices):
        #         ax.plot(np.arange(0, sets[ix].shape[0]), sets[ix], label=labels[ix])
        #     ax.legend()

        #     plt.savefig(f"proto_moving_avg_{global_step}.png")
        #     wandb.save(f"proto_moving_avg_{global_step}.png")

        # ##########################################################
        # # reward moving avg
        # sets = [r_mov_avg_5, r_mov_avg_10, r_mov_avg_20,
        #         r_mov_avg_50]

        # if global_step % 100000 == 0:

        #     plt.clf()
        #     fig, ax = plt.subplots(figsize=(15, 5))
        #     labels = ['mov_avg_5', 'mov_avg_10', 'mov_avg_20', 'mov_avg_50']

        #     for ix, x in enumerate(indices):
        #         ax.plot(np.arange(0, sets[ix].shape[0]), sets[ix], label=labels[ix])
        #     ax.legend()

        #     plt.savefig(f"gc_reward_moving_avg_{global_step}.png")
        #     wandb.save(f"gc_reward_moving_avg_{global_step}.png")

def save_stats_visitation(cfg, work_dir, global_step, state_visitation_proto, v_queue_ptr, v_queue):
    
    total_v = np.count_nonzero(state_visitation_proto)
    v_ptr = v_queue_ptr
    v_queue[v_ptr] = total_v
    v_queue_ptr = (v_ptr+1) % v_queue.shape[0]
    if cfg.debug:
        every = 1000
    else:
        every = 100000

    if global_step%every==0:
        df = pd.DataFrame()
        # import IPython as ipy; ipy.embed(colors='neutral')
        df[['visitation']] = v_queue
        path = os.path.join(work_dir, 'exploration_{}_{}.csv'.format(str(cfg.agent.name),global_step))
        df.to_csv(path, index=False)
    return v_queue_ptr, v_queue

            
def save_stats(cfg, work_dir, global_step, state_visitation_proto, reward_matrix_gc, pmm, v_queue_ptr, v_queue, r_queue_ptr, r_queue, count, 
mov_avg_5, mov_avg_10, mov_avg_20, mov_avg_50, r_mov_avg_5, r_mov_avg_10, r_mov_avg_20, r_mov_avg_50):
    #NOT USING

    #record changes in proto heatmap
    if global_step%1000==0 and global_step>5000:

        if pmm:

            total_v = np.count_nonzero(state_visitation_proto)
            print('total visitation', total_v)
            v_ptr = v_queue_ptr
            v_queue[v_ptr] = total_v
            v_queue_ptr = (v_ptr+1) % v_queue.shape[0]

            indices=[5,10,20,50]
            sets = [mov_avg_5, mov_avg_10, mov_avg_20,
                    mov_avg_50]

            for ix,x in enumerate(indices):
                if v_queue_ptr-x<0:
                    lst = np.concatenate([v_queue[:v_queue_ptr], v_queue[v_queue_ptr-x:]], axis=0)
                    sets[ix][count]=lst.mean()
                else:
                    sets[ix][count]=v_queue[v_queue_ptr-x:v_queue_ptr].mean()

            total_r = np.count_nonzero(reward_matrix_gc)
            print('total reward', total_r)
            r_ptr = r_queue_ptr
            r_queue[r_ptr] = total_r
            r_queue_ptr = (r_ptr+1) % r_queue.shape[0]

            sets = [r_mov_avg_5, r_mov_avg_10, r_mov_avg_20,
                    r_mov_avg_50]

            for ix,x in enumerate(indices):
                if r_queue_ptr-x<0:
                    lst = np.concatenate([r_queue[:r_queue_ptr], r_queue[r_queue_ptr-x:]], axis=0)
                    sets[ix][count]=lst.mean()
                else:
                    sets[ix][count]=r_queue[r_queue_ptr-x:r_queue_ptr].mean()


            count+=1

    #save stats
    #change to 100k when not testing
    if global_step%100000==0:
        df = pd.DataFrame()
        if pmm:
            df['mov_avg_5'] = mov_avg_5
            df['mov_avg_10'] = mov_avg_10
            df['mov_avg_20'] = mov_avg_20
            df['mov_avg_50'] = mov_avg_50
        df['r_mov_avg_5'] = r_mov_avg_5
        df['r_mov_avg_10'] = r_mov_avg_10
        df['r_mov_avg_20'] = r_mov_avg_20
        df['r_mov_avg_50'] = r_mov_avg_50
        path = os.path.join(work_dir, 'exploration_{}_{}.csv'.format(str(cfg.agent.name),global_step))
        df.to_csv(path, index=False)
        
        
def gc_or_proto(actor, actor1, proto_explore_count, gc_init):

    if proto_explore_count <= 25 and actor:
        actor1=False
        actor=True
        proto_explore_count+=1

    elif actor and proto_explore_count > 25:
        actor1=True
        actor=False
        proto_explore_count=0

    if proto_explore_count >= 25 and gc_init==False:
        gc_init=True

    return actor, actor1, proto_explore_count, gc_init
        
        
def make_env(cfg, actor1, init_idx, goal_state, pmm, current_init, train_env=None, train_env1=None, train_env_no_goal=None):


    if pmm:
        if goal_state is not None:
            goal_state = goal_state[:2]
        if init_idx is None:

            init_state = np.random.uniform(.25,.29,size=(2,))
            init_state[0] = init_state[0]*(-1)

        else: 

            init_state = current_init[init_idx,:2]

        if actor1:

            time_step1 = train_env1.reset(goal_state=goal_state, init_state = init_state)
            time_step_no_goal = train_env_no_goal.reset(goal_state=np.array([25,25]), init_state=time_step1.observation['observations'][:2])
            print('no goal', train_env_no_goal.physics.get_state())

        else:

            time_step = train_env.reset(goal_state=np.array([25,25]), init_state=init_state)
            print('init', init_state)
            print('proto reset')

        origin = init_state
    else:

        if init_idx is None:

            time_step = train_env.reset()
            origin = train_env.physics.get_state()

        else: 

            time_step = train_env.reset(goal_state=np.array([25,25]), init_state=current_init[init_idx])
            origin = current_init[init_idx]

    if actor1:
        return time_step1, train_env1, time_step_no_goal, train_env_no_goal, origin
    else:
        return time_step, train_env, None, None, origin
    
    
def sample_goal(cfg, proto_goals, proto_goals_state, unreached_goals, eval_env_no_goal):

    s = proto_goals.shape[0]
    num = s+unreached_goals.shape[0]
    idx = np.random.randint(num)

    if idx >= s:

        goal_idx = idx-s
        goal_state = unreached_goals[goal_idx]
        with eval_env_no_goal.physics.reset_context():
            eval_env_no_goal.physics.set_state(goal_state)
        goal_pix = eval_env_no_goal._env.physics.render(height=84, width=84, camera_id=cfg.camera_id)
        goal_pix = np.transpose(goal_pix, (2,0,1))
        goal_pix = np.tile(goal_pix, (cfg.frame_stack,1,1))
        unreached=True


    else:
        goal_idx = idx
        goal_state = proto_goals_state[goal_idx]
        goal_pix = proto_goals[goal_idx]
        unreached=False

    return goal_idx, goal_state, goal_pix, unreached


def calc_reward(cfg, agent, pix, goal_pix, goal_state, global_step, device, train_env1):
    #NOT USING RN

    if cfg.ot_reward:

        reward = agent.ot_rewarder(pix, goal_pix, global_step)

#                     elif _env.cfg.dac_reward:

#                         reward = _env.agent.dac_rewarder(time_step1.observation['pixels'], action1)

    elif cfg.neg_euclid:

        with torch.no_grad():
            obs = pix
            obs = torch.as_tensor(obs.copy(), device=device).unsqueeze(0)
            z1 = agent.encoder(obs)
            z1 = agent.predictor(z1)
            z1 = agent.projector(z1)
            z1 = F.normalize(z1, dim=1, p=2)

            goal = torch.as_tensor(goal_pix, device=device).unsqueeze(0).int()

            z2 = agent.encoder(goal)
            z2 = agent.predictor(z2)
            z2 = agent.projector(z2)
            z2 = F.normalize(z2, dim=1, p=2)

        reward = -torch.norm(z1-z2, dim=-1, p=2).item()

    elif cfg.neg_euclid_state:

        reward = -np.linalg.norm(train_env1.physics.get_state() - goal_state, axis=-1, ord=2)

    elif cfg.actionable:

        print('not implemented yet: actionable_reward')
    return reward


def goal_reached_save_stats(cfg, proto_goals, proto_goals_state, proto_goals_dist, current_init, goal_state, goal_idx, origin, 
        reached_goals, proto_goals_matrix, pmm, train_env1, unreached, unreached_goals=None):
    idx_x = int(goal_state[0] * 100) + 29
    idx_y = int(goal_state[1] * 100) + 29
    origin_x = int(origin[0] * 100) + 29
    origin_y = int(origin[1] * 100) + 29
    reached_goals[0, origin_x, idx_x] = 1
    reached_goals[1, origin_y, idx_y] = 1
    proto_goals_matrix[idx_x, idx_y] += 1

    ##############################
    # add non-pmm later
    if pmm:
        unreached_goals = np.round(unreached_goals, 2)
        print('u', unreached_goals)
        print('g', goal_state)

        if np.round(goal_state, 2) in unreached_goals:
            index = np.where((unreached_goals == np.round(goal_state, 2)).all(axis=1))
            unreached_goals = np.delete(unreached_goals, index, axis=0)
            print('removed goal from unreached', np.round(goal_state, 2))
            print('unreached', unreached_goals)

        if unreached == False:
            proto_goals = np.delete(proto_goals, goal_idx, axis=0)
            proto_goals_state = np.delete(proto_goals_state, goal_idx, axis=0)
            proto_goals_dist = np.delete(proto_goals_dist, goal_idx, axis=0)
        if cfg.gc_only is False:
            assert proto_goals.shape[0] == proto_goals_state.shape[0] == \
                proto_goals_dist.shape[0]
        elif cfg.gc_only and cfg.resume_training is False:
            assert proto_goals_state.shape[0] == proto_goals.shape[0]
    else:
        print('not implemented yet!!!!')

    current_init = np.append(current_init, train_env1.physics.get_state()[None, :],
                                  axis=0)
    if pmm:
        assert len(current_init.shape) == 2
    return reached_goals, proto_goals_matrix, unreached_goals, proto_goals, proto_goals_state, proto_goals_dist, current_init


def get_time_step(cfg, proto_last_explore, gc_only, current_init, actor, actor1, pmm, proto_goals=None, proto_goals_state=None, proto_goals_dist=None, 
unreached_goals=None, eval_env_no_goal=None, train_env=None, train_env1=None, train_env_no_goal=None):

    if proto_last_explore > cfg.proto_explore_episodes and gc_only is False:
        actor = True
        actor1 = False
        proto_last_explore = 0
        print('proto last >100')

    if actor and gc_only is False:
        assert actor1 is False
        if current_init.shape[0] != 0:

            if current_init.shape[0] > 2:

                chance = np.random.uniform()
                if chance < .8:
                    init_idx = -np.random.randint(1, 4)
                else:
                    init_idx = np.random.randint(current_init.shape[0])

            elif len(current_init) > 0:
                init_idx = -1

            time_step = train_env.reset(init_state=current_init[init_idx], goal_state=np.array([25, 25]))
            print('init_state', current_init[init_idx])

        else:
            if pmm:
                init_state = np.random.uniform(.25,.29,size=(train_env.physics.get_state().shape[0],))
                init_state[2] = 0
                init_state[3] = 0
                init_state[0] = init_state[0]*(-1)
            time_step = train_env.reset(init_state=init_state, goal_state=np.array([25, 25]))
            print('init', init_state)

        print('proto_explore', time_step.observation['observations'])

        goal_idx = None
        goal_state = None
        goal_pix = None
        time_step_no_goal = None
        return time_step, train_env, time_step_no_goal, goal_idx, goal_state, goal_pix, actor, actor1, proto_last_explore

    else:
        assert actor1 is True
        if cfg.gc_only is False:
            assert proto_goals.shape[0] == proto_goals_state.shape[0] == \
                   proto_goals_dist.shape[0]
        elif cfg.gc_only and cfg.resume_training is False:
            assert proto_goals_state.shape[0] == proto_goals.shape[0]

        goal_idx, goal_state, goal_pix, unreached = sample_goal(cfg, proto_goals, proto_goals_state, unreached_goals, eval_env_no_goal)

        if len(current_init) != 0:
            if current_init.shape[0] > 2:
                chance = np.random.uniform()
                if chance < .8:
                    init_idx = -np.random.randint(1, 4)
                else:
                    init_idx = np.random.randint(current_init.shape[0])
            elif len(current_init) > 0:
                init_idx = -1
            
            time_step1 = train_env1.reset(init_state=current_init[init_idx], goal_state=goal_state[:2])
            print('ts1', time_step1.observation['observations'])
            
            time_step_no_goal = train_env_no_goal.reset(init_state=train_env1.physics.get_state(), goal_state=np.array([25, 25]))
            print('ts no goal', time_step_no_goal.observation['observations'])
            
        else:
            if pmm:
                init_state = np.random.uniform(.25,.29,size=(train_env1.physics.get_state().shape[0],))
                init_state[2] = 0
                init_state[3] = 0
                init_state[0] = init_state[0]*(-1)

            time_step1 = train_env1.reset(init_state=init_state, goal_state=goal_state[:2])
            print('init', init_state)
            
            time_step_no_goal = train_env_no_goal.reset(init_state=train_env1.physics.get_state(), goal_state=np.array([25, 25]))
            print('ts no goal 2', time_step_no_goal.observation['observations']) 

        print('time step', time_step1.observation['observations'])
        print('sampled goal', goal_state)

        # unreached_goals : array of states (not pixel), need to set state & render to get pixel

        return time_step1, train_env1, time_step_no_goal, train_env_no_goal, goal_idx, goal_state, goal_pix, actor, actor1, proto_last_explore, unreached
