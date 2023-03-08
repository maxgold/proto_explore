import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
from logger import save
import torch
import seaborn as sns;
sns.set_theme()
from replay_buffer import make_replay_offline
from pathlib import Path
import torch.nn.functional as F
from agent_utils import *
import wandb
import dmc
from scipy.spatial.distance import cdist

def ndim_grid(ndims, space):
    L = [np.linspace(-.29,.29,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

def eval_proto_gc_only(cfg, agent, device, pwd, global_step, pmm, train_env, proto_goals, proto_goals_state, proto_goals_dist, dim, work_dir, 
current_init, state_visitation_gc, reward_matrix_gc, goal_state_matrix, state_visitation_proto, proto_goals_matrix, mov_avg_5, mov_avg_10, 
mov_avg_20, mov_avg_50, r_mov_avg_5, r_mov_avg_10, r_mov_avg_20, r_mov_avg_50, eval=False):
    print('eval_proto_gc_only')
    if global_step % 100000 == 0 and global_step!=0 and pmm:
        heatmaps(state_visitation_gc, reward_matrix_gc, goal_state_matrix, state_visitation_proto, proto_goals_matrix, global_step, gc=True, proto=False)
    
    #TODO: Add goal selection from pretrain_pixel_gc_only.py later (sample_goal_distance & under self.cfg.curriculu)

    while proto_goals.shape[0] < dim:
        proto_goals = np.append(proto_goals, np.zeros((1, 3 * cfg.frame_stack, 84, 84)), axis=0)
        proto_goals_state = np.append(proto_goals_state, np.array([[0., 0., 0., 0.]]), axis=0)
        proto_goals_dist = np.append(proto_goals_dist, np.array([[0.]]), axis=0)
    
    if len(current_init) != 0:
        reached = current_init
        current_init = np.random.uniform(.25, .29, size=(current_init.shape[0], 2))
        current_init[:,0] = -1 * current_init[:,0] 
    else:
        reached = np.random.uniform(.25, .29, size=(1, 2))
        reached[0] = -1 * reached[0]
    
    goal_array = ndim_grid(2,10)
    goal_array_ = []
    for x in reached:

        dist_goal = cdist(np.array([[x[0],x[1]]]), goal_array, 'euclidean')

        df1=pd.DataFrame()
        df1['distance'] = dist_goal.reshape((goal_array.shape[0],))
        df1['index'] = df1.index
        df1 = df1.sort_values(by='distance')

        for ix in range(5):
            goal_array_.append(goal_array[df1.iloc[ix,1]])
    
    random = np.random.uniform(.25, .29, size=(2,))
    random[0] = -1 * random[0]
    goal_array_.append(random)
    
    tmp_goal = np.array(goal_array_[-proto_goals_state.shape[0]:])
    proto_goals_state[:tmp_goal.shape[0]] = np.concatenate((tmp_goal, np.zeros((tmp_goal.shape[0],2))), axis=-1)

    eval_env_no_goal = dmc.make(cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                                 cfg.action_repeat, seed=None, goal=None,
                                                 init_state=None)
    for ix in range(proto_goals_state.shape[0]):
        with eval_env_no_goal.physics.reset_context():
            eval_env_no_goal.physics.set_state(proto_goals_state[ix])

        img = eval_env_no_goal._env.physics.render(height=84, width=84,
                                                        camera_id=cfg.camera_id)
        img = np.transpose(img, (2, 0, 1))
        img = np.tile(img, (cfg.frame_stack, 1, 1))
        proto_goals[ix] = img

    index = np.where((proto_goals_state == 0.).all(axis=1))[0]
    proto_goals = np.delete(proto_goals, index, axis=0)
    proto_goals_state = np.delete(proto_goals_state, index, axis=0)
    proto_goals_dist = np.delete(proto_goals_dist, index, axis=0)
    return current_init, proto_goals, proto_goals_state, proto_goals_dist

def eval_proto(cfg, agent, device, pwd, global_step, pmm, train_env, proto_goals, proto_goals_state, proto_goals_dist, dim, work_dir, 
current_init, state_visitation_gc, reward_matrix_gc, goal_state_matrix, state_visitation_proto, proto_goals_matrix, mov_avg_5, mov_avg_10, 
mov_avg_20, mov_avg_50, r_mov_avg_5, r_mov_avg_10, r_mov_avg_20, r_mov_avg_50, eval=False):

    if eval:
        # used for continue training 
        # finds the current init from buffer loaded in (reached goals of previous training)
        path = cfg.model_path.split('/')
        path = Path(pwd + '/'.join(path[:-1]))
        replay_buffer = make_replay_offline(
                                            path / 'buffer1' / 'buffer_copy',
                                            500000,
                                            0,
                                            0,
                                            .99,
                                            goal=False,
                                            relabel=False,
                                            replay_dir2=False,
                                            obs_type='pixels'
                                            )

        state, actions, rewards, goal_states, eps, index = replay_buffer.parse_dataset(goal_state=True)

        df = pd.DataFrame(
            {'s0': state[:, 0], 's1': state[:, 1], 'r': rewards[:, 0], 'g0': goal_states[:, 0], 'g1': goal_states[:, 1],
             'e': eps})
        df['eps'] = [x.split('/')[-1] for x in df['e']]
        df1 = pd.DataFrame()
        df1['g0'] = df.groupby('eps')['g0'].first()
        df1['g1'] = df.groupby('eps')['g1'].first()
        df1['r'] = df.groupby('eps')['r'].sum()
        df1 = df1[df1['r'] > 100]
        df1 = df1.reset_index(drop=True)

        current_init = df1[['g0', 'g1']].to_numpy()
        current_init = current_init[sorted(np.unique(current_init, return_index=True, axis=0)[1])]
        current_init = np.concatenate((current_init, np.zeros((current_init.shape[0], 2))), axis=1)


    else:
        if global_step % 100000 == 0 and global_step!=0 and pmm:
            # TODO
            # CHANGE
            heatmaps(state_visitation_gc, reward_matrix_gc, goal_state_matrix, state_visitation_proto, proto_goals_matrix, global_step, gc=True, proto=True)

        ########################################################################
        # how should we measure exploration in non-pmm w/o heatmaps

        ############################################################
        # offline loader is not adding in the replay_dir2 right now, change this !!!!!!!!!!!!!
        #path = Path(
        #    '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2023.02.01/231849_proto_inv')
        replay_buffer = make_replay_offline(
                                            work_dir / 'buffer2' / 'buffer_copy',
                                            500000,
                                            0,
                                            0,
                                            .99,
                                            goal=False,
                                            relabel=False,
                                            replay_dir2=False,
                                            obs_type='pixels'
                                            )

        state, actions, rewards, eps, index = replay_buffer.parse_dataset()
        state = state.reshape((state.shape[0], train_env.physics.get_state().shape[0]))

    while proto_goals.shape[0] < dim:
        proto_goals = np.append(proto_goals, np.zeros((1, 3 * cfg.frame_stack, 84, 84)), axis=0)
        proto_goals_state = np.append(proto_goals_state, np.array([[0., 0., 0., 0.]]), axis=0)
        proto_goals_dist = np.append(proto_goals_dist, np.array([[0.]]), axis=0)

    protos = agent.protos.weight.data.detach().clone()

    num_sample = 600
    idx = np.random.randint(0, state.shape[0], size=num_sample)
    state = state[idx]
    state = state.reshape(num_sample, train_env.physics.get_state().shape[0])
    a = state

    encoded = []
    proto = []
    actual_proto = []
    lst_proto = []

    for x in idx:
        fn = eps[x]
        idx_ = index[x]
        ep = np.load(fn)
        # pixels.append(ep['observation'][idx_])

        with torch.no_grad():
            obs = ep['observation'][idx_]
            # import IPython as ipy; ipy.embed(colors='neutral')
            ##################
            # why is there an extra dim here
            if obs.shape[0] != 1:
                obs = torch.as_tensor(obs.copy(), device=device).unsqueeze(0)
            else:
                obs = torch.as_tensor(obs.copy(), device=device)
            if cfg.sl == False:
                z = agent.encoder(obs)
            else:

                z = agent.encoder2(obs)
            encoded.append(z)
            z = agent.predictor(z)
            z = agent.projector(z)
            if cfg.normalize:
                z = F.normalize(z, dim=1, p=2)
            proto.append(z)
            sim = agent.protos(z)
            idx_ = sim.argmax()
            actual_proto.append(protos[idx_][None, :])

    encoded = torch.cat(encoded, axis=0)
    proto = torch.cat(proto, axis=0)
    actual_proto = torch.cat(actual_proto, axis=0)
    sample_dist = torch.norm(proto[:, None, :] - proto[None, :, :], dim=2, p=2)
    proto_dist = torch.norm(protos[:, None, :] - proto[None, :, :], dim=2, p=2)
    all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)
    p = _proto.clone().detach().cpu().numpy()

    eval_env_no_goal = dmc.make(cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                                 cfg.action_repeat, seed=None, goal=None,
                                                 init_state=None)
    if cfg.proto_goal_intr:

        goal_dist, goal_indices = eval_intrinsic(encoded, a)
        dist_arg = proto_goals_dist.argsort(axis=0)

        for ix, x in enumerate(goal_dist.clone().detach().cpu().numpy()):

            if x > proto_goals_dist[dist_arg[ix]]:
                proto_goals_dist[dist_arg[ix]] = x
                closest_sample = goal_indices[ix].clone().detach().cpu().numpy()

                ##################################
                # may need to debug this
                #                     fn = eps[closest_sample]
                #                     idx_ = index[closest_sample]
                #                     ep = np.load(fn)

                #                     with torch.no_grad():
                #                         obs = ep['observation'][idx_]

                #                     _env.proto_goals[dist_arg[ix]] = obs

                proto_goals_state[dist_arg[ix]] = a[closest_sample]

                with eval_env_no_goal.physics.reset_context():
                    eval_env_no_goal.physics.set_state(a[closest_sample][0])

                # img = eval_env_no_goal._env.physics.render(height=84, width=84,
                #                                                 camera_id=dict(quadruped=2).get(cfg.domain, 0))
                img = eval_env_no_goal._env.physics.render(height=84, width=84,
                                                                camera_id=cfg.camera_id)
                img = np.transpose(img, (2, 0, 1))
                img = np.tile(img, (cfg.frame_stack, 1, 1))
                proto_goals[dist_arg[ix]] = img

    elif cfg.proto_goal_random:

        closest_sample = _proto[:, 0].detach().clone().cpu().numpy()
        proto_goals_state = a[closest_sample]

        
        for ix in range(proto_goals_state.shape[0]):
            with eval_env_no_goal.physics.reset_context():
                eval_env_no_goal.physics.set_state(proto_goals_state[ix])

            img = eval_env_no_goal._env.physics.render(height=84, width=84,
                                                            camera_id=cfg.camera_id)
            img = np.transpose(img, (2, 0, 1))
            img = np.tile(img, (cfg.frame_stack, 1, 1))
            proto_goals[ix] = img

    if pmm:

        filenames = []
        plt.clf()
        fig, ax = plt.subplots()
        dist_np = np.empty((protos.shape[1], _proto.shape[1], 2))
        for ix in range(protos.shape[0]):
            txt = ''
            df = pd.DataFrame()
            count = 0
            for i in range(a.shape[0] + 1):
                if i != a.shape[0]:
                    df.loc[i, 'x'] = a[i, 0]
                    df.loc[i, 'y'] = a[i, 1]
                    if i in _proto[ix, :]:
                        df.loc[i, 'c'] = str(ix + 1)
                        dist_np[ix, count, 0] = a[i, 0]
                        dist_np[ix, count, 1] = a[i, 1]
                        count += 1

                    elif ix == 0 and (i not in _proto[ix, :]):
                        # color all samples blue
                        df.loc[i, 'c'] = str(0)

            palette = {
                '0': 'tab:blue',
                '1': 'tab:orange',
                '2': 'black',
                '3': 'silver',
                '4': 'green',
                '5': 'red',
                '6': 'purple',
                '7': 'brown',
                '8': 'pink',
                '9': 'gray',
                '10': 'olive',
                '11': 'cyan',
                '12': 'yellow',
                '13': 'skyblue',
                '14': 'magenta',
                '15': 'lightgreen',
                '16': 'blue',
                '17': 'lightcoral',
                '18': 'maroon',
                '19': 'saddlebrown',
                '20': 'peru',
                '21': 'tan',
                '22': 'darkkhaki',
                '23': 'darkolivegreen',
                '24': 'mediumaquamarine',
                '25': 'lightseagreen',
                '26': 'paleturquoise',
                '27': 'cadetblue',
                '28': 'steelblue',
                '29': 'thistle',
                '30': 'slateblue',
                '31': 'hotpink',
                '32': 'papayawhip'
            }
            ax = sns.scatterplot(x="x", y="y",
                                 hue="c", palette=palette,
                                 data=df, legend=True)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            # ax.set_title("\n".join(wrap(txt,75)))

            file1 = work_dir / f"10nn_actual_prototypes_{global_step}.png"
            plt.savefig(file1)
            wandb.save(f"10nn_actual_prototypes_{global_step}.png")

    ########################################################################
    # implement tsne for non-pmm prototype eval?
    if global_step % 100000 == 0 and pmm == False:
        eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, camera_id=cfg.camera_id)
        for ix in range(proto_goals_state.shape[0]):
            with eval_env.physics.reset_context():
                eval_env.physics.set_state(proto_goals_state[ix])

            plt.clf()
            img = eval_env._env.physics.render(height=84, width=84,
                                                    camera_id=cfg.camera_id)
            plt.imsave(f"goals_{ix}_{global_step}.png", img)
            wandb.save(f"goals_{ix}_{global_step}.png")

    # delete goals that have been reached
    if current_init.shape[0] > 0:
        index = np.where(((np.linalg.norm(proto_goals_state[:, None, :] - current_init[None, :, :], axis=-1,
                                          ord=2)) < .05))
        index = np.unique(index[0])
        print('delete goals', proto_goals_state[index])
        proto_goals = np.delete(proto_goals, index, axis=0)
        proto_goals_state = np.delete(proto_goals_state, index, axis=0)
        proto_goals_dist = np.delete(proto_goals_dist, index, axis=0)
        index = np.where((proto_goals_state == 0.).all(axis=1))[0]
        proto_goals = np.delete(proto_goals, index, axis=0)
        proto_goals_state = np.delete(proto_goals_state, index, axis=0)
        proto_goals_dist = np.delete(proto_goals_dist, index, axis=0)
        print('current goals', proto_goals_state)
        if cfg.gc_only is False:
            assert proto_goals_state.shape[0] == proto_goals.shape[0] == proto_goals_dist.shape[0]
        elif cfg.gc_only and cfg.resume_training is False:
            assert proto_goals_state.shape[0] == proto_goals.shape[0]
    else:
        print('current goals', proto_goals_state)

    return current_init, proto_goals, proto_goals_state, proto_goals_dist


def eval_pmm(cfg, agent, eval_reached, video_recorder, global_step, global_frame, work_dir, rand_init):
    df = pd.DataFrame(columns=['x', 'y', 'r'], dtype=np.float64)
    print('eval reached', eval_reached)
    for i, init in enumerate(rand_init):
        for idx in range(eval_reached.shape[0]):
            goal_array = ndim_grid(2, 20)
            print('init', init.shape)
            print('goal array', goal_array.shape)
            dist= torch.norm(torch.tensor(init[None,:]) - torch.tensor(goal_array), dim=-1, p=2)
            goal_dist, _ = torch.topk(dist, 50, dim=-1, largest=False)
            goal_array = goal_array[_]
            
            #TODO: delete this part when done debugging
            # lst=[]
            # for ix,x in enumerate(goal_array):
            #    if (-.02<x[0]  or  x[1]<.02):
            #        lst.append(ix)
            # goal_array=np.delete(goal_array, lst,0)

            # init = eval_reached[idx]
            print('goal array', goal_array)

            for ix, x in enumerate(goal_array):

                step, episode, total_reward = 0, 0, 0
                min_dist = 1
                max_action = 0

                goal_state = x
                print('goal state', goal_state)
                eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                        cfg.action_repeat, seed=None, goal=goal_state, init_state=init, camera_id=cfg.camera_id)
                eval_env_goal = dmc.make(cfg.task_no_goal, 'states', cfg.frame_stack,
                                            cfg.action_repeat, seed=None, goal=None, camera_id=cfg.camera_id)
                eval_until_episode = utils.Until(cfg.num_eval_episodes)
                meta = agent.init_meta()

                while eval_until_episode(episode):
                    
                    time_step = eval_env.reset()

                    if cfg.obs_type == 'pixels':
                        eval_env_no_goal = dmc.make(cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                                        cfg.action_repeat, seed=None, goal=None,
                                                        init_state=time_step.observation['observations'][:2], camera_id=cfg.camera_id)
                        time_step_no_goal = eval_env_no_goal.reset()

                        with eval_env_goal.physics.reset_context():
                            time_step_goal = eval_env_goal.physics.set_state(
                                np.array([goal_state[0], goal_state[1], 0, 0]))
                        time_step_goal = eval_env_goal._env.physics.render(height=84, width=84,
                                                                                camera_id=cfg.camera_id)
                        time_step_goal = np.transpose(time_step_goal, (2, 0, 1))
                    video_recorder.init(eval_env, enabled=(episode == 0))
                    
                    while step != cfg.episode_length:
                        with torch.no_grad(), utils.eval_mode(agent):
                            if cfg.obs_type == 'pixels':
                                action = agent.act(time_step_no_goal.observation['pixels'],
                                                        time_step_goal,
                                                        meta,
                                                        global_step,
                                                        eval_mode=True,
                                                        tile=cfg.frame_stack,
                                                        general=True)
                            else:
                                action = agent.act(time_step.observation,
                                                        x, 
                                                        meta,
                                                        global_step,
                                                        eval_mode=True,
                                                        tile=1)

                        if cfg.velocity_control:
                            vel = action.copy()
                            action = np.zeros(2)
                            eval_env.physics.data.qvel[0] = vel[0]
                            eval_env.physics.data.qvel[1] = vel[1]

                        time_step = eval_env.step(action)
                        max_action = np.maximum(abs(action).sum(), max_action)
                        # print('abs', abs(action).sum())
                        # print('max', max_action)
                        if cfg.obs_type == 'pixels':
                            time_step_no_goal = eval_env_no_goal.step(action)

                        video_recorder.record(eval_env)
                        total_reward += time_step.reward
                        step += 1
                        dist = np.linalg.norm(time_step.observation['observations'][:2] - goal_state)
                        min_dist = np.minimum(dist, min_dist)
                    episode += 1

                    if ix % 10 == 0:
                        video_recorder.save(f'{global_frame}_{ix}_{i}.mp4')

                    # save(str(work_dir) + '/eval_{}.csv'.format(global_step),
                    #      [[goal_state, total_reward, time_step.observation['observations'], step]])

                    # if total_reward > 10 * cfg.num_eval_episodes:
                    #     eval_reached = np.append(eval_reached, x, axis=0)
                    #     eval_reached = np.unique(eval_reached, axis=0)

                df.loc[ix, 'x'] = x[0].round(2)
                df.loc[ix, 'y'] = x[1].round(2)
                df.loc[ix, 'r'] = total_reward
                df.loc[ix, 'a'] = max_action
                df.loc[ix, 'd'] = min_dist
                print('r', total_reward)

        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
        result.fillna(0, inplace=True)
        print('result', result)
        plt.clf()
        fig, ax = plt.subplots()
        plt.title(str(init))
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
        plt.savefig(f"./{global_step}_{i}_heatmap_goal.png")
        wandb.save(f"./{global_step}_{i}_heatmap_goal.png")

        #action

        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['a']/2
        result.fillna(0, inplace=True)
        print('result', result)
        plt.clf()
        fig, ax = plt.subplots()
        plt.title(str(init))
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
        plt.savefig(f"./{global_step}_{i}_heatmap_max_action.png")
        wandb.save(f"./{global_step}_{i}_heatmap_max_action.png")

        #min dist to goal
        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['d']/2
        result.fillna(0, inplace=True)
        print('result', result)
        plt.clf()
        fig, ax = plt.subplots()
        plt.title(str(init))
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
        plt.savefig(f"./{global_step}_{i}_heatmap_dist.png")
        wandb.save(f"./{global_step}_{i}_heatmap_dist.png")



def eval(cfg, agent, proto_goals, video_recorder, pmm, global_step, global_frame, global_episode, logger):

    for goal in proto_goals:
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(cfg.num_eval_episodes)
        meta = agent.init_meta()
        eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, camera_id=cfg.camera_id)

        while eval_until_episode(episode):
            time_step = eval_env.reset()
            video_recorder.init(eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    if cfg.obs_type == 'pixels' and pmm:
                        action = agent.act(time_step.observation['pixels'].copy(),
                                                goal,
                                                meta,
                                                global_step,
                                                eval_mode=True,
                                                tile=1,
                                                general=True)
                        #non-pmm
                    elif cfg.obs_type == 'pixels':
                        action = agent.act(time_step.observation['pixels'].copy(),
                                                goal,
                                                meta,
                                                global_step,
                                                eval_mode=True,
                                                tile=1,
                                                general=True) 
                    else:
                        action = agent.act(time_step.observation,
                                        meta,
                                        global_step,
                                        eval_mode=True)

                if cfg.velocity_control:
                    vel = action.copy()
                    action = np.zeros(2)
                    eval_env.physics.data.qvel[0] = vel[0]
                    eval_env.physics.data.qvel[1] = vel[1]

                time_step = eval_env.step(action)
                video_recorder.record(eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            video_recorder.save(f'{global_frame}.mp4')

        with logger.log_and_dump_ctx(global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * cfg.action_repeat / episode)
            log('episode', global_episode)
            log('step', global_step)


def eval_intrinsic(cfg, agent, encoded, states, global_step):

    with torch.no_grad():
        reward = agent.compute_intr_reward(encoded, None, global_step, eval=True)

    if cfg.proto_goal_intr:
        #import IPython as ipy; ipy.embed(colors='neutral') 
        r, _ = torch.topk(reward,5,largest=True, dim=0)

    df = pd.DataFrame()
    df['x'] = states[:,0].round(2)
    df['y'] = states[:,1].round(2)
    df['r'] = reward.detach().clone().cpu().numpy()
    result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r'].round(2)
    #import IPython as ipy; ipy.embed(colors='neutral')
    result.fillna(0, inplace=True)
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,6))

    sns.heatmap(result, cmap="Blues_r",fmt='.2f', ax=ax).invert_yaxis()
    ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
    ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
    ax.set_title(global_step)
    plt.savefig(f"./{global_step}_intr_reward.png")
    wandb.save(f"./{global_step}_intr_reward.png")

    if cfg.proto_goal_intr:
        return r, _
