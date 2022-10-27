from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
import pandas as pd
import utils
from agent.ddpg import DDPGAgent



class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(pred_dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, pred_dim))

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.trunk(x)


class ProtoV2Agent(DDPGAgent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder, update_gc, offline, gc_only, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.queue_size=queue_size
        self.encoder_target_tau = encoder_target_tau
        #self.protos_target_tau = protos_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder
        self.update_gc = update_gc
        self.offline = offline
        self.gc_only = gc_only
        self.batch_size = 256
        print('bathch', self.batch_size)
        self.label_bank = torch.zeros((self.batch_size,))
        self.protos = torch.empty((self.num_protos, pred_dim))
        self.initialized = False
        self.rank=0
        self.min_cluster=1
        self.debug=False
        #self.load_protos = load_protos

        # models

        self.encoder_target = deepcopy(self.encoder)

        self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
        self.predictor.apply(utils.weight_init)
        self.predictor_target = deepcopy(self.predictor)

        self.projector = Projector(pred_dim, proj_dim).to(self.device)
        self.projector.apply(utils.weight_init)

        #self.protos_target = deepcopy(self.protos)
        # candidate queue
        self.queue = torch.zeros(queue_size, pred_dim, device=self.device)
        self.queue_ptr = 0

        #sample memory queue 
        self.feature_queue = torch.zeros(queue_size, pred_dim, device=self.device)
        self.feature_queue_ptr = 0

        # optimizers
        self.proto_opt = torch.optim.Adam(utils.chain(
            self.encoder.parameters(), self.predictor.parameters(),
            self.projector.parameters()), lr=self.lr)

        self.predictor.train()
        self.projector.train()


    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        if self.init_critic:
            utils.hard_update_params(other.critic, self.critic)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)

    def init_gc_from(self,critic, actor):
        utils.hard_update_params(critic, self.critic1)
        utils.hard_update_params(actor, self.actor1)
    
    def init_protos_from(self, protos):
        utils.hard_update_params(protos.predictor, self.predictor)
        utils.hard_update_params(protos.projector, self.projector)
        utils.hard_update_params(protos.encoder, self.encoder)
    
    def init_memory(self, feature):
        """Initialize memory modules."""
        self.initialized = True
        label = np.zeros((feature.shape[0],))
        self.label_bank.copy_(torch.from_numpy(label).long())
#         # make sure no empty clusters
#         assert (np.bincount(label, minlength=self.num_protos) != 0).all()
        if self.rank == 0:
            ########
            #what is this for?
            feature /= (np.linalg.norm(feature, axis=1).reshape(-1, 1) + 1e-10)

            protos = self._compute_protos()
            self.protos.copy_(protos)
    
    def _compute_protos_ind(self, cinds):
        """Compute a few centroids. e.g. after reassigning"""
        
        
        assert self.rank == 0
        num = len(cinds)
        protos = torch.zeros((num, self.feat_dim), dtype=torch.float32)
        
        for i, c in enumerate(cinds):
            ind = np.where(self.label_bank.numpy() == c)[0]
            protos[i, :] = self.feature_queue[ind, :].mean(dim=0)
        
        return protos
    
    def _compute_protos(self):
        """Compute all non-empty centroids."""
        assert self.rank == 0
        label_bank_np = self.label_bank.numpy()
        argl = np.argsort(label_bank_np)
        sortl = label_bank_np[argl]
        diff_pos = np.where(sortl[1:] - sortl[:-1] != 0)[0] + 1
        start = np.insert(diff_pos, 0, 0)
        end = np.insert(diff_pos, len(diff_pos), len(label_bank_np))
        class_start = sortl[start]
        # keep empty class centroids unchanged
        protos = self.protos.cpu().clone()
        for i, st, ed in zip(class_start, start, end):
            protos[i, :] = self.feature_queue[argl[st:ed], :].mean(dim=0)
        return protos
    
    def deal_with_small_clusters(self):
        """Deal with small clusters."""
        # check empty class
        histogram = np.bincount(
            self.label_bank.numpy(), minlength=self.num_protos)
        small_clusters = np.where(histogram < self.min_cluster)[0].tolist()
        if self.debug and self.rank == 0:
            print(f'mincluster: {histogram.min()}, '
                  f'num of small class: {len(small_clusters)}')
        if len(small_clusters) == 0:
            return
        
        # re-assign samples in small clusters to make them empty
        for s in small_clusters:
            ind = np.where(self.label_bank.numpy() == s)[0]
            if len(ind) > 0:
                inclusion = torch.from_numpy(
                    np.setdiff1d(
                        np.arange(self.num_protos),
                        np.array(small_clusters),
                        assume_unique=True)).cuda()
                
                if self.rank == 0:
                    #permute?
                    target_ind = torch.mm(
                        self.protos[inclusion, :],
                        self.feature_queue[ind, :].cuda().permute(
                            1, 0)).argmax(dim=0)
                    target = inclusion[target_ind]
                else:
                    target = torch.zeros((ind.shape[0], ),
                                         dtype=torch.int64).cuda()

                self.label_bank[ind] = torch.from_numpy(target.cpu().numpy())
        # deal with empty cluster
        self._redirect_empty_clusters(small_clusters)
        
    def update_protos_memory(self, cinds=None):
        """Update centroids memory."""
        if self.rank == 0:
            if self.debug:
                print('updating centroids ...')
            if cinds is None:
                center = self._compute_protos()
                self.protos.copy_(center)
            else:
                center = self._compute_protos_ind(cinds)
                self.protos[
                    torch.LongTensor(cinds).cuda(), :] = center.cuda()

        
    def _partition_max_cluster(self, max_cluster):
        """Partition the largest cluster into two sub-clusters."""
        assert self.rank == 0
        max_cluster_inds = np.where(self.label_bank == max_cluster)[0]

        assert len(max_cluster_inds) >= 2
        max_cluster_features = self.feature_queue[max_cluster_inds, :]
        if np.any(np.isnan(max_cluster_features.numpy())):
            raise Exception('Has nan in features.')
        kmeans_ret = self.kmeans.fit(max_cluster_features)
        #look at how its split 
        sub_cluster1_ind = max_cluster_inds[kmeans_ret.labels_ == 0]
        sub_cluster2_ind = max_cluster_inds[kmeans_ret.labels_ == 1]
        if not (len(sub_cluster1_ind) > 0 and len(sub_cluster2_ind) > 0):
            print(
                'Warning: kmeans partition fails, resort to random partition.')
            ##debug
            p_to_f = torch.norm(self.protos[max_cluster, None, :] - self.feature_queue[None, max_cluster_inds, :], dim=2, p=2)
            all_dists, _ = torch.topk(p_to_f, self.topk, dim=1, largest=True)
            
            sub_cluster1_ind = _[0].detach().cpu().numpy()
            sub_cluster2_ind = np.setdiff1d(
                max_cluster_inds, sub_cluster1_ind, assume_unique=True)
        return sub_cluster1_ind, sub_cluster2_ind
    
    def _redirect_empty_clusters(self, empty_clusters):
        """Re-direct empty clusters."""
        for e in empty_clusters:
            assert (self.label_bank != e).all().item(), \
                f'Cluster #{e} is not an empty cluster.'
            max_cluster = np.bincount(
                self.label_bank, minlength=self.num_protos).argmax().item()
            
            # gather partitioning indices
            if self.rank == 0:
                sub_cluster1_ind, sub_cluster2_ind = \
                    self._partition_max_cluster(max_cluster)
                size1 = torch.LongTensor([len(sub_cluster1_ind)]).cuda()
                size2 = torch.LongTensor([len(sub_cluster2_ind)]).cuda()
                sub_cluster1_ind_tensor = torch.from_numpy(
                    sub_cluster1_ind).long().cuda()
                sub_cluster2_ind_tensor = torch.from_numpy(
                    sub_cluster2_ind).long().cuda()
            else:
                size1 = torch.LongTensor([0]).cuda()
                size2 = torch.LongTensor([0]).cuda()

            if self.rank != 0:
                sub_cluster1_ind_tensor = torch.zeros(
                    (size1, ), dtype=torch.int64).cuda()
                sub_cluster2_ind_tensor = torch.zeros(
                    (size2, ), dtype=torch.int64).cuda()

            if self.rank != 0:
                sub_cluster1_ind = sub_cluster1_ind_tensor.cpu().numpy()
                sub_cluster2_ind = sub_cluster2_ind_tensor.cpu().numpy()

            # reassign samples in partition #2 to the empty class
            self.label_bank[sub_cluster2_ind] = e
            # update centroids of max_cluster and e
            self.update_protos_memory([max_cluster, e])
        
    def normalize_protos(self):
        print(self.protos)
        C = F.normalize(self.protos, dim=1, p=2)
        self.protos.copy_(C)

    def compute_intr_reward(self, obs, step):
        self.normalize_protos()
        # find a candidate for each prototype
        with torch.no_grad():
            z = self.encoder(obs)
            z = self.predictor(z)
            z = F.normalize(z, dim=1, p=2)
            import IPython as ipy; ipy.embed(colors='neutral')
            scores = torch.mm(z, self.protos)
            prob = F.softmax(scores, dim=1)
            candidates = pyd.Categorical(prob).sample()

        #if step>300000:
        #    import IPython as ipy; ipy.embed(colors='neutral')
        # enqueue candidates
        ptr = self.queue_ptr
        self.queue[ptr:ptr + self.num_protos] = z[candidates]
        self.queue_ptr = (ptr + self.num_protos) % self.queue.shape[0]

        # compute distances between the batch and the queue of candidates
        z_to_q = torch.norm(z[:, None, :] - self.queue[None, :, :], dim=2, p=2)
        all_dists, _ = torch.topk(z_to_q, self.topk, dim=1, largest=False)
        dist = all_dists[:, -1:]
        reward = dist

        #saving dist to see distribution for intrinsic reward
        #if step%1000 and step<300000:
        #    import IPython as ipy; ipy.embed(colors='neutral')
        #    dist_np = z_to_q
        #    dist_df = pd.DataFrame(dist_np.cpu())
        #    dist_df.to_csv(self.work_dir / 'dist_{}.csv'.format(step), index=False)  
        return reward

    def update_proto(self, obs, next_obs, step):
        metrics = dict()

        # normalize prototypes
        self.normalize_protos()

        #elf.protos)lf.protos)online network
        s = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        s = F.normalize(s, dim=1, p=2)
        self.protos = torch.as_tensor(self.protos, device=self.device)      
        similarity_to_protos = torch.mm(s, self.protos)
        #add log softmax
        #scores_s = similarity_to_protos.argmax(dim=0)
        log_p_s = F.log_softmax(similarity_to_protos, dim=1)
        # CxN
        
        # target network
        with torch.no_grad():
            t = self.encoder_target(next_obs)
            t = self.predictor_target(t)
            t = F.normalize(t, dim=1, p=2)
            q_t = torch.mm(t, self.protos)
            q_t = F.softmax(q_t, dim=1)
            #change to hard code?
        
        #using target network to add features of current samples to feature_queue
        ptr = self.feature_queue_ptr
        self.feature_queue[ptr:ptr + obs.shape[0]] = t
        self.feature_queue_ptr = (ptr + obs.shape[0]) % self.feature_queue.shape[0]
        if self.initialized==False and self.feature_queue[self.queue_size-1].sum()!=0:
            self.init_memory(t)
        
        # loss
        #debug
        target = q_t.argmax(dim=1)
        #B = torch.zeros((self.num_protos,), device=self.device)
        #B[torch.unique(target, return_counts=True)[0]] = torch.unique(target, return_counts=True)[1].float()
        #samples_weight=1/B
        #samples_weight[samples_weight==float('inf')]=0
        #import IPython as ipy; ipy.embed(colors='neutral')       
        histogram=np.bincount(target.detach().cpu().numpy(), minlength=16).astype(np.float32)
        inv_histogram=(1./(histogram+1e-10))**.5
        weight = inv_histogram/inv_histogram.sum()
        weight=torch.as_tensor(weight, device=self.device)
        #loss = -torch.mm((q_t * log_p_s), weight.tile((16,1))).sum(dim=1).mean()
        q_t = q_t*weight
        loss = - (q_t*log_p_s).sum(dim=1).mean() 
 
        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        self.proto_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.proto_opt.step()

        return metrics

    def update(self, replay_iter, step, actor1=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        

        batch = next(replay_iter)
        if actor1 and step % self.update_gc==0:
            obs, action, extr_reward, discount, next_obs, goal = utils.to_torch(
            batch, self.device)
            if self.obs_type=='states':
                goal = goal.reshape(-1, 2).float()
            
        elif actor1==False:
            obs, action, extr_reward, discount, next_obs = utils.to_torch(
                    batch, self.device)
        else:
            return metrics
        
        action = action.reshape(-1,2)
        discount = discount.reshape(-1,1)
        extr_reward = extr_reward.float()

        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            if actor1:
                goal = self.aug(goal)
           
        if actor1==False:

            if self.reward_free:
                metrics.update(self.update_proto(obs, next_obs, step))
                
                with torch.no_grad():
                    intr_reward = self.compute_intr_reward(next_obs, step)

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                
                reward = intr_reward
            else:
                reward = extr_reward

            if self.use_tb or self.use_wandb:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

            if not self.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()
            # update critic
            metrics.update(
                self.update_critic2(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

            # update actor
            metrics.update(self.update_actor2(obs.detach(), step))

            # update critic target
            #if step <300000:

            utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
            utils.soft_update_params(self.encoder, self.encoder_target,
                                                    self.encoder_target_tau)
            utils.soft_update_params(self.critic2, self.critic2_target,
                                 self.critic2_target_tau)
            
            self.update_centroids_memory()

        elif actor1 and step % self.update_gc==0:
            reward = extr_reward
            if self.use_tb or self.use_wandb:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)
            
            goal = self.encoder(goal)

            if not self.update_encoder:
            
                obs = obs.detach()
                next_obs = next_obs.detach()
                goal=goal.detach()
        
            # update critic
            metrics.update(
                self.update_critic(obs.detach(), goal.detach(), action, reward, discount,
                               next_obs.detach(), step))
            # update actor
            metrics.update(self.update_actor(obs.detach(), goal.detach(), step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        
        
        return metrics

    def get_q_value(self, obs,action):
        Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
        Q = torch.min(Q1, Q2)
        return Q
