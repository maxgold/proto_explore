import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
class Encoder1(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32*9*9
        #40*40*32
        #19*19*32
        #17*17*32
        #9*9*32
        
        #number of parameters:
        #(3*3*3+1)*32 = 896
        #(3*3*32+1)*32 = 9248
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Encoder2(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 64*7*7
        #(84-8+1)/4 -> 19*19*32
        #(19-4+1)/2 -> 8*8*64
        #(8-3+1)/1 -> 6*6*64
        #number of parameters:
        #(8*8*9+1)*32 = 18464
        #(4*4*32+1)*64 = 
        #(3*3*64+1)*64 = 
        #total: 88224
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                                    nn.ReLU(), nn.Conv2d(32,64,kernel_size=4,stride=2),
                                    nn.ReLU(), nn.Conv2d(64,64,kernel_size=3, stride=1),
                                    nn.ReLU())


        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Encoder3(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 64*2*2
        #40*40*32
        #19*19*64
        #15*15*64
        #7*7*64
        #3*3*64
        #2*2*64
        
        #number of parameters:
        #(3*3*3+1)*32 = 896
        #(3*3*32+1)*32 = 9248
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 5, stride=2),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     nn.Conv2d(32, 64, 4, stride=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     nn.Conv2d(64, 64, 2, stride=1),nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Encoder_sl(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 64*7*7
        #(84-8+1)/4 -> 19*19*32
        #(19-4+1)/2 -> 8*8*64
        #(8-3+1)/1 -> 6*6*64
        #number of parameters:
        #(8*8*9+1)*32 = 18464
        #(4*4*32+1)*64 = 
        #(3*3*64+1)*64 = 
        #total: 88224
        self.cnn1 = nn.Conv2d(obs_shape[0], 32, 8, stride=4)
        self.cnn2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.cnn3 = nn.Conv2d(64,64,kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, feature_dim)
        #send result of fc2 to actor
        self.fc3 = nn.Linear(feature_dim, 4)

        self.apply(utils.weight_init)

    def forward(self, obs):
        
        obs = obs / 255.0 - 0.5
        x1 = self.cnn1(obs)
        x1 = F.relu(x1)
        x2 = self.cnn2(x1)
        x2 = F.relu(x2)
        x3 = self.cnn3(x2)
        x3 = F.relu(x3)
        x3 = x3.view(x3.shape[0], -1)
        x4 = self.fc1(x3)
        x4 = F.relu(x4)
        x5 = self.fc2(x4)
        x5 = F.relu(x5)
        #send result of fc2 to actor
        x6 = self.fc3(x5)
        x6 = F.relu(x6)

        return x5, x6


class Actor_gc(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim+goal_dim, feature_dim), 
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, goal, std):
        obs_goal = torch.cat([obs, goal], dim=-1)
        h = self.trunk(obs_goal)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

class Actor_proto(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std, scale=None):
        h = self.trunk(obs)
        mu = self.policy(h)
        if scale is not None:
            mu = torch.tanh(mu)*scale
        else:
            mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist
    

class Critic_gc(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim + goal_dim, feature_dim),
                                        nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, goal, action):
        inpt = torch.cat([obs, goal], dim=-1) if self.obs_type == 'pixels' else torch.cat([obs, goal, action],dim=-1)
        
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2

class Critic_proto(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.obs_type = obs_type
        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)
        return q1, q2

class Actor_sl(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
#         self.trunk = nn.Sequential(nn.Linear(obs_dim+goal_dim, feature_dim), 
#                                    nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim+feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, h, std):
#         obs_goal = torch.cat([obs, goal], dim=-1)
#         h = self.trunk(obs_goal)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist



class Critic_sl(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
#             # for pixels actions will be added after trunk
#             self.trunk = nn.Sequential(nn.Linear(obs_dim + goal_dim, feature_dim),
#                                        nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim+feature_dim + action_dim

        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, h, action):
#         inpt = torch.cat([obs, goal], dim=-1) if self.obs_type == 'pixels' else torch.cat([obs, goal, action],dim=-1)
        
#         h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2

class LinearInverse(nn.Module):
    # NOTE: For now the input will be [robot_rotation, box_rotation, distance_bw]
    def __init__(self, feature_dim, action_dim, hidden_dim, init_from_ddpg=False, obs_type='pixels'):
        super().__init__()
        self.init_from_ddpg = init_from_ddpg
        if self.init_from_ddpg is False and obs_type == 'pixels':
            input_dim = feature_dim*2
        else:
            input_dim = feature_dim
            
        self.model = nn.Sequential(
            nn.Linear(input_dim, feature_dim*4), # input_dim*2: For current and goal obs
            nn.ReLU(),
            nn.Linear(feature_dim*4, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), action_dim)
        )

    def forward(self, obs, goal):
        if self.init_from_ddpg is False:
            x = torch.cat((obs, goal), dim=-1)
        else:
            x = obs
        x = self.model(x)
        return x
