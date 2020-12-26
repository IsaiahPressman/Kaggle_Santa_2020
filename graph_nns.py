import torch
from torch import distributions, nn
import torch.nn.functional as F


class FullyConnectedGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features, activation_func=nn.ReLU(), squeeze_out=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.transform_features = nn.Linear(in_features, out_features)
        self.message_passing_mat = nn.Parameter(
            (torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)) / (n_nodes - 1),
            requires_grad=False
        )
        self.recombine_features = nn.Linear(out_features*2, out_features)
        self.squeeze_out = squeeze_out
        # Initialize linear layer weights
        nn.init.normal_(self.transform_features.weight, mean=0., std=0.2)
        nn.init.normal_(self.recombine_features.weight, mean=0., std=0.2)
        nn.init.constant_(self.transform_features.bias, 0.)
        nn.init.constant_(self.recombine_features.bias, 0.)
    
    def forward(self, features):
        features_transformed = self.activation_func(
            self.transform_features(features)
        )
        messages = torch.matmul(self.message_passing_mat, features_transformed)
        features_messages_combined = self.activation_func(
            self.recombine_features(torch.cat([features_transformed, messages], dim=-1))
        )
        if self.squeeze_out:
            return features_messages_combined.squeeze(dim=-1)
        else:
            return features_messages_combined
        

class GraphNN_Residual_Base(nn.Module):
    def __init__(self, layers, skip_connection_n):
        super().__init__()
        assert skip_connection_n >= 1
        self.layers = nn.ModuleList(layers)
        self.skip_connection_n = skip_connection_n
        
    def forward(self, x):
        identity = None
        for layer_num, layer in enumerate(self.layers):
            if (len(self.layers) - layer_num - 1) % self.skip_connection_n == 0:
                x = layer(x)
                if identity is not None:
                    x += identity
                identity = x
            else:
                x = layer(x)
        return x
        
        
class GraphNN_A3C(nn.Module):
    def __init__(self, in_features, n_nodes, n_hidden_layers, layer_sizes,
                 activation_func = nn.ReLU(), skip_connection_n=2):
        super().__init__()
        
        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes] * (n_hidden_layers + 1)
        assert len(layer_sizes) == n_hidden_layers + 1, f'len(layer_sizes) must equal n_hidden_layers + 1, was {len(layer_sizes)} but should have been {n_hidden_layers+1}'
        layers = [FullyConnectedGNNLayer(n_nodes, in_features, layer_sizes[0], activation_func=activation_func)]
        for i in range(n_hidden_layers):
            layers.append(FullyConnectedGNNLayer(n_nodes, layer_sizes[i], layer_sizes[i+1], activation_func=activation_func))
        
        if skip_connection_n == 0:
            self.base = nn.Sequential(*layers)
        else:
            self.base = GraphNN_Residual_Base(layers, skip_connection_n)
        self.actor = FullyConnectedGNNLayer(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)
        self.critic = FullyConnectedGNNLayer(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)
    
    def forward(self, states):
        base_out = self.base(states)
        return self.actor(base_out), self.critic(base_out).mean(dim=-1)
    
    def sample_action(self, states):
        with torch.no_grad():
            logits, _ = self.forward(states)
            probs = F.softmax(logits, dim=-1)
            batch_size, n_envs, n_players, n_bandits = probs.shape
            m = distributions.Categorical(probs.view(batch_size * n_envs * n_players, n_bandits))
            return m.sample().view(batch_size, n_envs, n_players)
    
    def choose_best_action(self, states):
        with torch.no_grad():
            logits, _ = self.forward(states)
            return logits.argmax(dim=-1)
    
    def loss_func(self, states, actions, v_t):
        #print(f'states.shape: {states.shape}, actions.shape: {actions.shape}, v_t.shape: {v_t.shape}')
        logits, values = self.forward(states)
        
        #print(f'logits.shape: {logits.shape}, values.shape: {values.shape}')
        td = v_t - values
        #critic_loss = td.pow(2).view(-1)
        # Huber loss
        critic_loss = F.smooth_l1_loss(v_t, values, reduction='none').view(-1)
        
        probs = F.softmax(logits, dim=-1)
        batch_size, n_envs, n_players, n_bandits = probs.shape
        m = distributions.Categorical(probs.view(batch_size * n_envs * n_players, n_bandits))
        #print(f'm.log_prob(actions.view(batch_size * n_envs * n_players)).shape: {m.log_prob(actions.view(batch_size * n_envs * n_players)).shape}, td.shape: {td.shape}')
        actor_loss = -(m.log_prob(actions.view(-1)) * td.detach().view(-1))
        total_loss = (critic_loss + actor_loss).mean()
        return total_loss
