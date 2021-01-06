import torch
from torch import distributions, nn
import torch.nn.functional as F


class FullyConnectedGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None
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
        out = self.recombine_features(torch.cat([features_transformed, messages], dim=-1))
        if self.normalize:
            out_shape = out.shape
            out = out.view(torch.prod(torch.tensor(out_shape[:-2])).item(), *out_shape[-2:])
            out = self.norm_layer(out.transpose(1, 2)).transpose(1, 2)
            out = out.view(out_shape)
        out = self.activation_func(out)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out

    def reset_hidden_states(self):
        pass


class SmallFullyConnectedGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None
        self.message_passing_mat = nn.Parameter(
            (torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)) / (n_nodes - 1),
            requires_grad=False
        )
        self.recombine_features = nn.Linear(in_features * 2, out_features)
        self.squeeze_out = squeeze_out
        # Initialize linear layer weights
        nn.init.normal_(self.recombine_features.weight, mean=0., std=0.2)
        nn.init.constant_(self.recombine_features.bias, 0.)

    def forward(self, features):
        messages = torch.matmul(self.message_passing_mat, features)
        out = self.recombine_features(torch.cat([features, messages], dim=-1))
        if self.normalize:
            out_shape = out.shape
            out = out.view(torch.prod(torch.tensor(out_shape[:-2])).item(), *out_shape[-2:])
            out = self.norm_layer(out.transpose(1, 2)).transpose(1, 2)
            out = out.view(out_shape)
        out = self.activation_func(out)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out

    def reset_hidden_states(self):
        pass


class SmallRecurrentGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features, recurrent_layer_class=nn.LSTM, squeeze_out=False, **kwargs):
        super().__init__()
        self.n_nodes = n_nodes
        self.message_passing_mat = nn.Parameter(
            (torch.ones((n_nodes, n_nodes)) - torch.eye(n_nodes)) / (n_nodes - 1),
            requires_grad=False
        )
        self.recombine_features = recurrent_layer_class(in_features * 2, out_features)
        self.rf_hidden = None
        self.squeeze_out = squeeze_out
        # Initialize LSTM layer weights
        #nn.init.normal_(self.recombine_features.weight_ih_l0, mean=0., std=0.2)
        #nn.init.normal_(self.recombine_features.weight_hh_l0, mean=0., std=0.2)
        #nn.init.normal_(self.recombine_features.bias_ih_l0, mean=0., std=0.2)
        #nn.init.normal_(self.recombine_features.bias_hh_l0, mean=0., std=0.2)

    def forward(self, features):
        orig_shape = features.shape
        messages = torch.matmul(self.message_passing_mat, features)
        """
        if self.rf_hidden is None:
            features_messages_combined, self.rf_hidden = self.recombine_features(
                torch.cat([features, messages], dim=-1).view(orig_shape[0], -1, orig_shape[-1] * 2)
            )
        else:
            features_messages_combined, self.rf_hidden = self.recombine_features(
                torch.cat([features, messages], dim=-1).view(orig_shape[0], -1, orig_shape[-1] * 2),
                self.rf_hidden
            )"""
        features_messages_combined, self.rf_hidden = self.recombine_features(
            torch.cat([features, messages], dim=-1).view(orig_shape[0], -1, orig_shape[-1] * 2),
            self.rf_hidden
        )
        features_messages_combined = features_messages_combined.view((*orig_shape[:-1], -1))
        if self.squeeze_out:
            return features_messages_combined.squeeze(dim=-1)
        else:
            return features_messages_combined

    def reset_hidden_states(self):
        self.rf_hidden = None

    def detach_hidden_states(self):
        self.rf_hidden = [h.detach() for h in self.rf_hidden]


class GraphNNResidualBase(nn.Module):
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
                if identity is not None and identity.shape == x.shape:
                    x = x + identity
                identity = x
            else:
                x = layer(x)
        return x

    def reset_hidden_states(self):
        [layer.reset_hidden_states() for layer in self.layers]

    def detach_hidden_states(self):
        [layer.detach_hidden_states() for layer in self.layers]
        
        
class GraphNNActorCritic(nn.Module):
    def __init__(self, in_features, n_nodes, n_hidden_layers, layer_sizes, layer_class,
                 activation_func=nn.ReLU(), skip_connection_n=1, normalize=False):
        super().__init__()
        
        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes] * (n_hidden_layers + 1)
        if len(layer_sizes) != n_hidden_layers + 1:
            raise ValueError(f'len(layer_sizes) must equal n_hidden_layers + 1, '
                             f'was {len(layer_sizes)} but should have been {n_hidden_layers+1}')
        layers = [layer_class(n_nodes, in_features, layer_sizes[0], activation_func=activation_func)]
        for i in range(n_hidden_layers):
            layers.append(layer_class(n_nodes, layer_sizes[i], layer_sizes[i+1],
                                      activation_func=activation_func, normalize=normalize))
        
        if skip_connection_n == 0:
            self.base = nn.Sequential(*layers)
        else:
            self.base = GraphNNResidualBase(layers, skip_connection_n)
        self.actor = layer_class(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)
        self.critic = layer_class(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)
    
    def forward(self, states):
        base_out = self.base(states)
        return self.actor(base_out), self.critic(base_out).mean(dim=-1)
    
    def sample_action(self, states, train=False):
        if train:
            logits, values = self.forward(states)
        else:
            with torch.no_grad():
                logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        seq_len, n_envs, n_players, n_bandits = probs.shape
        m = distributions.Categorical(probs.view(seq_len * n_envs * n_players, n_bandits))
        sampled_actions = m.sample().view(seq_len, n_envs, n_players)
        if train:
            return sampled_actions, (logits, values)
        else:
            return sampled_actions
    
    def choose_best_action(self, states):
        with torch.no_grad():
            logits, _ = self.forward(states)
            return logits.argmax(dim=-1)

    def reset_hidden_states(self):
        self.base.reset_hidden_states()
        self.actor.reset_hidden_states()
        self.critic.reset_hidden_states()

    def detach_hidden_states(self):
        self.base.detach_hidden_states()
        self.actor.detach_hidden_states()
        self.critic.detach_hidden_states()
