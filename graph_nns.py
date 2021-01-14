import torch
from torch import distributions, nn
import torch.nn.functional as F

class AttentionGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False,nheads=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        self.squeeze_out=squeeze_out
        self.nheads=nheads
        inter=self.intermediate_size(in_features,nheads)
        self.attn = nn.MultiheadAttention(inter, nheads)
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None
        self.to_q=nn.Conv1d(in_features,inter,1)
        self.to_k=nn.Conv1d(in_features,inter,1)
        self.to_v=nn.Conv1d(in_features,inter,1)
        self.to_out=nn.Conv1d(inter,out_features,1)
    def forward(self, features):
        shape=features.shape
        justbatch=features.reshape(-1,shape[-2],shape[-1]).permute(0,2,1) #reshapes to a single batch dimension
        (q,k,v)=(self.to_q(justbatch),self.to_k(justbatch),self.to_v(justbatch))
        (q,k,v)=(q.permute(2,0,1),k.permute(2,0,1),v.permute(2,0,1))
        x=self.attn(q,k,v)[0]
        x=x.permute(1,2,0)
        x=self.to_out(x)
        x=x.permute(0,2,1)
        out=x.reshape(shape[0],shape[1],shape[2],shape[3],-1)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out
    def intermediate_size(self,in_features,nheads):
        return in_features+(-1*in_features)%nheads
    def reset_hidden_states(self):
        pass

class SqueezeExictationGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False,nheads=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        self.squeeze_out=squeeze_out
        self.conva=nn.Conv1d(in_features,out_features,1)
        self.conva=nn.Conv1d(out_features,out_features,1)
        self.lin=nn.Linear(out_features,out_features)
        print(inter)
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None
    def forward(self, features):
        shape=features.shape
        reshaped=features.reshape(-1,shape[-2],shape[-1]).permute(0,2,1) 
        x=self.conva(F.relu(reshaped))
        x=self.convb(F.relu(x))
        summed=torch.mean(x,dim=2)
        weighters=torch.sigmoid(self.lin(summed)).unsqueeze(dim=-1)
        x=x*weighters
        out=x.permute(0,2,1).reshape(shape[0],shape[1],shape[2],shape[3],-1)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out
    def reset_hidden_states(self):
        pass

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
                 preprocessing_layer=False, activation_func=nn.ReLU(), skip_connection_n=1, normalize=False):
        super().__init__()
        
        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes] * (n_hidden_layers + 1 + preprocessing_layer)
        elif len(layer_sizes) == 1:
            layer_sizes = layer_sizes * (n_hidden_layers + 1 + preprocessing_layer)
        if len(layer_sizes) != n_hidden_layers + 1 + preprocessing_layer:
            raise ValueError(f'len(layer_sizes) must equal n_hidden_layers + 1 (+ 1 again if preprocessing_layer), '
                             f'was {len(layer_sizes)} but should have been {n_hidden_layers+1+preprocessing_layer}')
        if preprocessing_layer:
            layers = [nn.Sequential(nn.Linear(in_features, layer_sizes[0]),
                                    activation_func),
                      layer_class(n_nodes, layer_sizes[0], layer_sizes[1], activation_func=activation_func)]
        else:
            layers = [layer_class(n_nodes, in_features, layer_sizes[0], activation_func=activation_func)]
        for i in range(n_hidden_layers):
            layers.append(layer_class(n_nodes, layer_sizes[i+preprocessing_layer], layer_sizes[i+1+preprocessing_layer],
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


class GraphNNPolicy(nn.Module):
    def __init__(self, in_features, n_nodes, n_hidden_layers, layer_sizes, layer_class,
                 activation_func=nn.ReLU(), skip_connection_n=1, normalize=False):
        super().__init__()

        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes] * (n_hidden_layers + 1)
        elif len(layer_sizes) == 1:
            layer_sizes = layer_sizes * (n_hidden_layers + 1)
        if len(layer_sizes) != n_hidden_layers + 1:
            raise ValueError(f'len(layer_sizes) must equal n_hidden_layers + 1, '
                             f'was {len(layer_sizes)} but should have been {n_hidden_layers + 1}')
        layers = [layer_class(n_nodes, in_features, layer_sizes[0], activation_func=activation_func)]
        for i in range(n_hidden_layers):
            layers.append(layer_class(n_nodes, layer_sizes[i], layer_sizes[i + 1],
                                      activation_func=activation_func, normalize=normalize))

        if skip_connection_n == 0:
            self.base = nn.Sequential(*layers)
        else:
            self.base = GraphNNResidualBase(layers, skip_connection_n)
        self.actor = layer_class(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)

    def forward(self, states):
        return self.actor(self.base(states))

    def sample_action(self, states, train=False):
        if train:
            logits = self.forward(states)
        else:
            with torch.no_grad():
                logits = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        probs_shape = probs.shape
        m = distributions.Categorical(probs.view(torch.prod(torch.tensor(probs_shape[:-1])).item(), probs_shape[-1]))
        sampled_actions = m.sample().view(probs_shape[:-1])
        if train:
            return sampled_actions, logits
        else:
            return sampled_actions

    def choose_best_action(self, states):
        with torch.no_grad():
            logits = self.forward(states)
            return logits.argmax(dim=-1)


class GraphNNQ(nn.Module):
    def __init__(self, in_features, n_nodes, n_hidden_layers, layer_sizes, layer_class,
                 activation_func=nn.ReLU(), skip_connection_n=1, normalize=False):
        super().__init__()
        self.action_space = n_nodes
        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes] * (n_hidden_layers + 1)
        elif len(layer_sizes) == 1:
            layer_sizes = layer_sizes * (n_hidden_layers + 1)
        if len(layer_sizes) != n_hidden_layers + 1:
            raise ValueError(f'len(layer_sizes) must equal n_hidden_layers + 1, '
                             f'was {len(layer_sizes)} but should have been {n_hidden_layers + 1}')
        layers = [layer_class(n_nodes, in_features, layer_sizes[0], activation_func=activation_func)]
        for i in range(n_hidden_layers):
            layers.append(layer_class(n_nodes, layer_sizes[i], layer_sizes[i + 1],
                                      activation_func=activation_func, normalize=normalize))

        if skip_connection_n == 0:
            self.base = nn.Sequential(*layers)
        else:
            self.base = GraphNNResidualBase(layers, skip_connection_n)
        self.critic = layer_class(n_nodes, layer_sizes[-1], 1, activation_func=nn.Identity(), squeeze_out=True)

    def forward(self, states):
        return self.critic(self.base(states))

    def sample_action_epsilon_greedy(self, states, epsilon):
        actions = self.choose_best_action(states)
        actions = torch.where(
            torch.rand(actions.shape, device=actions.device) < epsilon,
            torch.randint(self.action_space, size=actions.shape, device=actions.device),
            actions
        )
        return actions

    def choose_best_action(self, states):
        with torch.no_grad():
            q_vals = self.forward(states)
            return q_vals.argmax(dim=-1)
