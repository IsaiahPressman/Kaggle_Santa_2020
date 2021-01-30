from copy import copy
import torch
from torch import distributions, nn
import torch.nn.functional as F


class AttentionGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False, nheads=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        self.squeeze_out = squeeze_out
        self.nheads = nheads
        # Register nheads to prevent later accidentally loading the model with the wrong number of heads
        self.register_nheads = nn.Parameter(
            torch.zeros(self.nheads),
            requires_grad=False
        )
        inter = self.intermediate_size(in_features,nheads)
        self.attn = nn.MultiheadAttention(inter, nheads)
        if self.normalize:
            self.norm_layer_1 = nn.BatchNorm1d(out_features)
            self.norm_layer_2 = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer_1 = None
            self.norm_layer_2 = None
        self.in_features = in_features
        self.out_features = out_features
        if in_features == out_features:
            self.to_q = nn.Linear(in_features, inter)
            self.to_k = nn.Linear(in_features, inter)
            self.to_v = nn.Linear(in_features, inter)
            self.to_out = nn.Linear(inter, out_features)
            self.feedforwarda = nn.Linear(out_features, out_features)
            self.feedforwardb = nn.Linear(out_features, out_features)
        else:
            self.lin = nn.Linear(in_features, out_features)
        
    def forward(self, features):
        if self.in_features != self.out_features:
            out = self.lin(features)
        else:
            shape = features.shape
            justbatch = features.view(-1, *shape[-2:]) # reshapes to a single batch dimension
            (q, k, v) = (self.to_q(justbatch), self.to_k(justbatch), self.to_v(justbatch))
            (q, k, v) = (q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2))
            x = self.attn(q, k, v)[0]
            x = x.permute(1, 0, 2)
            x = self.to_out(x)
            mid = x + justbatch # residual connection part 1
            if self.normalize:
                mid = self.norm_layer_1(mid.transpose(1, 2)).transpose(1, 2)
            x = self.activation_func(self.feedforwarda(mid))
            x = self.activation_func(self.feedforwardb(x))
            x = x + mid
            if self.normalize:
                x = self.norm_layer_2(x.transpose(1, 2)).transpose(1, 2)
            out = x.view(*shape[:-1], -1)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out

    def intermediate_size(self, out_features, nheads):
        return out_features + (-1 * out_features) % nheads

    def reset_hidden_states(self):
        pass

class MaxAndMeanGNNLayer(nn.Module):
    def __init__(self, n_nodes, in_features, out_features,
                 activation_func=nn.ReLU(), normalize=False, squeeze_out=False,nheads=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.normalize = normalize
        self.squeeze_out=squeeze_out
        self.in_features=in_features
        self.out_features=out_features
        self.lina=nn.Linear(in_features,out_features)
        self.linb=nn.Linear(out_features*3,out_features)
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None

    def forward(self, features):
        shape=features.shape
        reshaped=features.view(-1,shape[-2],shape[-1])
        x=self.lina(F.relu(reshaped))
        summed=torch.mean(x,dim=1,keepdim=True).expand(-1,100,-1)
        maxed=(torch.max(x,dim=1,keepdim=True)[0]).expand(-1,100,-1)
        x=torch.cat([x,summed,maxed],dim=2)
        x=self.linb(F.relu(x))
        if(self.normalize):
            x=self.norm_layer(x.transpose(1, 2)).transpose(1, 2)
        out=x.view(shape[0],shape[1],shape[2],shape[3],-1)
        if self.squeeze_out:
            return out.squeeze(dim=-1)
        else:
            return out

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
        self.in_features=in_features
        self.out_features=out_features
        self.lina=nn.Linear(in_features,out_features)
        self.linb=nn.Linear(out_features,out_features)
        self.se_lin=nn.Linear(out_features,out_features)
        if self.normalize:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None

    def forward(self, features):
        shape=features.shape
        reshaped=features.reshape(-1,shape[-2],shape[-1])
        x=self.lina(F.relu(reshaped))
        x=self.linb(F.relu(x))
        summed=torch.mean(x,dim=1)
        weighters=torch.sigmoid(self.se_lin(summed)).unsqueeze(dim=1)
        x=x*weighters
        if(self.normalize):
            x=self.norm_layer(x)
        out=x.reshape(shape[0],shape[1],shape[2],shape[3],-1)
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


class SmallMeanGNNLayer(nn.Module):
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
        self.recombine_features = nn.Linear(in_features * 2, out_features)
        self.squeeze_out = squeeze_out
        # Initialize linear layer weights
        # default inintilizaer tends to work better for resnets
        # nn.init.normal_(self.recombine_features.weight, mean=0., std=0.2)
        # nn.init.constant_(self.recombine_features.bias, 0.)

    def forward(self, features):
        features_shape = features.shape
        features = features.reshape(-1, *features_shape[-2:])
        messages = features.mean(dim=-2, keepdims=True).expand(-1, self.n_nodes, -1)
        out = self.recombine_features(torch.cat([features, messages], dim=-1))
        if self.normalize:
            out = self.norm_layer(out.transpose(1, 2)).transpose(1, 2)
        out = self.activation_func(out).view(*features_shape[:-1], -1)
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


class GraphNNPreprocessingLayer(nn.Module):
    def __init__(self, preprocessing_layer, mean_out=True, ravel_out=False):
        super().__init__()
        self.preprocessing_layer = preprocessing_layer
        self.mean_out = mean_out
        self.ravel_out = ravel_out
        assert not (self.mean_out and self.ravel_out)

    def forward(self, x):
        out = self.preprocessing_layer(x)
        if self.mean_out:
            out = out.mean(dim=-2)
        if self.ravel_out:
            out = out.view(*out.shape[:-2], -1)
        return out
        
        
class GraphNNActorCritic(nn.Module):
    def __init__(self, in_features, n_nodes, n_hidden_layers, layer_sizes, layer_class,
                 preprocessing_layer=None, skip_connection_n=1, deprecated_version=False, **layer_class_kwargs):
        super().__init__()
        
        # Define network
        if type(layer_sizes) == int:
            layer_sizes = [layer_sizes]
        if len(layer_sizes) == 1:
            layer_sizes = layer_sizes * (n_hidden_layers + 1)
        if len(layer_sizes) != n_hidden_layers + 1:
            raise ValueError(f'len(layer_sizes) must equal n_hidden_layers + 1, '
                             f'was {len(layer_sizes)} but should have been {n_hidden_layers + 1}')
        layers = []
        if preprocessing_layer is not None:
            layers.append(preprocessing_layer)
        if deprecated_version:
            layer_class_kwargs_copy = copy(layer_class_kwargs)
            layer_class_kwargs_copy.pop('normalize', None)
            layers.append(layer_class(n_nodes, in_features, layer_sizes[0], **layer_class_kwargs_copy))
        else:
            layers.append(layer_class(n_nodes, in_features, layer_sizes[0], **layer_class_kwargs))
        for i in range(n_hidden_layers):
            layers.append(layer_class(n_nodes, layer_sizes[i], layer_sizes[i + 1],
                                      **layer_class_kwargs))
        
        if skip_connection_n == 0:
            self.base = nn.Sequential(*layers)
        else:
            self.base = GraphNNResidualBase(layers, skip_connection_n)
        layer_class_kwargs_copy = copy(layer_class_kwargs)
        layer_class_kwargs_copy.pop('activation_func', None)
        layer_class_kwargs_copy.pop('normalize', None)
        self.actor = layer_class(n_nodes, layer_sizes[-1], 1,
                                 activation_func=nn.Identity(), squeeze_out=True,
                                 **layer_class_kwargs_copy
                                 )
        self.critic = layer_class(n_nodes, layer_sizes[-1], 1,
                                  activation_func=nn.Identity(), squeeze_out=True,
                                  **layer_class_kwargs_copy
                                  )
    
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


class LearnableWeightsModel(nn.Module):
    def __init__(self, n_nodes, n_models, hidden_layer_size, activation_func=nn.ReLU()):
        super().__init__()
        self.processing_layer = SmallFullyConnectedGNNLayer(n_nodes, n_models * 2, hidden_layer_size,
                                                            activation_func=activation_func)
        self.weights_out = SmallFullyConnectedGNNLayer(n_nodes, hidden_layer_size, n_models,
                                                       activation_func=nn.Identity())
        self.value_out = SmallFullyConnectedGNNLayer(n_nodes, hidden_layer_size, 1,
                                                     squeeze_out=True, activation_func=nn.Identity())

    def forward(self, logits, values):
        values = values.unsqueeze(dim=-2).expand_as(logits)
        x = torch.cat([logits, values], dim=-1)
        x = self.processing_layer(x)
        weights = self.weights_out(x).mean(dim=-2, keepdims=True)
        weights = F.softmax(weights, dim=-1)
        values = self.value_out(x).mean(dim=-1)

        return values, weights


class GraphNNActorCriticEnsemble(nn.Module):
    def __init__(self, actor_critic_models, learnable_weights_model=None, weight_logits=False):
        super().__init__()
        self.models = nn.ModuleList(actor_critic_models)
        for m in self.models:
            for param in m.parameters():
                param.requires_grad = False
        self.learnable_weights_model = learnable_weights_model
        self.weight_logits = weight_logits
        # Register weight_logits to prevent later accidentally loading the model with the wrong setting
        self.register_weight_logits = nn.Parameter(
            torch.zeros(int(self.weight_logits) + 1),
            requires_grad=False
        )

    def forward(self, states):
        logits = []
        values = []
        for m in self.models:
            l, v = m(states)
            logits.append(l)
            values.append(v)
        logits = torch.stack(logits, dim=-1)
        values = torch.stack(values, dim=-1)
        if self.learnable_weights_model is not None:
            values, weights = self.learnable_weights_model(logits, values)
        else:
            values = values.mean(dim=-1)
            weights = torch.ones_like(logits) / logits.shape[-1]

        # Original way:
        if self.weight_logits:
            logits = torch.sum(logits * weights, dim=-1)
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.sum(F.softmax(logits, dim=-2) * weights, dim=-1)
        return probs, values

    def choose_best_action(self, states):
        with torch.no_grad():
            probs, _ = self.forward(states)
            return probs.argmax(dim=-1)

    def sample_action(self, states, train=False):
        if train:
            probs, values = self.forward(states)
        else:
            with torch.no_grad():
                probs, values = self.forward(states)
        seq_len, n_envs, n_players, n_bandits = probs.shape
        m = distributions.Categorical(probs.view(seq_len * n_envs * n_players, n_bandits))
        sampled_actions = m.sample().view(seq_len, n_envs, n_players)
        if train:
            return sampled_actions, (probs, values)
        else:
            return sampled_actions


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
