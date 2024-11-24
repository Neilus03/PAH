from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaSequential
import torch
import re
import torch.nn as nn


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)



def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, 
    as for instance output by a hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        if self.bias is not None:
            bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        # if self.bias is not None:
        #     output += bias.unsqueeze(-2)

        return output



class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None,bias = True):
        super().__init__()

        # nonlinearity = 'sine'

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        # nls_and_inits = 
                         

        nl, nl_weight_init, first_layer_init = nn.ReLU(inplace=True), init_weights_normal, None

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features,bias=bias), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features,bias=bias), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features,bias=bias)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features,bias=bias), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # print('passing on with siren ', siren, get_subdict(params, 'net').keys())
        output = self.net(coords, params=get_subdict(params, 'net'))
        # output = self.net(coords)
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

    def get_optimizer_list(self):
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-4}]
        return optimizer_list



########################
# HyperNetwork modules
class HyperNetwork(nn.Module):
    def __init__(self, 
              hyper_in_features, 
              hyper_hidden_layers, 
              hyper_hidden_features, 
              hypo_module, 
              activation='relu'):
        '''
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()
        
        hypo_parameters = hypo_module.state_dict().items()
        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []

        for name, param in hypo_parameters:
            # print("Inside HyperNetwork",name, param.size())
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features, 
                    out_features=int(torch.prod(torch.tensor(param.size()))),
                    num_hidden_layers=hyper_hidden_layers, 
                    hidden_features=hyper_hidden_features,
                    outermost_linear=True,
                    nonlinearity=activation)

            if 'weight' in name:
                hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name or 'offsets' in name:
                hn.net[-1].apply(lambda m: hyper_bias_init(m))
            
            self.nets.append(hn)

    def forward(self, input_hyp):
        '''
        Args:-
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(input_hyp).reshape(batch_param_shape)
            
        return params

    def get_optimizer_list(self):
        optimizer_list = list()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            optimizer_list.extend(net.get_optimizer_list())
        return optimizer_list


############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1e1



def hyper_bias_init(m, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e1