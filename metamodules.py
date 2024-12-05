from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaSequential
import torch
import re
import torch.nn as nn


def get_subdict(dictionary, key=None):
    """
    Extracts a subdictionary based on a given key prefix.
    
    Args:
        dictionary (OrderedDict): The dictionary to extract from.
        key (str, optional): The prefix to filter keys. If None or empty, returns the entire dictionary.

    Returns:
        OrderedDict: A subdictionary containing only the keys that start with the specified prefix,
        with the prefix removed from the keys.
    
    Example:
        dictionary = {'layer1.weight': ..., 'layer2.bias': ...}
        get_subdict(dictionary, 'layer1') -> {'weight': ...}
    """
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)



def init_weights_normal(m):
    """
    Initializes weights of a given module with Kaiming normal initialization.
    
    Args:
        m (nn.Module): The module to initialize. Should be of type BatchLinear or nn.Linear.
    
    Note:
        The initialization uses `relu` as the nonlinearity and `fan_in` mode.
    """
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return self._conv_forward(input, params['weight'], bias)

class BatchLinear(nn.Linear, MetaModule):
    """
    A linear meta-layer that supports batched weight matrices and biases.
    
    This layer can process weight and bias matrices in a batched fashion, which is useful
    when using it in combination with hypernetworks that output parameters for multiple tasks
    or instances. It extends PyTorch's `nn.Linear` to handle batched weights and integrates
    with MetaModules for parameter swapping.

    Inherits:
        nn.Linear: Provides standard linear layer functionality.
        MetaModule: Enables parameter substitution for meta-learning tasks.

    Attributes:
        __doc__: Inherits the docstring from `nn.Linear`.

    Methods:
        forward(input, params=None): Computes the output of the linear layer using the given input
                                     and optionally supplied weights and biases.
    """
    __doc__ = nn.Linear.__doc__  # Inherit nn.Linear's documentation for completeness

    def forward(self, input, params=None):
        """
        Forward pass for the batched linear layer.

        Args:
            input (torch.Tensor): The input tensor of shape `(batch_size, ..., input_dim)`.
            params (OrderedDict, optional): An optional dictionary containing the weights and biases.
                - `params['weight']`: Batched weight matrix of shape `(batch_size, output_dim, input_dim)`.
                - `params['bias']` (optional): Batched bias vector of shape `(batch_size, output_dim)`.

        Returns:
            torch.Tensor: The result of applying the linear transformation, of shape 
            `(batch_size, ..., output_dim)`.

        Notes:
            - If `params` is not provided, the layer uses its own parameters.
            - The weight matrix is permuted to align the last two dimensions for matrix multiplication.
        """
        # Use layer's own parameters if none are supplied
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Extract weight and bias from the provided or default parameters
        weight = params['weight']
        bias = params.get('bias', None)  # Bias is optional

        # Perform batched matrix multiplication
        # Permutes weight dimensions for proper broadcasting with input
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        # Add bias if available
        if bias is not None:
            output += bias.unsqueeze(-2)

        return output



class FCBlock(MetaModule):
    """
    A fully connected (FC) neural network with support for weight swapping using a hypernetwork.

    This module can function as a regular fully connected network or integrate with a hypernetwork
    to allow swapping out weights dynamically during training or inference. It includes support for
    customizable nonlinearity, initialization, and flexible architectures with multiple hidden layers.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        num_hidden_layers (int): Number of hidden layers.
        hidden_features (int): Number of neurons in each hidden layer.
        outermost_linear (bool, optional): If True, the last layer does not include a nonlinearity.
                                           Default is False.
        nonlinearity (str, optional): Nonlinearity to use between layers. Default is 'relu'.
        weight_init (callable, optional): Custom weight initialization function. Default is None.
        bias (bool, optional): Whether to include bias in the linear layers. Default is True.

    Attributes:
        net (MetaSequential): The sequential stack of layers making up the network.
        weight_init (callable): The weight initialization function applied to the layers.
        first_layer_init (callable): Special initialization for the first layer (if applicable).
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, bias=True):
        super().__init__()

        # Initialize attributes for special initialization
        self.first_layer_init = None

        # Set default nonlinearity and initialization
        nl, nl_weight_init, first_layer_init = nn.ReLU(inplace=True), init_weights_normal, None

        # Overwrite weight initialization if a custom function is provided
        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        # Build the network as a stack of layers
        self.net = []

        # Add the first layer
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features, bias=bias), nl
        ))

        # Add hidden layers
        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features, bias=bias), nl
            ))

        # Add the final layer, optionally without nonlinearity
        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features, bias=bias)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features, bias=bias), nl
            ))

        # Convert the list of layers into a MetaSequential module
        self.net = MetaSequential(*self.net)

        # Apply weight initialization to the network
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        # Apply special initialization to the first layer if specified
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        """
        Forward pass through the fully connected network.

        Args:
            coords (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.
            params (OrderedDict, optional): Dictionary of parameters for meta-learning. If None,
                                            the layer uses its own parameters.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_features)`.
        """
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Forward pass through the network using meta-learning parameters
        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        """
        Forward pass that returns intermediate activations for each layer.

        Args:
            coords (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.
            params (OrderedDict, optional): Dictionary of parameters for meta-learning.
                                            If None, the layer uses its own parameters.
            retain_grad (bool, optional): If True, retains gradients for intermediate activations.
                                          Default is False.

        Returns:
            OrderedDict: Dictionary of intermediate activations, including the input and output of each layer.
        """
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        # Clone input to ensure it does not modify the original tensor
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x

        # Pass through each layer and collect activations
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, f'net.{i}')
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, f'{j}'))
                else:
                    x = sublayer(x)

                # Retain gradient for intermediate activations if specified
                if retain_grad:
                    x.retain_grad()
                activations[f'{sublayer.__class__.__name__}_{i}'] = x
        return activations

    def get_optimizer_list(self):
        """
        Returns a list of optimizers for the network.

        Returns:
            list of dict: List containing optimizer configurations for the parameters.
        """
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-4}]
        return optimizer_list



########################
# HyperNetwork modules
class HyperNetwork(nn.Module):
    """
    A hypernetwork module for generating parameters for a target (hypo) module.

    The hypernetwork takes an input embedding and generates weights and biases
    for the target (hypo) module dynamically. Each parameter in the hypo module
    is generated by a small fully connected (FC) network.

    Args:
        hyper_in_features (int): Number of input features to the hypernetwork.
        hyper_hidden_layers (int): Number of hidden layers in each FCBlock of the hypernetwork.
        hyper_hidden_features (int): Number of hidden units in each hidden layer of the FCBlock.
        hypo_module (MetaModule): The target module whose parameters are generated by the hypernetwork.
        activation (str, optional): Activation function to use in the FCBlocks. Default is 'relu'.

    Attributes:
        names (list of str): Names of the parameters in the hypo module.
        nets (nn.ModuleList): A list of FCBlock networks, one for each parameter in the hypo module.
        param_shapes (list of torch.Size): Shapes of the parameters in the hypo module.
    """

    def __init__(self, 
                 hyper_in_features, 
                 hyper_hidden_layers, 
                 hyper_hidden_features, 
                 hypo_module, 
                 activation='relu'):
        super().__init__()

        # Extract parameter names, shapes, and initialize FCBlocks for each parameter
        hypo_parameters = hypo_module.state_dict().items()

        self.names = []  # Stores the names of the parameters in the hypo module
        self.nets = nn.ModuleList()  # Stores the FCBlock for each parameter
        self.param_shapes = []  # Stores the shape of each parameter

        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            # Create an FCBlock for each parameter
            if 'bias' in name:
                hn = FCBlock(in_features=hyper_in_features, 
                    out_features=int(torch.prod(torch.tensor(param.size()))),
                    num_hidden_layers=0, 
                    hidden_features=hyper_in_features,
                    outermost_linear=True,
                    nonlinearity=activation)
            
            else:
                hn = FCBlock(in_features=hyper_in_features, 
                        out_features=int(torch.prod(torch.tensor(param.size()))),
                        num_hidden_layers=hyper_hidden_layers, 
                        hidden_features=hyper_hidden_features,
                        outermost_linear=True,
                        nonlinearity=activation)

            # Apply custom initialization based on the parameter type
            if 'weight' in name:
                hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name or 'offsets' in name:
                hn.net[-1].apply(lambda m: hyper_bias_init(m))
            
            self.nets.append(hn)  # Add the FCBlock to the list

    def forward(self, input_hyp):
        """
        Forward pass of the hypernetwork.

        Args:
            input_hyp (torch.Tensor): Input tensor (embedding) of shape `(batch_size, hyper_in_features)`.

        Returns:
            OrderedDict: A dictionary where keys are parameter names from the hypo module and
                         values are the corresponding generated parameters, reshaped to match
                         their original shapes in the hypo module.
        """
        params = OrderedDict()

        # Generate each parameter using the corresponding FCBlock
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape  # Add batch dimension
            params[name] = net(input_hyp).reshape(batch_param_shape)
        
        return params

    def get_optimizer_list(self):
        """
        Creates a list of optimizers for the hypernetwork's parameters.

        Returns:
            list: A list of dictionaries, each containing parameter groups for optimization.
        """
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-3}]
        return optimizer_list



class HyperNetwork_seq(nn.Module):
    """
    A hypernetwork module for generating parameters for a target (hypo) module.

    The hypernetwork takes an input embedding and generates weights and biases
    for the target (hypo) module dynamically. Each parameter in the hypo module
    is generated by a small fully connected (FC) network.

    Args:
        hyper_in_features (int): Number of input features to the hypernetwork.
        hyper_hidden_layers (int): Number of hidden layers in each FCBlock of the hypernetwork.
        hyper_hidden_features (int): Number of hidden units in each hidden layer of the FCBlock.
        hypo_module (MetaModule): The target module whose parameters are generated by the hypernetwork.
        activation (str, optional): Activation function to use in the FCBlocks. Default is 'relu'.

    Attributes:
        names (list of str): Names of the parameters in the hypo module.
        nets (nn.ModuleList): A list of FCBlock networks, one for each parameter in the hypo module.
        param_shapes (list of torch.Size): Shapes of the parameters in the hypo module.
    """

    def __init__(self, 
                 hyper_in_features, 
                 hyper_hidden_layers, 
                 hyper_hidden_features, 
                 hypo_module, 
                 activation='relu'):
        super().__init__()

        # Extract parameter names, shapes, and initialize FCBlocks for each parameter
        hypo_parameters = hypo_module.state_dict().items()

        self.names = []  # Stores the names of the parameters in the hypo module
        self.nets = nn.ModuleList()  # Stores the FCBlock for each parameter
        self.param_shapes = []  # Stores the shape of each parameter

        for i, (name, param) in enumerate(hypo_parameters):
            self.names.append(name)
            self.param_shapes.append(param.size())

            if i != 0:
                in_features = last_out_shape
            else:
                in_features = hyper_in_features
            
            # Create an FCBlock for each parameter
            # if 'bias' in name:
            #     hn = FCBlock(in_features=in_features, 
            #         out_features=int(torch.prod(torch.tensor(param.size()))),
            #         num_hidden_layers=0, 
            #         hidden_features=hyper_in_features,
            #         outermost_linear=True,
            #         nonlinearity=activation)
            
            # else:
            hn = FCBlock(in_features=in_features, 
                    out_features=int(torch.prod(torch.tensor(param.size()))),
                    num_hidden_layers=hyper_hidden_layers, 
                    hidden_features=hyper_hidden_features,
                    outermost_linear=True,
                    nonlinearity=activation, 
                    bias=False)
                
            last_out_shape = int(torch.prod(torch.tensor(param.size())))

            # Apply custom initialization based on the parameter type
            if 'weight' in name:
                hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            # elif 'bias' in name or 'offsets' in name:
            #     hn.net[-1].apply(lambda m: hyper_bias_init(m))
            
            self.nets.append(hn)  # Add the FCBlock to the list

    def forward(self, input_hyp):
        """
        Forward pass of the hypernetwork.

        Args:
            input_hyp (torch.Tensor): Input tensor (embedding) of shape `(batch_size, hyper_in_features)`.

        Returns:
            OrderedDict: A dictionary where keys are parameter names from the hypo module and
                         values are the corresponding generated parameters, reshaped to match
                         their original shapes in the hypo module.
        """
        params = OrderedDict()

        # Generate each parameter using the corresponding FCBlock
        for i, (name, net, param_shape) in enumerate(zip(self.names, self.nets, self.param_shapes)):
            batch_param_shape = (-1,) + param_shape  # Add batch dimension
            
            if i != 0:
                input_hyp = last_out
            
            last_out = net(input_hyp)
            params[name] = last_out.reshape(batch_param_shape)
        return params

    def get_optimizer_list(self):
        """
        Creates a list of optimizers for the hypernetwork's parameters.

        Returns:
            list: A list of dictionaries, each containing parameter groups for optimization.
        """
        optimizer_list = [{'params': self.parameters(), 'lr': 1e-3}]
        return optimizer_list




############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
    """
    Initializes weights for a hypernetwork-generated weight matrix.

    Args:
        m (nn.Module): The module to initialize.
        in_features_main_net (int): Number of input features for the main network.
        siren (bool, optional): Indicates whether to use initialization tailored for SIREN models.
    
    Note:
        The weights are initialized using Kaiming normal initialization and scaled down.
    """
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1e1


def hyper_bias_init(m, siren=False):
    """
    Initializes biases for a hypernetwork-generated bias vector.

    Args:
        m (nn.Module): The module to initialize.
        siren (bool, optional): Indicates whether to use initialization tailored for SIREN models.
    
    Note:
        The biases are initialized using Kaiming normal initialization and scaled down.
    """
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e1



# import torch.nn as nn
# import torch.nn.functional as F

# from collections import OrderedDict
# from torch.nn.modules.batchnorm import _BatchNorm
# from torchmeta.modules.module import MetaModule

# class _MetaBatchNorm(_BatchNorm, MetaModule):
#     def forward(self, input, params=None):
#         self._check_input_dim(input)
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # exponential_average_factor is self.momentum set to
#         # (when it is available) only so that if gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum

#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum

#         weight = params.get('weight', None)
#         bias = params.get('bias', None)

#         return F.batch_norm(
#             input, self.running_mean, self.running_var, weight, bias,
#             self.training or not self.track_running_stats,
#             exponential_average_factor, self.eps)

# class MetaBatchNorm1d(_MetaBatchNorm):
#     __doc__ = nn.BatchNorm1d.__doc__

#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError('expected 2D or 3D input (got {}D input)'
#                              .format(input.dim()))

# class MetaBatchNorm2d(_MetaBatchNorm):
#     __doc__ = nn.BatchNorm2d.__doc__

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))

# class MetaBatchNorm3d(_MetaBatchNorm):
#     __doc__ = nn.BatchNorm3d.__doc__

#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'
#                              .format(input.dim()))