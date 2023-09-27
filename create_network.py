from network.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from network.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def create_network(conv_kernel_sizes,
                           UNet_class_name,
                           num_classes,
                           n_conv_per_stage_decoder,
                           n_conv_per_stage_encoder,
                           UNet_base_num_features,
                           unet_max_num_features,
                           pool_op_kernel_sizes,
                           dropout_p = None,
                           num_input_channels: int = 1,
                           deep_supervision: bool = True,
                           is_BN: bool = False,
                           is_inference: bool = False ):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(conv_kernel_sizes)
    norm = nn.BatchNorm3d if is_BN else nn.InstanceNorm3d
    dropout_op = None if dropout_p is None or dropout_p == 0 else nn.Dropout3d
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': nn.InstanceNorm3d,
            'norm_op': norm,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': dropout_op, 'dropout_op_kwargs': {'p': dropout_p,'inplace': True},
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            # 'norm_op': nn.InstanceNorm3d,
            'norm_op': norm,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': dropout_op, 'dropout_op_kwargs': {'p': dropout_p,'inplace': True},
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert UNet_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[UNet_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': n_conv_per_stage_decoder
    }
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                unet_max_num_features) for i in range(num_stages)],
        conv_op=nn.Conv3d,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[UNet_class_name],
        is_inference = is_inference

    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    return model
