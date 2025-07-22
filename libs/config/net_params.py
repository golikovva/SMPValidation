import sys

sys.path.insert(0, '../..')

from correction.config.cfg import cfg

from correction.models.trajGRU import TrajGRU
from correction.models.model import activation
from correction.models.convLSTM import ConvLSTM
from collections import OrderedDict

batch_size = cfg.run_config.batch_size

# in_size = 4 + 3*cfg.run_config.use_spatial_encoding + 2*cfg.run_config.use_time_encoding
in_size = cfg.model_args.DETrajGRU.encoder_cahnnels
rnn_activation = activation('leaky', negative_slope=0.2, inplace=True)
unet_params = {"n_channels": in_size,
               "n_classes": 3,
               "bilinear": True
               }

encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [in_size, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, (5, 6), (3, 4), 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 42, 56), zoneout=0.0, L=13,  # 96, 96
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,  # 32, 32
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=9,  # 16, 16
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=rnn_activation)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, (5, 6), (3, 4), 1]}),
        OrderedDict({
            # 'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'upsample_leaky_1': [64, 8, 4, 2, 1, 10],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 2, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=13,  # 16 16
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=rnn_activation),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,  # 32 32
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 42, 56), zoneout=0.0, L=9,  # 96 96
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation)
    ]
]


dencoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, (5, 6), (3, 4), 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 42, 56), zoneout=0.0, L=13,  # 96, 96
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,  # 32, 32
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=9,  # 16, 16
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=rnn_activation)
    ]
]

deforecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192*2, 192*2, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192*2, 64*2, (5, 6), (3, 4), 1]}),
        OrderedDict({
            # 'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'upsample_leaky_1': [64*2, 8*2, 4, 2, 1, 10],
            'conv3_leaky_2': [8*2, 8, 3, 1, 1],
            'conv3_3': [8, 2, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192*2, num_filter=192*2, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=13,  # 16 16
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=rnn_activation),

        TrajGRU(input_channel=192*2, num_filter=192*2, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,  # 32 32
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation),
        TrajGRU(input_channel=64*2, num_filter=64*2, b_h_w=(batch_size, 42, 56), zoneout=0.0, L=9,  # 96 96
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=rnn_activation)
    ]
]


# build model
conv2d_params = OrderedDict({
    # 'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv1_relu_1': [4, 64, 9, 7, 1],
    'conv2_relu_1': [64, 192, 7, 5, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 7, 5, 1],
    'deconv3_relu_1': [64, 64, 9, 7, 1],  # 7 5 1
    'conv3_relu_2': [64, 24, 3, 1, 1],
    # 'conv3_3': [20, 20, 1, 1, 0]
    'conv3_3': [24, 4, 1, 1, 0]
})

conv3d_params = OrderedDict({
    # 'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv1_relu_1': [4, 64, 9, 7, 1],
    'conv2_relu_1': [64, 192, 7, 5, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 7, 5, 1],
    'deconv3_relu_1': [64, 64, 9, 7, 1],  # 7 5 1
    'conv3_relu_2': [64, 24, 3, 1, 1],
    # 'conv3_3': [20, 20, 1, 1, 0]
    'conv3_3': [24, 4, 1, 1, 0]
})

# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [in_size, 8, 9, 7, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 7, 5, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 30, 40),  # 96 96
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 6, 8),  # 32 32
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 3, 4),  # 16 16
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 7, 5, 1]}),
        OrderedDict({
            # 'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'deconv3_leaky_1': [64, 8, 9, 7, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 3, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 3, 4),  # 16 16
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 6, 8),  # 32 32
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 30, 40),  # 96 96
                 kernel_size=3, stride=1, padding=1),
    ]
]
