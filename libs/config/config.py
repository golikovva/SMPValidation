from correction.helpers.ordered_easydict import OrderedEasyDict as edict
from correction.config.create_config import get_args
import numpy as np
import os
import torch


args = get_args()

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device is:', __C.GLOBAL.DEVICE)
#
if args.running_env == 'docker':
    print('Running script in Docker')
    __C.GLOBAL.BASE_DIR = '/home'
elif args.running_env == 'io':
    print('Running script on IO server')
    __C.GLOBAL.BASE_DIR = '/app/Precipitation-Nowcasting-master'
elif args.running_env == 'home':
    print('Running script on local machine')
    __C.GLOBAL.BASE_DIR = 'C:\\Users\\User\\ml\\WRFCorrection'
else:
    pass

__C.GLOBAL.MODEL_SAVE_DIR = os.path.join(__C.GLOBAL.BASE_DIR, 'logs')
assert __C.GLOBAL.MODEL_SAVE_DIR is not None


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

__C.HKO = edict()

__C.HKO.BENCHMARK = edict()

__C.HKO.BENCHMARK.IN_LEN = args.input_sequence_length
__C.HKO.BENCHMARK.OUT_LEN = args.input_sequence_length

__C.MODEL = edict()
from correction.models.model import activation
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)

# __C.run_config = edict()
# __C.run_config.epochs = args.epochs
# __C.run_config.batch_size = args.batch_size
# __C.run_config.beta1 = args.beta1
# __C.run_config.beta2 = args.beta2
# __C.run_config.beta3 = [args.beta3_w10, args.beta3_t2]
# __C.run_config.beta4 = args.beta4
# __C.run_config.lr = args.lr
# __C.run_config.workers = args.workers
# __C.run_config.model_type = args.model
# __C.run_config.use_era_data = bool(args.use_era_data)
# __C.run_config.use_stations_data = bool(args.use_stations_data)
# __C.run_config.use_scatter_data = bool(args.use_scatter_data)
# __C.run_config.use_spatiotemporal_encoding = bool(args.use_spatiotemporal_encoding)
# __C.run_config.use_time_encoding = bool(args.use_time_encoding)
# __C.run_config.run_mode = args.run_mode
# __C.run_config.run_id = args.run_id
# __C.run_config.best_epoch_id = args.best_epoch_id
# __C.run_config.draw_plots = bool(args.draw_plots)
# __C.run_config.weighted_meaner = bool(args.weighted_meaner)
# __C.run_config.loss_kernel = args.loss_kernel
# print(args.draw_plots, 'draw plots')
# print(args.use_spatiotemporal_encoding, 'use spt encoding')
# print(args.weighted_meaner, 'weighted meaner')
# print(args.loss_kernel, 'loss_kernel')
