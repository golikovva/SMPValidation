import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train the model on WRF and ERA fields')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--beta1', type=float, default=1.0, help='beta1 parameter for loss function')
    parser.add_argument('--beta2', type=float, default=10, help='beta2 parameter for loss function')
    parser.add_argument('--beta3_w10', type=float, default=0.06, help='beta3 wind speed parameter for loss function')
    parser.add_argument('--beta3_t2', type=float, default=0.03, help='beta3 2m temperature parameter for loss function')
    parser.add_argument('--beta4', type=float, default=0.85, help='beta4 parameter for loss function')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--workers', default=20, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='BERTunet',
                        help='Choose model architecture: trajGRU, convLSTM, conv2d or BERTunet')
    parser.add_argument('--scheduler', type=str, default='MultiStepLR',
                        help='Choose scheduler type: MultiStepLR, linearWarmUp')
    parser.add_argument('--use-era-data', '-er', default=1, type=int,
                        help='If use era5 data while training')
    parser.add_argument('--use-stations-data', '-st', default=1, type=int,
                        help='If use stations data while training')
    parser.add_argument('--use-scatter-data', '-sc', default=1, type=int,
                        help='If use scatterometer data while training')
    parser.add_argument('--use-spatiotemporal-encoding', '-sp', default=1, type=int,
                        help='If use spatiotemporal encoding')
    parser.add_argument('--use-time-encoding', '-t', default=1, type=int,
                        help='If use spatiotemporal encoding')
    parser.add_argument('--input-sequence-length', default=4, type=int, metavar='ISL',
                        help='input data sequence length')
    parser.add_argument('--output-sequence-length', default=4, type=int, metavar='zoSL',
                        help='output data sequence length')
    parser.add_argument('--run-mode', '-r', dest='run_mode', type=str, default='train',
                        help='run mode: train, test or train+test')
    parser.add_argument('--run-id', dest='run_id', type=int, default=4,
                        help='if run mode is test select run id')
    parser.add_argument('--best-epoch-id', dest='best_epoch_id', type=int, default=15,
                        help='if run mode is test select best epoch id')
    parser.add_argument('--draw-plots', type=int, default=1,
                        help='If draw resulting plots')
    parser.add_argument('--weighted-meaner', type=int, default=1,
                        help='If use weighted mean for loss function')
    parser.add_argument('--loss-kernel', type=str, default='gauss',
                        help='Choose loss kernel: mean, gauss')
    parser.add_argument('--running-env', type=str, default='home',
                        help='Specify where you run the script: docker, io, home')
    return parser.parse_args()
