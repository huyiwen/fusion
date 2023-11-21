import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='MOSI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Tasks
parser.add_argument('--dataset', type=str, default='mosi')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--model_name', type=str, default='fusion', choices=['fusion', 'coordination'])

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when_decay', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=2023,
                    help='random seed')
parser.add_argument('--aligned', type=bool, default=True,
                    help='aligned')

####################################################################
#
# Setups
#
####################################################################

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
setup_seed(args.seed)
dataset = str.lower(args.dataset.strip())

use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    use_cuda = True

####################################################################
#
# Load the dataset
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

train_loader = DataLoaderX(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoaderX(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoaderX(test_data, batch_size=args.batch_size, shuffle=True)
print(next(iter(train_loader)))

print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.output_dim = 1
hyp_params.when_decay = args.when_decay
hyp_params.criterion = 'L1Loss'


if __name__ == '__main__':
    print(hyp_params)
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

