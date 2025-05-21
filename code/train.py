import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from model_framework import LGNJSDE, train
from mydataset import  load_dataset
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name', default='taobao')
parser.add_argument('-f', type=str, choices=['sde', 'lnsde'], default='sde',
                    help=' structure')
parser.add_argument('-z', type=int, default=32,
                    help='dim for hidden state of lambda')
parser.add_argument('-d', type=int, default=64,
                    help='dim for hidden layers')
parser.add_argument('-l', type=float, default=1e-3,
                    help='dim for hidden layer')
parser.add_argument('-e', type=int, default=300,
                    help='epochs')
parser.add_argument('-b', type=float, default=32,
                    help='batch size')
parser.add_argument('-s', default=2023, type=int, help='random seed')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    args = parser.parse_args()
    data = args.data
    h = args.z
    hidden_dim = args.d
    func_type = args.f
    seed = args.s
    lr = args.l
    max_epochs = args.e
    batch_size = args.b

    seed_everything(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, dev_dataset, train_num_types, test_time, test_type, num_test_event = load_dataset(data, device)
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    log = open('./logs/{}-{}-{}.txt'.format(data, func_type, seed), mode="a+", encoding="utf-8")

    model = LGNJSDE(num_vertex = train_num_types, h_dim = h, hidden_dim = hidden_dim, num_hidden = 2, act = nn.Tanh(),
                    device=device, func_type=func_type)

    train(model, max_epochs = max_epochs, train_dataloader = dl_train, val_dataloader = dl_val,
          seed = seed, data_name = data, log = log, lr = lr, impatience = 20)

    best_model = torch.load('./ckpt/{}-{}-{}-model.pth'.format(data, func_type, seed), map_location=device)

    best_model.eval()

    with torch.no_grad():
        test_loss, RMSE, acc, f1 = best_model.predict(test_time, test_type, h=8, n_samples=1000, device = device)

        print("Test Result s{} : nll: {:10.4f}, RMSE: {:10.4f}, acc: {:10.4f}, f1: {:10.4f}".format(seed,
                                                                                                test_loss / num_test_event,
                                                                                                RMSE, acc, f1))

        print("Test Result s{} : nll: {:10.4f}, RMSE: {:10.4f}, acc: {:10.4f}, f1: {:10.4f}".format(seed, test_loss / num_test_event,
                                                                              RMSE, acc, f1), flush=True, file=log)