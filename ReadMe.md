Use train.py 

```
python train.py
```

More details of hyper-parameters 

```
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
```

Dataset Available:

See the folder ./data

The result and model will be saved in ./logs and ./ckpt.