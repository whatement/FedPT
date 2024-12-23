import argparse
import os



# 参数设置
def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    # General setup
    parser.add_argument('--gpu', type=str, default='cuda:0')
    # 2023 7 777
    parser.add_argument('--seed', type=int, default=2024)

    # Experimental setup
    parser.add_argument('--dataset', type=str, default="cifar10", help="cifar10, cifar100, tiny-imagenet")
    parser.add_argument('--model', type=str, default="cnn_cifar", help="cnn_cifar, resnet18")
    parser.add_argument('--algorithm', type=str, default="fedavg")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--radio', type=float, default=1)
    parser.add_argument('--test_rounds', type=int, default=200, help="200, 1")
    parser.add_argument('--test_all', type=bool, default=True)

    # Optimizer setup
    parser.add_argument('--optimizer', type=str, default="sgd", help='sgd or other optimizers')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--sgd_momentum', type=float, default=0.9)
    parser.add_argument('--sgd_weight_decay', type=float, default=1e-5)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-5)

    # Non-IID setup
    parser.add_argument('--partition', type=str, default="dirichlet",
                        help='homo, dirichlet')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3)
    parser.add_argument('--digit_class', type=int, default=2)
    parser.add_argument('--class_sample', type=int, default=200)

    # # wandb
    # parser.add_argument('--wandb_project', type=str, default="x")
    # parser.add_argument('--wandb_entity', type=str, default="x")

    # Fedpt
    parser.add_argument('--fedpt_rate', type=float, default=0.75)
    parser.add_argument('--fedpt_temp', type=float, default=0.1)

    # Fednh
    parser.add_argument('--fednh_s', type=float, default=30)

    args = parser.parse_args()

    return args