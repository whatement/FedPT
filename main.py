from torchvision import transforms, datasets
from algorithm.fedpt import FedPT
from algorithm.fedavg import FedAvg
from algorithm.fednh import FedNH
from algorithm.fedetf import FedETF
from model.cnn import CNNCifar
from model.resnet import resnet18
from options import args_parser
from src.split import split_dataset
from src.tinyimagenet import TinyImageNet
from src.utils import seed_setup

def init_set(args):

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10('~/dataset/cifar10', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10('~/dataset/cifar10', train=False, download=True, transform=transform_test)

    elif args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                 (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        train = datasets.CIFAR100('~/dataset/cifar100', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR100('~/dataset/cifar100', train=False, download=True, transform=transform_test)
    
    elif args.dataset == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train = TinyImageNet('~/dataset/tiny-imagenet-200', train=True, transform=transform_train)
        test = TinyImageNet('~/dataset/tiny-imagenet-200', train=False, transform=transform_test)   

    else:
        raise ValueError("No such dataset: {}".format(args.dataset))

    if args.model == 'resnet18':
        m = resnet18(pretrained=False, num_classes=args.num_classes) 

    elif args.model == 'cnn_cifar':
        m = CNNCifar(num_classes=args.num_classes)

    else:
        raise ValueError("No such model: {}".format(args.model))

    return train, test, m


if __name__ == '__main__':
    args = args_parser()

    seed_setup(args.seed)

    train_dataset, test_dataset, model = init_set(args)

    train_user_groups, test_user_groups, data_distribution_lists = split_dataset(args, train_dataset, test_dataset)

    if args.algorithm == "fedavg":
        alg = FedAvg(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)
    
    elif args.algorithm == "fedpt":
        alg = FedPT(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)
    
    elif args.algorithm == "fednh":
        alg = FedNH(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedmoon":
    #     alg = FedMoon(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedproc":
    #     alg = FedProc(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedproto":
    #     alg = FedProto(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedprox":
    #     alg = FedProx(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedrep":
    #     alg = FedRep(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)
    
    # elif args.algorithm == "fedrod":
    #     alg = FedRod(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "fedetf":
    #     alg = FedETF(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)

    # elif args.algorithm == "local":
    #     alg = Local(train_dataset, test_dataset, train_user_groups, test_user_groups, data_distribution_lists, model, args)
    
    else:
        raise NotImplementedError

    alg.trainer()


    