import torch


def optimizer_opt(model, options):
    if options.optimizer == "sgd":
        optim = sgd_optimizer(model, options.lr, options.sgd_momentum, options.sgd_weight_decay)
    elif options.optimizer == "adam":
        optim = adam_optimizer(model, options.lr, options.adam_weight_decay)
    elif options.optimizer == "adamw":
        optim = adamw_optimizer(model, options.lr)

    else:
        raise ValueError("No such optimizer: {}".format(options.optimizer))

    return optim


def sgd_optimizer(model, lr, momentum, weight_decay):
    sgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return sgd


def adam_optimizer(model, lr, weight_decay):
    adam = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return adam


def adamw_optimizer(model, lr):
    adamw = torch.optim.AdamW(model.parameters(), lr=lr)
    return adamw
