import argparse
import collections
import warnings

import numpy as np
import torch

import model as module_arch
import loss as module_loss
import logger as module_loggers
from trainer import Trainer
from utils.object_loading import get_dataloaders
from utils import prepare_device
from utils.parse_config import ConfigParser
from datasets.collate_fn import mel_transformer

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    # setup data_loader instances
    dataloaders = get_dataloaders(config["data"])

    # build model architecture, then print to console
    generator = config.init_obj(config["gen_arch"], module_arch)
    discriminator = config.init_obj(config["dis_arch"], module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler, logger
    gen_trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    gen_optimizer = config.init_obj(config["gen_optimizer"], torch.optim, gen_trainable_params)
    gen_scheduler = config.init_obj(config["gen_scheduler"], torch.optim.lr_scheduler, gen_optimizer)
    dis_trainable_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    dis_optimizer = config.init_obj(config["dis_optimizer"], torch.optim, dis_trainable_params)
    dis_scheduler = config.init_obj(config["dis_scheduler"], torch.optim.lr_scheduler, dis_optimizer)
    logger = config.init_obj(config["logger"], module_loggers, config)

    trainer = Trainer(
        generator,
        discriminator,
        loss_module,
        gen_optimizer,
        dis_optimizer,
        
        logger,
        config=config,
        device=device,
        dataloaders=dataloaders,
        mel_transformer=mel_transformer,
        gen_scheduler=gen_scheduler,
        dis_scheduler=dis_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    print('Start training')
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)