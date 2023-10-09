# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from wenet.dataset.dataset_s3_emotion import Dataset
from wenet.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)
from wenet.utils.config import override_config
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.scheduler import WarmupLR
from wenet.utils.init_model import init_model
# from wenet.utils.executor_ssl import Executor
from wenet.utils.executor import Executor



def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='whether to use gpu')
    parser.add_argument("--local_rank", default=1, type=int)

    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument(
        "--non_lang_syms",
        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=1000,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument(
        "--enc_init_mods",
        default="encoder.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument("--partial_modules_path",
                        default=None,
                        type=str,
                        help="partition modules to init model")


    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')


    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(args.model_dir + '/log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.local_rank)
    # print(args.local_rank)

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # args.world_size = int(os.environ["WORLD_SIZE"])
    # args.rank = int(os.environ['RANK'])

    distributed = args.world_size > 1
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(
            args.rank))
        dist.init_process_group(args.dist_backend, init_method="env://")
        # dist.init_process_group(args.dist_backend,
        #                         init_method=args.init_method,
        #                         world_size=args.world_size,
        #                         rank=args.rank)

    symbol_table = read_symbol_table(args.symbol_table)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['shuffle'] = False

    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   persistent_workers=True,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    configs['vocab_size'] = vocab_size
    configs['blank_id']  = 0
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True

    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_model(configs)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    # if args.rank == 0:
        # script_model = torch.jit.script(model)
        # script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logging.info('load pretrained encoders: {}'.format(args.enc_init))
        infos = load_trained_modules(model, args)
    elif args.partial_modules_path is not None:
        logging.info('load partial module: {}'.format(
            args.partial_modules_path))
        infos = load_checkpoint(model, args.partial_modules_path)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import \
                default as comm_hooks
            model.register_comm_hook(state=None,
                                     hook=comm_hooks.fp16_compress_hook)
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
    # zjm add
    use_2_optimizer = True
    use_constant_lr = False
    if use_2_optimizer :
        if use_constant_lr:
            optimizer_1 = optim.Adam(model.encoder.encoders.parameters(),**configs['optim_conf_1'])
            optimizer_2 = optim.Adam(model.linear.parameters(),**configs['optim_conf_2'])

            from torch.optim.lr_scheduler import LambdaLR
            #scheduler_1 = WarmupLR(optimizer_1, **configs['scheduler_conf_1'])
            scheduler_1 = LambdaLR(optimizer_1, lr_lambda=lambda epoch:1.0)
            scheduler_2 = WarmupLR(optimizer_2, **configs['scheduler_conf_2'])
            #scheduler_2 = LambdaLR(optimizer_2, lr_lambda=lambda epoch:1.0)
            executor.step = step
            scheduler_2.set_step(step)
        else:
            optimizer_1 = optim.Adam(model.encoder.encoders.parameters(), **configs['optim_conf_1'])
            optimizer_2 = optim.Adam(model.linear.parameters(), **configs['optim_conf_2'])
            scheduler_1 = WarmupLR(optimizer_1, **configs['scheduler_conf_1'])
            scheduler_2 = WarmupLR(optimizer_2, **configs['scheduler_conf_2'])
            # Start training loop
            executor.step = step
            scheduler_1.set_step(step)
            scheduler_2.set_step(step)
    else:
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
        # Start training loop
        executor.step = step
        scheduler.set_step(step)

    final_epoch = None
    configs['rank'] = args.rank
    configs['is_distributed'] = distributed
    configs['use_amp'] = args.use_amp
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        if use_2_optimizer:
            lr_1 = optimizer_1.param_groups[0]['lr']
            lr_2 = optimizer_2.param_groups[0]['lr']
            logging.info('Epoch {} TRAIN info lr_1 {} ; lr_2 {}'.format(epoch, lr_1, lr_2))
            executor.train_with_2_optimizer(model, optimizer_1, optimizer_2, scheduler_1, scheduler_2, 
                                            train_data_loader, device,writer, configs, scaler, logger)
        else:
            lr = optimizer.param_groups[0]['lr']
            logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
            executor.train(model, optimizer, scheduler, train_data_loader, device,
                        writer, configs, scaler, logger)
        total_loss, num_seen_utts, acc = executor.cv(model, cv_data_loader, device,
                                                configs, logger)
        cv_loss = total_loss / num_seen_utts
        logging.info('Epoch {} CV info cv_loss {}, acc {}'.format(epoch, cv_loss, acc))
        if args.rank == 0:
            # save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_model_path = os.path.join(model_dir, '{}.pt'.format("latest"))
            lr=lr_2
            save_checkpoint(
                model, save_model_path, {
                    'epoch': epoch,
                    'lr': lr,
                    'cv_loss': cv_loss,
                    'step': executor.step
                })
            writer.add_scalar('epoch/test_loss', cv_loss, epoch)
            writer.add_scalar('epoch/lr_1', lr_1, epoch)
            writer.add_scalar('epoch/lr_2', lr_2, epoch)
            writer.add_scalar('epoch/acc', acc, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
