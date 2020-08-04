import torch
import argparse
import sys
import random
import numpy as np
import wandb
import os
import math


def get_run_script():
    run_script = 'python'
    for e in sys.argv:
        run_script += (' ' + e)

    return run_script


def make_arg_parser():
    # converts string argument to boolean
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # arguments
    parser = argparse.ArgumentParser()

    # add arguments below
    base_args = parser.add_argument_group('Base args')
    base_args.add_argument('--run_script')
    base_args.add_argument('--device', type=str)
    base_args.add_argument('--gpu', type=str, default='7')
    base_args.add_argument('--debug_mode', type=str2bool, default=1)
    base_args.add_argument('--num_workers', type=int, default=4)
    base_args.add_argument('--weight_path', type=str, default='/shared/pebg_weights')
    base_args.add_argument('--train_dataset_path', type=str, default='/shared/vida/processed/question_tag_data_dic.pkl')
    base_args.add_argument('--machine_name', type=str)

    wandb_args = parser.add_argument_group('Wandb args')
    wandb_args.add_argument('--use_wandb', type=str2bool, default=1)
    wandb_args.add_argument('--project', type=str, default='pebg')
    wandb_args.add_argument('--name', type=str)
    wandb_args.add_argument('--tags', type=str)

    train_args = parser.add_argument_group('Train args')
    train_args.add_argument('--random_seed', type=int, default=1234)
    train_args.add_argument('--n_epochs', type=int, default=10000)
    train_args.add_argument('--train_batch_size', type=int, default=512)
    train_args.add_argument('--lr', type=float, default=0.001)
    train_args.add_argument('--emb_dim', type=int, default=256)
    train_args.add_argument('--product_layer_dim', type=int, default=128)
    train_args.add_argument('--dropout_rate', type=float, default=0.5)

    return parser


def get_args():
    parser = make_arg_parser()
    args = parser.parse_args()
    args.run_script = get_run_script()

    torch.set_printoptions(threshold=10000)

    # name
    args.name = f'e_dim:{args.emb_dim}_pl_dim:{args.product_layer_dim}_dr:{args.dropout_rate}_lr:{args.lr}'

    # parse tags
    args.tags = args.tags.split(',') if args.tags is not None else ['test']
    args.tags.append(args.name)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # debug
    if args.debug_mode:
        args.train_dataset_path = '/shared/vida/processed/debug_question_tag_data_dic.pkl'
        args.num_workers = 0

    # wandb
    if args.use_wandb:
        wandb.init(project=args.project, name=args.name, tags=args.tags, config=args)

    # parse gpus
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        assert torch.cuda.is_available()
        args.device = "cuda"
    else:
        args.device = "cpu"

    os.makedirs(args.weight_path, exist_ok=True)

    return parser, args


def print_args(parser, args):
    info = '\n[args]\n'
    for sub_args in parser._action_groups:
        if sub_args.title in ['positional arguments', 'optional arguments']:
            continue
        size_sub = len(sub_args._group_actions)
        info += f'  {sub_args.title} ({size_sub})\n'
        for i, arg in enumerate(sub_args._group_actions):
            prefix = '-'
            info += f'      {prefix} {arg.dest:20s}: {getattr(args, arg.dest)}\n'
    info += '\n'
    print(info)


parser, ARGS = get_args()


class Constants:
    PAD_IDX = 0
    MAX_NUM_TAGS_PER_ITEM = 6
    MAX_ELAPSED_TIME = 300
