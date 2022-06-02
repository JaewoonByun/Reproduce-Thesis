import argparse
from ast import arg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import argparse
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_vit_args():
    # create the parser
    arg_parser = argparse.ArgumentParser()

    # hyper-parameter for debugging
    arg_parser.add_argument('--debug_mode', type=str2bool, default=False, help='in order to count "#" of model parameters')
    # use pre-trained model
    arg_parser.add_argument('--official_name', type=str, default='',
                            help='use official vit model such as \
                            "B_16", "B_32", "L_16", "L_32", "B_16_imagenet1k", "B_32_imagenet1k", "L_16_imagenet1k", "L_32_imagenet1k"')
    # hyper-parameter for leaning
    arg_parser.add_argument('--epoch', type=int, default=20)
    arg_parser.add_argument('--batch_size', type=int, default=100)
    arg_parser.add_argument('--lr', type=float, default=2e-3)
    arg_parser.add_argument('--lr_scheduler', type=str, default='cosine', help='"linear" or "cosine"')
    arg_parser.add_argument('--optim', type=str, default='adam', help='"adam" or "sgd"')
    arg_parser.add_argument('--w_d', type=float, default=0.1, help='(if use adam) weight decay')
    arg_parser.add_argument('--d_o', type=float, default=0.1, help='drop out')
    arg_parser.add_argument('--momentum', type=float, default=0.9, help='(if use sgd) momentum')
    # hyper-parameter for models
    arg_parser.add_argument('--n_cls', default=10, type=int, help='"#" of classes')
    arg_parser.add_argument('--img_size', default=384, type=int) # width == height
    arg_parser.add_argument('--patch_size', default=16, type=int)
    arg_parser.add_argument('--h_d', type=int, default=768, help='hidden dim')
    arg_parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp hidden dim ratio => h.d * mlp_ratio')
    arg_parser.add_argument('--n_layers', type=int, default=12)
    arg_parser.add_argument('--n_heads', type=int, default=12)
    # hyper-parameter for others
    arg_parser.add_argument('--gpu_mode', type=str2bool, default=True, help='force to use "cpu"')
    arg_parser.add_argument('--continue_train', type=str2bool, default=False, help='whether continue train the previous model or not')
    arg_parser.add_argument('--train_mode', type=str2bool, default=True)
    arg_parser.add_argument('--eval_mode', type=str2bool, default=True)

    args = arg_parser.parse_args()
    
    return args


def print_vit_args(args):
    if args is None:
        return
    
    print('=== VIT ===')
    print('+_ debugging params...')
    print(f'\t- debug mode: {args.debug_mode}',
          f'\t- official_name: {args.official_name}', sep='\n')
    print('+_ learning params...')
    print(f'\t- epoch: {args.epoch}',
          f'\t- batch_size: {args.batch_size}',
          f'\t- lr: {args.lr}',
          f'\t- lr_scheduler: {args.lr_scheduler}',
          f'\t- optim: {args.optim}',
          f'\t- weight decay: {args.w_d}',
          f'\t- dropout: {args.d_o}',
          f'\t- momentum: {args.momentum}', sep='\n')
    print('+_ model(ViT) params...')
    print(f'\t- n_cls: {args.n_cls}',
          f'\t- img_size: {args.img_size}',
          f'\t- patch_size: {args.patch_size}',
          f'\t- h_d: {args.h_d}',
          f'\t- mlp_ratio: {args.mlp_ratio}',
          f'\t- n_layers: {args.n_layers}',
          f'\t- n_heads: {args.n_heads}', sep='\n')
    print('+_ other params...')
    print(f'\t- gpu_mode: {args.gpu_mode}')
    print('===')