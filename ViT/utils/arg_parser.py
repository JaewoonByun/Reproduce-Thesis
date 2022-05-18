import argparse

def get_vit_args():
    # create the parser
    arg_parser = argparse.ArgumentParser()

    # hyper-parameter for debugging
    arg_parser.add_argument('--debug_mode', type=bool, help='in order to count "#" of model parameters')
    arg_parser.add_argument('--official_name', type=str, 
                            help='use official vit model such as \
                            "B_16", "B_32", "L_16", "L_32", "B_16_imagenet1k", "B_32_imagenet1k", "L_16_imagenet1k", "L_32_imagenet1k"')
    # hyper-parameter for leaning
    arg_parser.add_argument('--epoch', type=int)
    arg_parser.add_argument('--batch_size', type=int)
    arg_parser.add_argument('--lr', type=float)
    arg_parser.add_argument('--lr_scheduler', type=str, help='"linear" or "cosine"')
    arg_parser.add_argument('--optim', type=str, help='"adam" or "sgd"')
    arg_parser.add_argument('--w_d', type=float, help='(if use adam) weight decay')
    arg_parser.add_argument('--d_o', type=float, help='drop out')
    arg_parser.add_argument('--momentum', type=float, help='(if use sgd) momentum')
    # hyper-parameter for models
    arg_parser.add_argument('--n_cls', type=int, help='"#" of classes')
    arg_parser.add_argument('--img_size', type=int) # width == height
    arg_parser.add_argument('--patch_size', type=int)
    arg_parser.add_argument('--h_d', type=int, help='hidden dim')
    arg_parser.add_argument('--mlp_ratio', type=int, help='mlp hidden dim ratio => h.d * mlp_ratio')
    arg_parser.add_argument('--n_layers', type=int)
    arg_parser.add_argument('--n_heads', type=int)
    # hyper-parameter for others
    arg_parser.add_argument('--gpu_mode', type=bool, help='force to use "cpu"')

    args = arg_parser.parse_args()

    # debugging params
    args.debug_mode = False if args.debug_mode is None else args.debug_mode
    args.official_name = '' if args.official_name is None else args.official_name

    # learning params
    args.epoch = 20 if args.epoch is None else args.epoch
    args.batch_size = 100 if args.batch_size is None else args.batch_size
    args.lr = 2e-3 if args.lr is None else args.lr
    args.lr_scheduler = 'consine' if args.lr_scheduler is None else args.lr_scheduler
    args.optim = 'adam' if args.optim is None else args.optim
    args.w_d = 0.1 if args.w_d is None else args.w_d
    args.d_o = 0.1 if args.d_o is None else args.d_o
    args.momentum = 0.9 if args.momentum is None else args.momentum

    # model params
    args.n_cls = 10 if args.n_cls is None else args.n_cls
    args.img_size = 32 if args.img_size is None else args.img_size
    args.patch_size = 16 if args.patch_size is None else args.patch_size
    args.h_d = 768 if args.h_d is None else args.h_d
    args.mlp_ratio = 4 if args.mlp_ratio is None else args.mlp_ratio
    args.n_layers = 12 if args.n_layers is None else args.n_layers
    args.n_heads = 12 if args.n_heads is None else args.n_heads

    # other params
    args.gpu_mode = True if args.gpu_mode is None else args.gpu_mode
    
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