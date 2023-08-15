import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RIS training and testing')
    # Dataset settings
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--img_size', default=448, type=int, help='input image size')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--refer_data_root', default=None, help='REFER dataset root directory')
    parser.add_argument('--refer_root', default=None, help='REFER annotations root directory')
    # General model settings
    parser.add_argument('--model', default=None, help='model: lavt, lavt_one')
    parser.add_argument('--model_id', default=None, help='name to identify the model')
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--pretrained_swin_weights', default='',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--clip_path', default='', help='path to pre-trained CLIP weights')
    # For training
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('--clip_grads', action='store_true',
                        help='if true, enable gradient clipping.')
    parser.add_argument('--clip_value', default=1.0, type=float, help='max norm of the gradients.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', default=0.00005, type=float, help='the initial learning rate')
    parser.add_argument('--min_lr', default=0, type=float, help='the minimal learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--warmup', action='store_true', help='if true, use warmup for training.')
    parser.add_argument('--warmup_iters', default=100, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    # For testing
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('--eval_ori_size', action='store_true',
                        help='evaluation with original image size')
    # Experment settings
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--output-dir', default=None, help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--mix', action='store_true',
                        help='if true, use refcoco/+/g mixed dataset for training.')
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
