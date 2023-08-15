import datetime
import os
import time
import itertools
import torch
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn

from model import builder

import transforms as T
import utils

def is_distributed():
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    return distributed

def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset, ReferDatasetTest
    if image_set == 'val':
        ds = ReferDatasetTest(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    else:
        ds = ReferDataset(args,
                        split=image_set,
                        image_transforms=transform,
                        target_transforms=None
                        )
    num_classes = 2

    return ds, num_classes

def maybe_add_full_model_gradient_clipping(optim, args):  # optim: the optimizer class
    clip_norm_val = args.clip_value
    enable = args.clip_grads

    class FullModelGradientClippingOptimizer(optim):
        def step(self, closure=None):
            all_params = itertools.chain(*[x["params"] for x in self.param_groups])
            torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
            super().step(closure=closure)

    return FullModelGradientClippingOptimizer if enable else optim

def get_criterion(model):
    from criterion import criterion_dict
    return criterion_dict[model]

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)

def batch_IoU(pred, gt):
    intersection = torch.sum(torch.mul(pred, gt), dim=1)
    union = torch.sum(torch.add(pred, gt), dim=1) - intersection

    iou = intersection.float() / union.float()

    return iou, intersection, union

def batch_evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_num = len(data_loader.dataset)
    acc_ious = torch.zeros(1).cuda()

    # evaluation variables
    # cum_I, cum_U = 0, 0
    cum_I = torch.zeros(1).cuda()
    cum_U = torch.zeros(1).cuda()
    eval_seg_iou_list = [.5, .7, .9]
    seg_correct = torch.zeros(len(eval_seg_iou_list)).cuda()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            image, targets, sentences, attentions = data
            image, sentences, attentions = image.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)
            target = targets['mask'].cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            with torch.cuda.amp.autocast():
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = batch_IoU(output.flatten(1), target.flatten(1))
            acc_ious += iou.sum()
            cum_I += I.sum()
            cum_U += U.sum()
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou).sum()
        
    torch.cuda.synchronize()
    if is_distributed():
        cum_I = utils.all_reduce_tensor(cum_I, norm=False).cpu().numpy()
        cum_U = utils.all_reduce_tensor(cum_U, norm=False).cpu().numpy()
        acc_ious = utils.all_reduce_tensor(acc_ious, norm=False).cpu().numpy()
        seg_correct = utils.all_reduce_tensor(seg_correct, norm=False).cpu().numpy()
    else:
        cum_I = cum_I.cpu().numpy()
        cum_U = cum_U.cpu().numpy()
        acc_ious = acc_ious.cpu().numpy()
        seg_correct = seg_correct.cpu().numpy()

    mIoU = acc_ious / total_num
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / total_num)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * mIoU, 100 * cum_I / cum_U

def train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, print_freq, loss_scaler, clip_grad):
    model.train()
    optimizer.zero_grad()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4e}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):
        image, targets, sentences, attentions = data
        image, sentences, attentions = image.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        for k, v in targets.items():
            if isinstance(v, list):
                targets[k] = [m.cuda(non_blocking=True) for m in v]
            else:
                targets[k] = v.cuda(non_blocking=True)

        sentences = sentences.squeeze(1) # [B, N_l]
        attentions = attentions.squeeze(1) # [B, N_l]

        with torch.cuda.amp.autocast():
            loss_dict = model(image, sentences, l_mask=attentions, targets=targets)

        total_loss = loss_dict['total_loss']
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        lr_scheduler.step()

        torch.cuda.synchronize()
        metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(**loss_dict)

def main(args, distributed):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built val dataset.")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                        shuffle=True, drop_last=True)
        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False, drop_last=False)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True
    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, 
        drop_last=True, collate_fn=utils.collate_func)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers)

    # model initialization
    print(args.model)
    criterion = get_criterion(args.model)()
    single_model = builder.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args, criterion=criterion)
    single_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(single_model)
    print(single_model)
    single_model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(single_model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(single_model)
    # resume training
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])

    # optimizer
    params_to_optimize = single_model.params_to_optimize()
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )
    loss_scaler = utils.NativeScalerWithGradNormCount()
    clip_grad = args.clip_value if args.clip_grads else None

    total_iters = (len(data_loader) * args.epochs)
    lr_scheduler = utils.WarmUpPolyLRScheduler(optimizer, total_iters, power=0.9, min_lr=args.min_lr, 
        warmup=args.warmup, warmup_iters=args.warmup_iters, warmup_ratio=args.warmup_ratio)
    # housekeeping
    start_time = time.time()
    best_oIoU = -0.1

    # resume training (optimizer, lr scheduler, and the epoch)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        resume_epoch = checkpoint['epoch']
    else:
        resume_epoch = -999

    # training loops
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        if distributed:
            data_loader.sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, lr_scheduler, epoch, args.print_freq, loss_scaler, clip_grad)
        if epoch % 10 == 0 or epoch >= args.epochs - 16:
            iou, overallIoU = batch_evaluate(model, data_loader_test)

            print('Average object IoU {}'.format(iou))
            print('Overall IoU {}'.format(overallIoU))
            save_checkpoint = (best_oIoU < overallIoU)
            if save_checkpoint:
                print('Better epoch: {}\n'.format(epoch))
                dict_to_save = {'model': single_model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'args': args,
                                'lr_scheduler': lr_scheduler.state_dict(), 'scaler': loss_scaler.state_dict()}

                utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                                'model_best_{}.pth'.format(args.model_id)))
                best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    distributed = is_distributed()
    if distributed:
        utils.init_distributed_mode(args)
    cudnn.benchmark = True
    print('Image size: {}'.format(str(args.img_size)))
    main(args, distributed)
