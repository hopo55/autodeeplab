import os
import pdb
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

from torchview import draw_graph
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF


def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    model_fname = args.checkname + '{0}_epoch%d.pth'.format(args.backbone)
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    elif args.dataset == 'sealer':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, test_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
        model_graph = draw_graph(model, graph_name='AutoDeepLab', input_size=(4,3,128,128), expand_nested=True, save_graph=True, filename='search_model', directory=args.checkname)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        train_iou = AverageMeter()

        for i, sample in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            scheduler(optimizer, cur_iter)
            # inputs = sample['image'].cuda()
            # target = sample['label'].cuda()
            inputs = sample[0].cuda()
            target = sample[1].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            target = target.unsqueeze(1)
            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, 'binary', threshold=0.5)
            iou_value = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            train_iou.update(iou_value)

            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})\t''mIoU: {train_iou.avg:.4f}'.format(epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses, train_iou=train_iou))
        
        # if epoch < args.epochs - 50:
        #     if epoch % 50 == 0:
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #         }, model_fname % (epoch + 1))
        if epoch == args.epochs - 1:
            model_fname = args.checkname + '{0}_last.pth'.format(args.backbone)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fname)

        # print('reset local total loss!')

    # Test
    iou = AverageMeter()
    latency = AverageMeter()

    model.eval()
    with torch.no_grad():
        # warm_up for latency calculation
        rand_img = torch.rand(4, 3, 128, 128).cuda()
        for _ in range(5):
            _ = model(rand_img)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        for i, sample in enumerate(test_loader):
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            starter.record()

            inputs = sample[0].cuda()
            target = sample[1].cuda()
            outputs = model(inputs)

            target = target.unsqueeze(1)
            target = target.long()
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, target, 'binary', threshold=0.5)
            iou_value = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            ori_image = inputs * 255.0
            output_image = outputs * 255.0

            ori_image[:, 0] = output_image[:, 0]

            for n, (ori_img, out_img) in enumerate(zip(ori_image, output_image)):
                index = i * (len(ori_image))
                ori_img = TF.to_pil_image(ori_img.squeeze().byte(), mode='RGB')
                ori_img.save("overlap/output_image" + str(index + n) + ".jpg")

                out_img = TF.to_pil_image(out_img.squeeze().byte(), mode='L')
                out_img.save("results/output_image" + str(index + n) + ".jpg")

            iou.update(iou_value)

            ender.record()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            latency_time = starter.elapsed_time(ender) / inputs.size(0)    # μs ()
            torch.cuda.empty_cache()

            latency.update(latency_time)
        
    latency_avg = latency.avg
    fps = 1000./latency_avg
    sec = latency_avg/1000.
    print("Test | IoU: %.4f, Latency: %.4f sec, FPS: %.4f\n" % (iou.avg, sec, fps))

if __name__ == "__main__":
    main()
