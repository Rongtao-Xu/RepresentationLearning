import argparse
import datetime
import logging
import os
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from datasets import voc
from utils.losses import DenseEnergyLoss, get_APML_loss
from network.PAR import PAR
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_to_label, cams_to_refine_label, cams_to_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_ref_mat,
                            propagte_ref_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)
from utils.optimizer import PolyWarmupAdamW
from network.RML_model import RML
from torch.nn.functional import kl_div



parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/coco_config.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")

parser.add_argument("--high_thre", default=0.55, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.35, type=float, help="low_bkg_score")

parser.add_argument('--backend', default='nccl')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename='test.log'):
    ## setup logger
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    # time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def validate(model=None, data_loader=None, cfg=None):
    preds, gts, cams, ref_gts, crf = [], [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, _, attn_pred = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = multi_scale_cam(model, inputs, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, cfg=cfg)

            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=args.radius)
            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)
            ref_cam = propagte_ref_cam_with_bkg(valid_cam_resized, ref=attn_pred, mask=infer_mask, cls_labels=cls_label,
                                                bkg_score=0.35)

            ref_cam = F.interpolate(ref_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            ref_label = ref_cam.argmax(dim=1)

            #preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            ref_gts += list(ref_label.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]

    cls_score = avg_meter.pop('cls_score')
    #seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    ref_score = evaluate.scores(gts, ref_gts)

    model.train()
    return cls_score, cam_score, ref_score


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    # _hw = (h + max(dilations)) * (w + max(dilations))
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def feat_feat_mi_estimation( F1, F2,dim):
    """
        F1: [B,48,96,72]
        F2: [B,48,96,72]
        F1 -> F2
    """
    batch_size = F1.shape[0]
    temperature = 0.05
    F1 = F1.reshape(batch_size, dim, -1).reshape(batch_size * dim, -1)
    F2 = F2.reshape(batch_size, dim, -1).reshape(batch_size * dim, -1)
    softmax = torch.nn.Softmax(dim=1)
    mi = kl_div(input=softmax(F1.detach() / temperature), target=softmax(F2 / temperature))

    return mi

def feat_label_mi_estimation( Feat, Y):
    """
        F: [B,1,h,w]
        Y: [B,1,h,w]
    """
    batch_size = Feat.shape[0]
    temperature = 0.05
    pred_Y = Feat  # B,48,h,w -> B,1,h,w
    pred_Y = pred_Y.reshape(batch_size, 1, -1).reshape(batch_size * 1, -1)
    Y = Y.reshape(batch_size, 1, -1).reshape(batch_size * 1, -1)
    softmax = torch.nn.Softmax(dim=1)
    mi = kl_div(input=softmax(pred_Y.detach() / temperature), target=softmax(Y / temperature),
                reduction='mean')  # pixel-level

    return mi


def train(cfg):
    num_workers = 10

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = coco.CocoClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    RML = RML(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling, )
    logging.info('\nNetwork config: \n%s' % (RML))
    param_groups = RML.get_param_groups()
    PAR = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    RML.to(device)
    PAR.to(device)

    mask_size = int(cfg.dataset.crop_size // 16)

    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)

    if args.local_rank == 0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        dummy_input = torch.rand(1, 3, 384, 384).cuda(0)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    RML = DistributedDataParallel(RML, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)

        cls, segs, attns, attn_pred = RML(inputs, seg_detach=args.seg_detach)

        cams, ref_mat = multi_scale_cam_with_ref_mat(RML, inputs=inputs, scales=cfg.cam.scales)
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                               cfg=cfg)

        scale_factor = 0.3 #random.uniform(0.2,0.7)
        #scale_factor = round(scale_factor,1)

        img2 = F.interpolate(inputs, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        cls2, segs2, attns2, attn_pred2 = RML(img2, seg_detach=args.seg_detach)
        cams2 = multi_scale_cam(RML, inputs=img2, scales=cfg.cam.scales)
        cams1 = F.interpolate(cams, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        CIML_loss_cam = torch.mean(torch.abs(cams1[:, 1:, :, :] - cams2[:, 1:, :, :]))

        cams1_max =  F.adaptive_avg_pool2d(cams1[:, 1:, :, :], output_size=1)

        cams2_max = F.adaptive_avg_pool2d(cams2[:, 1:, :, :], output_size=1)

        similarity = torch.cosine_similarity(cams1_max, cams1_max.squeeze(-1).unsqueeze(1), dim=3)

        similarity1 = torch.cosine_similarity(cams2_max, cams2_max.squeeze(-1).unsqueeze(1), dim=3)
        CIML_loss = 0.1*( similarity + similarity1).mean() + CIML_loss_cam


        segs = F.interpolate(segs, size=cams.shape[2:], mode='bilinear', align_corners=True)
        segs2 = F.interpolate(segs2, size=cams1.shape[2:], mode='bilinear', align_corners=True)


        segs1 = F.interpolate(segs, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        MFML_loss_fea = torch.mean(torch.abs(segs1[:, 1:, :, :] - segs2[:, 1:, :, :]))
        MFML_loss =  100*feat_feat_mi_estimation(segs1[:, 1:, :, :], segs2[:, 1:, :, :],14) + MFML_loss_fea



        refined_pseudo_label = refine_cams_with_bkg_v2(PAR, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg,
                                                       img_box=img_box)
        ref_label = cams_to_refine_label(refined_pseudo_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        APML_loss, pos_count, neg_count = get_APML_loss(attn_pred, ref_label)

        refined_ref_label = refined_pseudo_label

        attn_pred1 = F.interpolate(attn_pred.unsqueeze(1), size=refined_pseudo_label.shape[1:], mode='bilinear',align_corners=True)
        attn_pred2 = F.interpolate(attn_pred2.unsqueeze(1), size=refined_pseudo_label.shape[1:], mode='bilinear',align_corners=True)
        lossmi = feat_feat_mi_estimation(attn_pred1, attn_pred2,1)
        lossmil = feat_label_mi_estimation(attn_pred1, refined_pseudo_label.unsqueeze(1))

        lossmi2 = feat_feat_mi_estimation(attn_pred2, attn_pred1, 1)
        lossmil2 = feat_label_mi_estimation(attn_pred2, refined_pseudo_label.unsqueeze(1))

        APML_loss = APML_loss - 100 * (lossmil - lossmi) - 100 * (lossmil2 - lossmi2)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss  + 0.0 * MFML_loss + 0.0 * (APML_loss)
        else:
            loss = 1.0 * cls_loss  + 0.1 * (
            APML_loss) + 0.1 * MFML_loss + 0.1 * CIML_loss


        avg_meter.add({'cls_loss': cls_loss.item(), 'cam_loss': CIML_loss.item(), 'APML_loss': APML_loss.item(),
                       'corr_loss': MFML_loss.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs, dim=1, ).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
            ref_gts = refined_ref_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds == gts).sum() / preds.size

            grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)

            _attns_detach = [a.detach() for a in attns]
            _attns_detach.append(attn_pred.detach())

            grid_attns = imutils.tensorboard_attn2(attns=_attns_detach, n_row=cfg.train.samples_per_gpu)

            grid_labels = imutils.tensorboard_label(labels=gts)
            grid_preds = imutils.tensorboard_label(labels=preds)
            grid_refined_gt = imutils.tensorboard_label(labels=refined_gts)
            grid_ref_gt = imutils.tensorboard_label(labels=ref_gts)


            if args.local_rank == 0:
                logging.info(
                    "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, APML_loss: %.4f, corr_loss: %.4f, cam_loss %.4f, pseudo_seg_mAcc: %.4f" % (
                        n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('APML_loss'),
                        avg_meter.pop('corr_loss'), avg_meter.pop('cam_loss'), seg_mAcc))

                writer.add_image("RML/images", grid_imgs, global_step=n_iter)
                writer.add_image("RML/preds", grid_preds, global_step=n_iter)
                writer.add_image("RML/pseudo_gts", grid_labels, global_step=n_iter)
                writer.add_image("RML/pseudo_ref_gts", grid_refined_gt, global_step=n_iter)
                writer.add_image("RML/ref_gts", grid_ref_gt, global_step=n_iter)

                writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)

                writer.add_image("attns/top_stages_case0", grid_attns[0], global_step=n_iter)
                writer.add_image("attns/top_stages_case1", grid_attns[1], global_step=n_iter)
                writer.add_image("attns/top_stages_case2", grid_attns[2], global_step=n_iter)
                writer.add_image("attns/top_stages_case3", grid_attns[3], global_step=n_iter)

                writer.add_image("attns/last_stage_case0", grid_attns[4], global_step=n_iter)
                writer.add_image("attns/last_stage_case1", grid_attns[5], global_step=n_iter)
                writer.add_image("attns/last_stage_case2", grid_attns[6], global_step=n_iter)
                writer.add_image("attns/last_stage_case3", grid_attns[7], global_step=n_iter)

                writer.add_scalars('RML/loss', {"seg_loss": cls_loss.item(), "cls_loss": cls_loss.item()},
                                   global_step=n_iter)

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "RML_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                torch.save(RML.state_dict(), ckpt_name)
            cls_score, cam_score, ref_score = validate(model=RML, data_loader=val_loader, cfg=cfg)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (cls_score))
                logging.info("cams score:")
                logging.info(cam_score)
                logging.info("ref cams score:")
                logging.info(ref_score)


    return True





if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    cfg.cam.high_thre = args.high_thre
    cfg.cam.low_thre = args.low_thre

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    ## fix random seed
    setup_seed(1)
    train(cfg=cfg)
