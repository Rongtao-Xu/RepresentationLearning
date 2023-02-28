import argparse
import os
import numpy as np
import os.path as osp

from misc import pyutils

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--mscoco_root", default='/data/coco/', type=str, help="Path to MSCOCO")

    parser.add_argument("--num_classes", default=80, type=int)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=8, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.1, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.15, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    # wavecam
    parser.add_argument("--wavecam_num_epoches", default=4, type=int)
    parser.add_argument("--wavecam_learning_rate", default=0.0005, type=float)
    parser.add_argument("--wavecam_loss_weight", default=0.1, type=float)
    parser.add_argument("--wavecam_batch_size", default=3, type=int)
    

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.35, type=float)
    parser.add_argument("--conf_bg_thres", default=0.1, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=16, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--ins_seg_bg_thres", default=0.25)
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--work_space", default="result_default5", type=str) # set your path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="ins_seg", type=str)
    parser.add_argument("--wavecam_weight_dir", default="wavecam_weight", type=str)

    # Step
    parser.add_argument("--train_cam_pass", type=str2bool, default=False)
    parser.add_argument("--train_wavecam_pass", type=str2bool, default=False)
    parser.add_argument("--make_cam_pass", type=str2bool, default=False)
    parser.add_argument("--make_wavecam_pass", type=str2bool, default=False)
    parser.add_argument("--eval_cam_pass", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label_pass", type=str2bool, default=False)
    parser.add_argument("--train_irn_pass", type=str2bool, default=False)
    parser.add_argument("--make_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--eval_ins_seg_pass", type=str2bool, default=False)
    parser.add_argument("--make_sem_seg_pass", type=str2bool, default=False) 
    parser.add_argument("--eval_sem_seg_pass", type=str2bool, default=False)

    args = parser.parse_args()
    args.log_name = osp.join(args.work_space,args.log_name)
    args.cam_weights_name = osp.join(args.work_space,args.cam_weights_name)
    args.irn_weights_name = osp.join(args.work_space,args.irn_weights_name)
    args.cam_out_dir = osp.join(args.work_space,args.cam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space,args.ir_label_out_dir)
    args.sem_seg_out_dir = osp.join(args.work_space,args.sem_seg_out_dir)
    args.ins_seg_out_dir = osp.join(args.work_space,args.ins_seg_out_dir)
    args.wavecam_weight_dir = osp.join(args.work_space,args.wavecam_weight_dir)

    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    os.makedirs(args.wavecam_weight_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))
    
    if args.train_cam_pass is True:
        import step_coco.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step_coco.train_cam.run(args)
    
    if args.train_wavecam_pass is True:
        import step_coco.train_wavecam

        timer = pyutils.Timer('step.train_wavecam:')
        step_coco.train_wavecam.run(args)
    
    if args.make_cam_pass is True:
        import step_coco.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step_coco.make_cam.run(args)
    
    if args.make_wavecam_pass is True:
        import step_coco.make_wavecam

        timer = pyutils.Timer('step.make_wavecam:')
        step_coco.make_wavecam.run(args)

    if args.eval_cam_pass is True:
        import step_coco.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step_coco.eval_cam.run(args)
    if args.cam_to_ir_label_pass is True:
        import step_coco.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step_coco.cam_to_ir_label.run(args)

    if args.train_irn_pass is True:
        import step_coco.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step_coco.train_irn.run(args)

    if args.make_sem_seg_pass is True:
        import step_coco.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step_coco.make_sem_seg_labels.run(args)

    if args.eval_sem_seg_pass is True:
        import step_coco.eval_sem_seg
        timer = pyutils.Timer('step.eval_sem_seg:')
        step_coco.eval_sem_seg.run(args)

