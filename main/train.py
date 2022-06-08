# Copyright (c) 2020 Graz University of Technology All rights reserved.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')
import argparse
from main.config import cfg
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from common.base import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--run_dir_name', dest='run_dir_name', type=str, default='train', help='Name of the Run')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset', default='all', choices=['all', 'machine_annot', 'human_annot'],
                        help='annotation subset for InterHand2.6M dataset. Irrelavant for other datasets')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.run_dir_name, args.continue_train)
    cfg.create_run_dirs()
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cudnn.benchmark = True

    if cfg.dataset == 'InterHand2.6M':
        assert args.annot_subset, "Please set proper annotation subset. Select one of all, human_annot, machine_annot"
    else:
        args.annot_subset = None



    members = [attr for attr in dir(cfg) if not callable(getattr(cfg, attr)) and not attr.startswith("__")]
    cfg_dict = {}
    for m in members:
        cfg_dict[m] = cfg.__getattribute__(m)
    f = os.path.join(cfg.model_dir, 'cfg.txt')
    with open(f, 'w') as file:
        for arg in cfg_dict.keys():
            file.write('{} = {}\n'.format(arg, cfg_dict[arg]))

    f = os.path.join(cfg.model_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    trainer = Trainer()
    trainer._make_batch_generator(args.annot_subset)
    trainer._make_model()

    writer = SummaryWriter(cfg.tensorboard_dir)
    
    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):

        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()


            # forward
            trainer.optimizer.zero_grad()
            model_out = trainer.model(inputs, targets, meta_info, 'train', epoch)
            out = {k[:-4]:model_out[k] for k in model_out.keys() if '_out' in k}
            loss = {k: model_out[k] for k in model_out.keys() if '_out' not in k}
            loss = {k: loss[k].mean() for k in loss}
            joint_heatmap_out = out['joint_heatmap']


            loss['joint_heatmap'] *= cfg.hm_weight
            if cfg.has_object:
                loss['obj_seg'] *= cfg.obj_hm_weight

                loss['obj_rot'] *= cfg.obj_rot_weight
                loss['obj_trans'] *= cfg.obj_trans_weight
                loss['obj_corners'] *= cfg.obj_corner_weight
                loss['obj_corners_proj'] *= cfg.obj_corner_proj_weight
                loss['obj_weak_proj'] *= cfg.obj_weak_proj_weight


            if cfg.hand_type == 'both':
                loss['rel_trans'] *= cfg.rel_trans_weight
                loss['hand_type'] *= cfg.hand_type_weight

            if cfg.predict_type == 'angles':
                loss['pose'] *= cfg.pose_weight
                loss['shape_reg'] *= cfg.shape_reg_weight
                loss['vertex_loss'] *= cfg.vertex_weight
                loss['shape_loss'] *= cfg.shape_weight
                loss['joints_loss'] *= cfg.joint_weight
                loss['joints2d_loss'] *= cfg.joint_2d_weight
            elif cfg.predict_type == 'vectors':
                if cfg.predict_2p5d:
                    loss['joint_2p5d_hm'] *= cfg.joint_2p5d_weight
                else:
                    loss['joint_vec'] *= cfg.joint_vec_weight
                    loss['joints_loss'] *= cfg.joint_weight
                    loss['joints2d_loss'] *= cfg.joint_2d_weight

            loss['cls'] *= cfg.cls_weight

            if cfg.use_2D_loss and ((not cfg.predict_2p5d) or (cfg.predict_type=='angles')) :
                loss['cam_trans'] *= cfg.cam_trans_weight
                loss['cam_scale'] *= cfg.cam_scale_weight


            if itr % 20 == 0:
                # dump the outputs
                img_grid = torchvision.utils.make_grid(inputs['img'][:4])
                writer.add_image('input patches', img_grid, epoch * len(trainer.batch_generator) + itr)
                hm_grid = torchvision.utils.make_grid(joint_heatmap_out[:4].unsqueeze(1), normalize=True)

                if cfg.has_object:
                    seg_grid_gt = torchvision.utils.make_grid(out['obj_kps_gt'][:4].unsqueeze(1), normalize=True)
                    seg_grid_pred = torchvision.utils.make_grid(out['obj_seg_pred'][:4].unsqueeze(1), normalize=True)
                    writer.add_image('seg gt patches', seg_grid_gt, epoch * len(trainer.batch_generator) + itr)
                    writer.add_image('seg pred patches', seg_grid_pred, epoch * len(trainer.batch_generator) + itr)

                writer.add_image('heatmap', hm_grid, epoch * len(trainer.batch_generator) + itr)


                # dump the losses
                if cfg.predict_type == 'angles':
                    writer.add_scalar('pose loss', loss['pose'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('shape_reg loss', loss['shape_reg'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('vertex loss', loss['vertex_loss'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('shape', loss['shape_loss'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('joints loss', loss['joints_loss'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('joints2d_loss loss', loss['joints2d_loss'],
                                      epoch * len(trainer.batch_generator) + itr)

                elif cfg.predict_type == 'vectors':
                    if cfg.predict_2p5d:
                        writer.add_scalar('joint_2p5d_hm loss', loss['joint_2p5d_hm'],
                                          epoch * len(trainer.batch_generator) + itr)
                    else:
                        writer.add_scalar('joint_vec loss', loss['joint_vec'], epoch * len(trainer.batch_generator) + itr)
                        writer.add_scalar('joints loss', loss['joints_loss'], epoch * len(trainer.batch_generator) + itr)
                        writer.add_scalar('joints2d_loss loss', loss['joints2d_loss'],
                                          epoch * len(trainer.batch_generator) + itr)

                if cfg.hand_type == 'both':
                    writer.add_scalar('rel_trans loss', loss['rel_trans'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('hand_type loss', loss['hand_type'], epoch * len(trainer.batch_generator) + itr)

                if cfg.has_object:
                    writer.add_scalar('obj_seg loss', loss['obj_seg'], epoch * len(trainer.batch_generator) + itr)

                    writer.add_scalar('obj_rot loss', loss['obj_rot'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('obj_trans loss', loss['obj_trans'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('obj_corners loss', loss['obj_corners'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('obj_corners_proj loss', loss['obj_corners_proj'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('obj_weak_proj loss', loss['obj_weak_proj'], epoch * len(trainer.batch_generator) + itr)

                writer.add_scalar('cls loss', loss['cls'], epoch * len(trainer.batch_generator) + itr)

                writer.add_scalar('heatmap loss', loss['joint_heatmap'], epoch * len(trainer.batch_generator) + itr)

                writer.add_scalar('Learning rate', trainer.lr_scheduler.get_last_lr()[-1], epoch * len(trainer.batch_generator) + itr)
                writer.add_scalar('total loss', sum(loss[k] for k in loss), epoch * len(trainer.batch_generator) + itr)

                if cfg.use_2D_loss and ((not cfg.predict_2p5d) or (cfg.predict_type=='angles')) :
                    writer.add_scalar('cam trans', loss['cam_trans'], epoch * len(trainer.batch_generator) + itr)
                    writer.add_scalar('cam scale', loss['cam_scale'], epoch * len(trainer.batch_generator) + itr)



            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.lr_scheduler.get_last_lr()[-1]),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

            if (itr%1000 == 0) and (itr>0):
                # save model
                trainer.save_model({
                    'epoch': epoch,
                    'network': trainer.model.state_dict(),
                    'optimizer': trainer.optimizer.state_dict(),
                    'lr_scheduler': trainer.lr_scheduler.state_dict(),
                }, epoch, iter=itr)

        trainer.lr_scheduler.step()

        # save model
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'lr_scheduler': trainer.lr_scheduler.state_dict(),
        }, epoch, iter=itr)

        

if __name__ == "__main__":
    main()
