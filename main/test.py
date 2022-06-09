# Copyright (c) 2020 Graz University of Technology All rights reserved.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir+'/..')
import configargparse
from tqdm import tqdm
import torch
from common.base import Tester
import torch.backends.cudnn as cudnn


from common.utils.transforms import rot_param_rot_mat, rot_param_rot_mat_np
import smplx
from common.utils.vis import *
import gc




jointsMapManoToDefault = [
                         16, 15, 14, 13,
                         17, 3, 2, 1,
                         18, 6, 5, 4,
                         19, 12, 11, 10,
                         20, 9, 8, 7,
                         0]


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset', default='all', help='all/human_annot/machine_annot')
    parser.add_argument('--test_set', type=str, dest='test_set', default='test', help='Split type (test/train/val)')
    parser.add_argument('--ckpt_path', type=str, dest='ckpt_path', help='Full path to the checkpoint file')
    parser.add_argument('--use_big_decoder', action='store_true', help='Use Big Decoder for U-Net')
    parser.add_argument('--dec_layers', type=int, default=1, help='Number of Cross-attention layers')
    args = parser.parse_args()
    args.capture, args.camera, args.seq_name = None, None, None

    cfg.use_big_decoder = args.use_big_decoder
    cfg.dec_layers = args.dec_layers
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)


    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    # assert args.test_epoch, 'Test epoch is required.'
    return args

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]



def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, '')

    tester = Tester(args.ckpt_path)
    tester._make_batch_generator(args.test_set, args.annot_subset, args.capture, args.camera, args.seq_name)
    tester._make_model()

    preds = {'heatmaps':[], 'inputs':[], 'frame':[], 'joints':[],
             'hand_type':[], 'seq_id':[], 'cam':[], 'capture':[], 'rel_trans':[], 'inv_trans':[], 'joint_coord':[], 'obj_corners':[],
             'obj_rot': [], 'obj_trans': [], 'verts': [], 'abs_depth_left': [], 'abs_depth_right': []}
    gt = {'joints':[], 'joint_valid':[], 'hand_type':[], 'joint_coord':[], 'mano_joints':[], 'rel_trans':[], 'obj_corners_rest':[],
          'obj_rot': [], 'obj_trans': [], 'obj_id':[], 'hand_type_valid': [], 'princpt': [], 'focal': []}

    if cfg.predict_type == 'angles':
        mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                           'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
            mano_layer['left'].shapedirs[:, 0, :] *= -1

    ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
    ih26m_joint_regressor = torch.FloatTensor(ih26m_joint_regressor).unsqueeze(0)  # 1 x 21 x 778


    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):


            # forward
            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)

            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}

            hand_type_np = out['hand_type'].cpu().numpy() # N


            if cfg.predict_type == 'angles':
                start_indx = 0
                if cfg.hand_type in ['right', 'both']:
                    mesh_right = mano_layer['right'](global_orient=out['pose'][-1].permute(1, 0, 2)[:,0].to(torch.device('cpu')), # N x 32 x 3
                                        hand_pose=out['pose'][-1].permute(1, 0, 2)[:, 1:16].reshape(-1, 45).to(torch.device('cpu')),  # N x 32 x 3
                                        betas=out['shape'][-1].to(torch.device('cpu')), # N x 1 x 10
                                        )
                    joints_right = torch.matmul(ih26m_joint_regressor, mesh_right.vertices)# N x 21 x 3
                    start_indx += 16
                else:
                    joints_right = torch.zeros((inputs['img'].shape[0], 21, 3))
                mano_joints_right_gt = torch.matmul(ih26m_joint_regressor, targets['verts'][:,:778])

                if cfg.hand_type in ['left', 'both']:
                    mesh_left = mano_layer['left'](
                        global_orient=out['pose'][-1].permute(1, 0, 2)[:, start_indx].to(torch.device('cpu')),  # N x 32 x 3
                        hand_pose=out['pose'][-1].permute(1, 0, 2)[:, (start_indx+1):].reshape(-1, 45).to(torch.device('cpu')),  # N x 32 x 3
                        betas=out['shape'][-1].to(torch.device('cpu')),  # N x 10
                        # betas=targets['mano_shape'][:, 10:].to(torch.device('cpu')).to(torch.float32),
                        transl=out['rel_trans'].to(torch.device('cpu')))#*meta_info['root_valid']) # N x 3

                    joints_left = torch.matmul(ih26m_joint_regressor, mesh_left.vertices)  # N x 21 x 3
                else:
                    joints_left = torch.zeros((inputs['img'].shape[0], 21, 3))
                mano_joints_left_gt = torch.matmul(ih26m_joint_regressor, targets['verts'][:,778:])

            elif cfg.predict_type == 'vectors':
                if not cfg.predict_2p5d:
                    root_joint = torch.zeros((out['joint_3d_right'].shape[1],1,3)).to(out['joint_3d_right'].device)
                    if cfg.hand_type in ['right', 'both']:
                        joints_right = out['joint_3d_right'][-1]
                        joints_right = torch.cat([joints_right, root_joint], dim = 1)

                    if cfg.hand_type in ['left', 'both']:
                        joints_left = out['joint_3d_left'][-1]
                        joints_left = torch.cat([joints_left, root_joint], dim=1)
                else:
                    joints_2d_right = out['joint_2p5d'][:, :21]
                    joints_2d_left = out['joint_2p5d'][:, 21:]


            if cfg.has_object:
                pred_obj_corners_all = []
                for ii in range(out['obj_rot'].shape[0]):
                    if cfg.use_obj_rot_parameterization:
                        rot_mat = rot_param_rot_mat(out['obj_rot'][ii:ii + 1].reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
                    else:
                        rot_mat = cv2.Rodrigues(out['obj_rot'][ii].cpu().numpy())[0]
                    pred_obj_corners = meta_info['obj_bb_rest'][ii].cpu().numpy().dot(rot_mat.T) \
                                       + out['obj_trans'][ii].cpu().numpy()
                    pred_obj_corners_all.append(pred_obj_corners)
                pred_obj_corners_all = np.stack(pred_obj_corners_all, axis=0)



            # Save the outputs to lists
            preds['inv_trans'].append(out['inv_trans'].cpu().numpy())
            preds['rel_trans'].append(out['rel_trans'].detach().cpu().numpy())
            if cfg.predict_2p5d and cfg.predict_type=='vectors':
                preds['joint_coord'].append(out['joint_2p5d'].detach().cpu().numpy().astype(np.int32))
            else:
                preds['joints'].append(torch.cat([joints_right, joints_left], dim=1).detach().cpu().numpy()*1000)
            preds['hand_type'].append(hand_type_np)
            if cfg.predict_type == 'angles':
                if 'h2o3d' in cfg.dataset:
                    all_verts = np.concatenate([mesh_right.vertices.detach().cpu().numpy(), mesh_left.vertices.detach().cpu().numpy()], axis=2)
                    preds['verts'].append(all_verts)
                else:
                    preds['verts'].append(mesh_right.vertices.detach().cpu().numpy())
            else:
                if 'h2o3d' in cfg.dataset:
                    preds['verts'].append(np.zeros((inputs['img'].shape[0], 778, 6)))
                elif cfg.dataset == 'ho3d':
                    preds['verts'].append(np.zeros((inputs['img'].shape[0], 778, 3)))
                else:
                    preds['verts'].append(np.zeros((inputs['img'].shape[0], 1, 3)))

            preds['frame'].append(meta_info['frame'].cpu().numpy().astype(np.int32))
            preds['capture'].append(meta_info['capture'].cpu().numpy().astype(np.int32))
            preds['cam'].append(meta_info['cam'].cpu().numpy().astype(np.int32))
            preds['seq_id'] = preds['seq_id'] + meta_info['seq_id']

            if cfg.dataset == 'InterHand2.6M':
                preds['abs_depth_left'].append(meta_info['abs_depth_left'].cpu().numpy())
                preds['abs_depth_right'].append(meta_info['abs_depth_right'].cpu().numpy())
                gt['hand_type_valid'].append(meta_info['hand_type_valid'].detach().cpu().numpy())
                gt['focal'].append(meta_info['focal'].detach().cpu().numpy())
                gt['princpt'].append(meta_info['princpt'].detach().cpu().numpy())

            # Save the ground-truths to lists
            gt['joints'].append(targets['joint_cam_no_trans'].detach().cpu().numpy())
            if cfg.predict_type == 'angles':
                gt['mano_joints'].append(torch.cat([mano_joints_right_gt, mano_joints_left_gt], axis=1).detach().cpu().numpy()*1000)
            gt['joint_valid'].append(meta_info['joint_valid'].detach().cpu().numpy())
            gt['hand_type'].append(targets['hand_type'].detach().cpu().numpy())
            gt['rel_trans'].append(targets['rel_trans_hands_rTol'].detach().cpu().numpy())



            if cfg.has_object:
                for ii in range(joints_right.shape[0]):
                    pred_obj_corners = pred_obj_corners_all[ii]
                    if meta_info['obj_pose_valid'][ii].cpu().numpy() > 0:
                        pred_obj_trans = out['obj_trans'][ii].cpu().numpy()
                        if cfg.predict_type == 'angles':
                            # during training the object trans is relative to mano origin. during testing the mano origin is shifted to wrist joint.
                            pred_obj_trans = pred_obj_trans - joints_right[ii, 20].cpu().numpy() #
                        pred_obj_rot = cv2.Rodrigues(rot_param_rot_mat_np(out['obj_rot'][ii:ii+1].cpu().numpy())[0])[0].squeeze()

                        # Save the outputs to lists
                        preds['obj_corners'].append(np.expand_dims(pred_obj_corners, 0))
                        preds['obj_rot'].append(np.expand_dims(pred_obj_rot, 0))
                        preds['obj_trans'].append(np.expand_dims(pred_obj_trans, 0))

                        # Save the ground-truths to lists
                        gt['obj_corners_rest'].append(meta_info['obj_bb_rest'][ii:ii+1].cpu().numpy())
                        gt['obj_rot'].append(targets['obj_rot'][ii:ii+1].cpu().numpy())
                        gt['obj_trans'].append(targets['rel_obj_trans'][ii:ii+1].cpu().numpy())
                        gt['obj_id'].append(meta_info['obj_id'].cpu().numpy()[ii:ii+1])

            del model_out
            del targets
            del inputs
            del meta_info
            gc.collect()






    # evaluate
    preds_out = {k: np.concatenate(v,axis=0) for k,v in preds.items() if len(v)>0 and k!='seq_id'}
    preds_out['seq_id'] = preds['seq_id']
    gt_out = {}
    for k, v in gt.items():
        if len(v) > 0 and k not in ['seq_id']:
            gt_out[k] = np.concatenate(v)
        else:
            gt_out[k] = v


    tester._evaluate(preds_out,gt_out, args.ckpt_path, args.annot_subset)

if __name__ == "__main__":
    main()
