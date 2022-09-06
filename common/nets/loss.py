# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import smplx
import open3d as o3d
from common.utils.misc import hungarian_match_2djoints, nearest_match_2djoints, batch_rodrigues
from common.utils.transforms import rot_param_rot_mat
from common.utils.preprocessing import load_skeleton

class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid, hm_valid):
        hm_valid = hm_valid[:, :, None] # N x 1 x 1
        loss = (joint_out - joint_gt)**2 * hm_valid
        return loss

class ObjSegLoss(nn.Module):
    def __ini__(self):
        super(ObjSegLoss, self).__init__()

    def forward(self, seg_pred, seg_gt):
        loss = (seg_pred - seg_gt)**2
        return loss

class CameraParamLoss(nn.Module):
    def __ini__(self):
        super(CameraParamLoss, self).__init__()

    def forward(self, joints_gt, joint_valid_in, joint_loc_2d_gt, cam_param_pred, rel_trans_valid, rel_trans_pred):
        '''

        :param joints_gt: N x 42 x 3
        :param joint_valid: N x 42
        :param joint_loc_2d_gt: N x 42 x 2
        :param cam_param_pred: 6 x N x 3
        :return:
        '''
        bs = joints_gt.shape[0]

        col1 = torch.zeros((42*2,), dtype=torch.float32).to(joints_gt.device)
        col1[0::2] = 1
        col1 = col1.unsqueeze(0).repeat(bs,1)
        col2 = torch.zeros((42 * 2,), dtype=torch.float32).to(joints_gt.device)
        col2[1::2] = 1
        col2 = col2.unsqueeze(0).repeat(bs, 1)
        joint_valid = joint_valid_in.detach().clone()

        # when its only left hand image, get the gt camera param based on predicted rel_trans
        joints_gt1 = joints_gt / 1000


        if cfg.predict_type == 'vectors' or True:
            # when its 2 hand image and the left or right root joint has no annotation
            joint_valid[:,21:] *= (rel_trans_valid)


        # Compute the orthographic projection matrix
        with torch.no_grad():
            b = joint_loc_2d_gt.reshape(bs, -1)  # N x 42*2
            b[:, ::2] *= joint_valid
            b[:, 1::2] *= joint_valid
            cam_param_gt_list = []
            for i in range(1):
                mat_A = torch.stack([joints_gt1[:,:,:2].reshape(bs, -1), col1, col2], dim=2) # N x 42*2 x 3

                mat_A[:, ::2] *= joint_valid.unsqueeze(2)
                mat_A[:, 1::2] *= joint_valid.unsqueeze(2)

                tmp = torch.inverse(torch.matmul(mat_A.transpose(2,1), mat_A)+torch.eye(3, device=mat_A.device, dtype=mat_A.dtype)*1e-4)
                cam_param_gt_curr = torch.matmul(tmp, torch.matmul(mat_A.transpose(2,1), b.unsqueeze(2))).squeeze(2).unsqueeze(0) # 1 x N x 3
                cam_param_gt_list.append(cam_param_gt_curr)

            cam_param_gt = torch.cat(cam_param_gt_list, dim=0) # 6 x N x 3
            cam_valid = torch.sum(joint_valid, dim=1) > 0 # N
            cam_valid = cam_valid.unsqueeze(0).unsqueeze(2) # 1 x N x 1


        # Loss terms
        loss_scale = torch.abs(cam_param_pred[:,:,:1] - cam_param_gt[:,:,:1])*cam_valid
        loss_trans = torch.abs(cam_param_pred[:, :, 1:] - cam_param_gt[:, :, 1:]) * cam_valid


        return loss_scale, loss_trans, cam_param_gt

class HandTypeLoss(nn.Module):
    def __init__(self):
        super(HandTypeLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        hand_type_out1 = hand_type_out[:,0]  # 2 x N x 3

        hand_type_gt_val = (hand_type_gt[:,0] + hand_type_gt[:,1]*2 - 1)*hand_type_valid # 0 - right, 1 - left, 2 - both
        hand_type_gt_val = hand_type_gt_val.unsqueeze(0).repeat(hand_type_out1.shape[0],1) # 2 x N
        loss = self.loss(hand_type_out1.reshape(-1, 3), hand_type_gt_val.view(-1).to(torch.int64)) # 2*N
        loss = loss.view(hand_type_out1.shape[0], hand_type_out1.shape[1]) * hand_type_valid.unsqueeze(0) # 2 x N

        return loss


class BottleNeckHandTypeLoss(nn.Module):
    def __init__(self):
        super(BottleNeckHandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss

class RelRootDepthLoss(nn.Module):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = torch.abs(root_depth_out - root_depth_gt) * root_valid
        return loss


class PoseLoss(nn.Module):
    '''
    MANO Joint angle loss
    '''
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt, mano_valid):
        pose_gt = pose_gt.view(-1, 32, 3)
        pose_out = pose_out.permute(0,2,1,3)# 6 x N x 32 x 3
        normalizer = 32#16 * mano_valid[0] + 16 * mano_valid[1]

        start_indx = 0
        if cfg.hand_type in ['right', 'both']:
            loss_right = (pose_out[:, :, :16]-pose_gt[:, :16].unsqueeze(0))*mano_valid[:, :1].unsqueeze(2).unsqueeze(0)
            loss_right = torch.sum(torch.sum(torch.abs(loss_right), 3), 2)
            loss = loss_right
            normalizer = 16
            start_indx += 16
        if cfg.hand_type in ['left', 'both']:
            loss_left = (pose_out[:, :, start_indx:] - pose_gt[:, 16:].unsqueeze(0)) * mano_valid[:, 1:].unsqueeze(2).unsqueeze(0)
            loss_left = torch.sum(torch.sum(torch.abs(loss_left), 3), 2)
            loss = loss + loss_left
            normalizer = normalizer + 16

        loss = loss/normalizer

        return loss

class RelTransLoss(nn.Module):
    '''
    Hands relative translation loss
    '''
    def __init__(self):
        super(RelTransLoss, self).__init__()

    def forward(self, rel_trans_out, rel_trans_gt, rel_trans_valid):
        loss = torch.abs(rel_trans_out - rel_trans_gt.unsqueeze(0))* rel_trans_valid

        return loss


class ShapeRegularize(nn.Module):
    def __init__(self):
        super(ShapeRegularize, self).__init__()

    def forward(self, shape_out):
        loss = torch.mean(shape_out**2, 2)

        return loss

class ShapeLoss(nn.Module):
    '''
    Loss on the MANO shape parameters
    '''
    def __init__(self):
        super(ShapeLoss, self).__init__()

    def forward(self, shape_gt, shape_pred, mano_valid):
        '''

        :param shape_gt: N x 20
        :param shape_pred: 6 x N x 10
        :param mano_valid: N x 2
        :return:
        '''

        shape_gt1 = (shape_gt[:,:10]*mano_valid[:,:1] + shape_gt[:,10:]*mano_valid[:,1:])/(torch.sum(mano_valid,1,keepdim=True)+1e-8)
        shape_valid = torch.sum(mano_valid, dim=1, keepdim=True).unsqueeze(0) > 0 # 1 x N x 1
        loss = torch.abs(shape_gt1.unsqueeze(0) - shape_pred)*shape_valid

        return loss

class ObjPoseLoss(nn.Module):
    '''
    Object Pose Loss
    '''
    def __init__(self):
        super(ObjPoseLoss, self).__init__()

    def forward(self, obj_rot_pred, obj_trans_pred, obj_trans_left_pred, hand_rel_trans_pred, obj_rot_gt, obj_trans_gt, hand_rel_trans_gt, obj_bb_rest,
                obj_pose_valid, obj_corner_pred, obj_corner_gt, obj_id, hand_rel_trans_valid, cam_param_pred, cam_param_gt):
        '''

        :param obj_rot_pred: 6 x N x 3
        :param obj_trans_pred: 6 x N x 3
        :param obj_rot_gt:  N x 3
        :param obj_trans_gt: N x 3
        :param obj_bb_rest: N x 8 x 3
        :param obj_corner_pred: 6 x N x 16
        :param obj_corner_gt: N x 8 x 3
        :param obj_pose_valid: N
        :param cam_param_gt: N x 3
        :param cam_param_pred: 6 x N x 3
        :return:
        '''


        # Get the object corners by flipping along the axis of symmetry

        obj_bb_gt = torch.matmul(obj_bb_rest, batch_rodrigues(obj_rot_gt).transpose(2, 1)) + obj_trans_gt.unsqueeze(1)  # N x 8 x 3

        rot_angle = np.pi
        # rotation about z
        rot_dir_z = batch_rodrigues(obj_rot_gt)[:, :3, 2] * rot_angle  # N x 3
        flipped_z_obj_rot = torch.matmul(batch_rodrigues(rot_dir_z), batch_rodrigues(obj_rot_gt))  # N x 3 x 3 # flipped rot
        obj_bb_flipped_z_gt = torch.matmul(obj_bb_rest, flipped_z_obj_rot.transpose(2, 1)) + obj_trans_gt.unsqueeze(1)  # N x 8 x 3

        # rotation about y
        rot_dir_y = batch_rodrigues(obj_rot_gt)[:, :3, 1] * rot_angle  # N x 3
        flipped_y_obj_rot = torch.matmul(batch_rodrigues(rot_dir_y), batch_rodrigues(obj_rot_gt))  # N x 3 x 3 # flipped rot
        obj_bb_flipped_y_gt = torch.matmul(obj_bb_rest, flipped_y_obj_rot.transpose(2, 1)) + obj_trans_gt.unsqueeze(1)  # N x 8 x 3

        # rotation about z and then y
        rot_dir_y = flipped_z_obj_rot[:, :3, 1] * rot_angle  # N x 3
        flipped_yz_obj_rot = torch.matmul(batch_rodrigues(rot_dir_y), flipped_z_obj_rot)  # N x 3 x 3 # flipped rot
        obj_bb_flipped_yz_gt = torch.matmul(obj_bb_rest, flipped_yz_obj_rot.transpose(2, 1)) + obj_trans_gt.unsqueeze(1)  # N x 8 x 3

        # get flipped z bb
        is_z_rot_sym_objs = torch.logical_or(obj_id==6, obj_id==21) # N # ['006_mustard_bottle', '021_bleach_cleanser']
        is_z_rot_sym_objs = torch.logical_or(is_z_rot_sym_objs, obj_id == 10)  # N # ['010_potted_meat_can']
        is_z_rot_sym_objs = torch.logical_or(is_z_rot_sym_objs, obj_id == 4)  # N # ['004_sugar_box']
        is_z_rot_sym_objs = torch.logical_or(is_z_rot_sym_objs, obj_id == 3)  # N # ['003_cracker_box']
        obj_bb_flipped_z_gt = obj_bb_flipped_z_gt*is_z_rot_sym_objs[:,None,None] + obj_bb_gt*(torch.logical_not(is_z_rot_sym_objs[:,None,None]))

        # get flipped y and yz bb
        is_yz_rot_sym_objs = torch.zeros_like(obj_id)*1.0
        obj_bb_flipped_y_gt = obj_bb_flipped_y_gt * is_yz_rot_sym_objs[:, None, None] + obj_bb_gt * (torch.logical_not(is_yz_rot_sym_objs[:, None, None]))
        obj_bb_flipped_yz_gt = obj_bb_flipped_yz_gt * is_yz_rot_sym_objs[:, None, None] + obj_bb_gt * (torch.logical_not(is_yz_rot_sym_objs[:, None, None]))

        # get flipped 2d bb
        obj_bb_2d_flipped_z_gt = obj_bb_flipped_z_gt[:, :, :2] * cam_param_gt[:, None, :1] + cam_param_gt[:, None, 1:]
        obj_bb_2d_flipped_y_gt = obj_bb_flipped_y_gt[:, :, :2] * cam_param_gt[:, None, :1] + cam_param_gt[:, None, 1:]
        obj_bb_2d_flipped_yz_gt = obj_bb_flipped_yz_gt[:, :, :2] * cam_param_gt[:, None, :1] + cam_param_gt[:, None, 1:]

        obj_bb_2d_gt = obj_bb_gt[:, :, :2] * cam_param_gt[:, None, :1] + cam_param_gt[:, None, 1:]  # N x 8 x 2

        # more weight for bowl
        is_bowl = obj_id==24 # o24_bowl
        is_bowl = torch.logical_or(is_bowl, obj_id==10)
        obj_pose_valid_bowl = obj_pose_valid*torch.logical_not(is_bowl) + 4*is_bowl


        # 2d corner prediction loss
        if obj_corner_pred is not None:
            obj_corner_pred1 = obj_corner_pred.reshape(obj_corner_pred.shape[0], obj_corner_pred.shape[1], 8, 2) # 6 x N x 8 x 2
            loss_obj_corner_pred = torch.abs(obj_corner_pred1 - obj_corner_gt[:,:,:2].unsqueeze(0))*obj_pose_valid_bowl.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            loss_obj_corner_pred = torch.zeros((1,)).to(obj_rot_pred.device)


        obj_bb_rest1 = obj_bb_rest.unsqueeze(0).repeat(obj_rot_pred.shape[0],1,1,1).reshape(-1,8,3)
        if cfg.use_obj_rot_parameterization:
            rot_mat = rot_param_rot_mat(obj_rot_pred.reshape(-1,6)) # 6*N x 3 x 3
            obj_bb_pred = torch.matmul(obj_bb_rest1, rot_mat.transpose(2, 1)) \
                          + obj_trans_pred.reshape(-1, 3).unsqueeze(1)  # 6*N x 8 x 3
            loss_obj_rot = torch.zeros((1,)).to(obj_rot_pred.device)
        else:
            obj_bb_pred = torch.matmul(obj_bb_rest1, batch_rodrigues(obj_rot_pred.reshape(-1,3)).transpose(2,1))\
                          + obj_trans_pred.reshape(-1,3).unsqueeze(1) # 6*N x 8 x 3
            loss_obj_rot = torch.abs(obj_rot_pred - obj_rot_gt.unsqueeze(0)) * obj_pose_valid_bowl.unsqueeze(0).unsqueeze(2)


        # 2d projection error
        obj_bb_2d_pred = obj_bb_pred[:,:,:2].reshape(cam_param_pred.shape[0], cam_param_pred.shape[1], 8, 2)*cam_param_pred[:,:,None,:1] + \
                         cam_param_pred[:, :, None, 1:] # 6 x N x 8 x 2
        loss_corners_2d_unflipped = torch.abs(obj_bb_2d_pred-obj_bb_2d_gt.unsqueeze(0)) # 6 x N x 8 x 2
        loss_corners_2d_unflipped = torch.mean(torch.mean(loss_corners_2d_unflipped, dim=3), dim=2)

        loss_corners_2d_flipped_z = torch.abs(obj_bb_2d_pred - obj_bb_2d_flipped_z_gt.unsqueeze(0))  # 6 x N x 8 x 2
        loss_corners_2d_flipped_z = torch.mean(torch.mean(loss_corners_2d_flipped_z, dim=3), dim=2) # 6 x N

        loss_corners_2d_flipped_y = torch.abs(obj_bb_2d_pred - obj_bb_2d_flipped_y_gt.unsqueeze(0))  # 6 x N x 8 x 2
        loss_corners_2d_flipped_y = torch.mean(torch.mean(loss_corners_2d_flipped_y, dim=3), dim=2)  # 6 x N

        loss_corners_2d_flipped_yz = torch.abs(obj_bb_2d_pred - obj_bb_2d_flipped_yz_gt.unsqueeze(0))  # 6 x N x 8 x 2
        loss_corners_2d_flipped_yz = torch.mean(torch.mean(loss_corners_2d_flipped_yz, dim=3), dim=2)  # 6 x N

        loss_corners_2d = torch.minimum(loss_corners_2d_unflipped, loss_corners_2d_flipped_z)
        loss_corners_2d = torch.minimum(loss_corners_2d, loss_corners_2d_flipped_y)
        loss_corners_2d = torch.minimum(loss_corners_2d, loss_corners_2d_flipped_yz)*obj_pose_valid_bowl.unsqueeze(0)


        # 3d corners error
        obj_pose_valid1 = obj_pose_valid_bowl.unsqueeze(1).unsqueeze(0).unsqueeze(2) # 1 x N x 1 x 1
        obj_bb_pred1 = obj_bb_pred.view(obj_rot_pred.shape[0], obj_rot_pred.shape[1], 8, 3) # 6 x N x 8 x 3

        loss_corners_unflipped = torch.abs(obj_bb_pred1 - obj_bb_gt.unsqueeze(0))*1000*obj_pose_valid1 # 6 x N x 8 x 3
        loss_corners_unflipped = torch.mean(torch.mean(loss_corners_unflipped, dim=3), dim=2)

        loss_corners_flipped_z = torch.abs(obj_bb_pred1 - obj_bb_flipped_z_gt.unsqueeze(0))*1000*obj_pose_valid1 # 6*N
        loss_corners_flipped_z = torch.mean(torch.mean(loss_corners_flipped_z, dim=3), dim=2)

        loss_corners_flipped_y = torch.abs(obj_bb_pred1 - obj_bb_flipped_y_gt.unsqueeze(0)) * 1000 * obj_pose_valid1  # 6*N
        loss_corners_flipped_y = torch.mean(torch.mean(loss_corners_flipped_y, dim=3), dim=2)

        loss_corners_flipped_yz = torch.abs(obj_bb_pred1 - obj_bb_flipped_yz_gt.unsqueeze(0)) * 1000 * obj_pose_valid1  # 6*N
        loss_corners_flipped_yz = torch.mean(torch.mean(loss_corners_flipped_yz, dim=3), dim=2)

        loss_corners = torch.minimum(loss_corners_flipped_z, loss_corners_unflipped)
        loss_corners = torch.minimum(loss_corners, loss_corners_flipped_y)
        loss_corners = torch.minimum(loss_corners, loss_corners_flipped_yz)

        # trans error wrt right hand
        loss_obj_trans = torch.abs(obj_trans_pred - obj_trans_gt.unsqueeze(0))*obj_pose_valid_bowl.unsqueeze(0).unsqueeze(2)

        # trans error wrt left hand
        if obj_trans_left_pred is not None:
            obj_trans_left_gt = obj_trans_gt - hand_rel_trans_gt
            loss_obj_trans_left = torch.abs(obj_trans_left_pred - obj_trans_left_gt.unsqueeze(0)) * obj_pose_valid_bowl.unsqueeze(
                0).unsqueeze(2)
            loss_obj_hand_tri = torch.abs(hand_rel_trans_pred+obj_trans_left_pred - obj_trans_pred)* obj_pose_valid_bowl.unsqueeze(
                0).unsqueeze(2)
            loss_obj_trans = loss_obj_trans + loss_obj_trans_left + loss_obj_hand_tri


        return loss_corners, loss_obj_rot, loss_obj_trans, loss_obj_corner_pred, loss_corners_2d




class ManoMesh():
    '''
    Helper Class for MANO mesh
    '''
    def __init__(self):

        self.mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                      'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}

        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1

        self.ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
        self.ih26m_joint_regressor = torch.FloatTensor(self.ih26m_joint_regressor).unsqueeze(0)  # 1 x 21 x 778

        self.finger_tip_vert_inds = [745, 320, 444, 555, 672]  # Thumb,  index, middle, ring, pinky
        self.jointsMapManoToDefault = [
                         16, 15, 14, 13,
                         17, 3, 2, 1,
                         18, 6, 5, 4,
                         19, 12, 11, 10,
                         20, 9, 8, 7,
                         0]

    def get_mano_mesh(self, pose, shape, rel_trans_in, rel_trans_valid, cam_param):
        pose = pose.permute(0, 2, 1, 3)  # 6 x N x 32 x 3(9)
        pose = pose.reshape(-1, pose.shape[2], pose.shape[3])  # 6N x 32 x 3(9)
        shape = shape.reshape(-1, 10)  # 6N x 10
        rel_trans = rel_trans_in * rel_trans_valid.unsqueeze(0)
        rel_trans = rel_trans.reshape(-1, 3)

        joints_pred = {}
        start_indx = 0

        if cfg.hand_type in ['right', 'both']:
            mesh_right = self.mano_layer['right'](global_orient=pose[:, 0, :3].to(torch.device('cpu')),
                                                  hand_pose=pose[:, 1:cfg.num_joint_queries_per_hand].reshape(pose.shape[0],
                                                                                                              -1).to(
                                                      torch.device('cpu')),
                                                  betas=shape.to(torch.device('cpu'))) # 6N x 778 x 3
            joints_pred['mesh_right'] = mesh_right.vertices # 6N x 778 x 3

            # these are the mano original joints in MANO order
            finger_tips = joints_pred['mesh_right'][:, self.finger_tip_vert_inds] # 6N x 5 x 3
            joints_pred['joints_mano_right'] = torch.cat([mesh_right.joints, finger_tips], dim=1) # 6N x 21 x 3
            joints_pred['joints_mano_right'] = joints_pred['joints_mano_right'].view(cfg.dec_layers, -1, 21,
                                                                                     3).to(pose.device)  # 6 x N x 21 x 3

            joints_pred['joints_right'] = torch.matmul(self.ih26m_joint_regressor, joints_pred['mesh_right']).view(cfg.dec_layers, -1, 21,
                                                                                     3).to(pose.device)  # 6 x N x 21 x 3
            joints_pred['mesh_right'] = joints_pred['mesh_right'].view(cfg.dec_layers, -1,
                                                                       joints_pred['mesh_right'].shape[1],
                                                                       3).to(pose.device)  # 6 x N x 778 x 3
            joints_pred['joints2d_right'] = joints_pred['joints_right'][:,:,:,:2]*cam_param[:,:,:1].unsqueeze(2) + cam_param[:,:,1:].unsqueeze(2)
            start_indx += cfg.num_joint_queries_per_hand

        else:
            joints_pred['mesh_right'] = None

        if cfg.hand_type in ['left', 'both']:
            mesh_left = self.mano_layer['left'](global_orient=pose[:, start_indx, :3].to(torch.device('cpu')),
                                                hand_pose=pose[:, (start_indx + 1):].reshape(pose.shape[0], -1).to(
                                                    torch.device('cpu')),
                                                betas=shape.to(torch.device('cpu')),
                                                transl=rel_trans.to(torch.device('cpu'))) # 6N x 778 x 3
            joints_pred['mesh_left'] = mesh_left.vertices  # 6N x 778 x 3

            # these are the mano original joints in MANO order
            finger_tips = joints_pred['mesh_left'][:, self.finger_tip_vert_inds]  # 6N x 5 x 3
            joints_pred['joints_mano_left'] = torch.cat([mesh_left.joints, finger_tips], dim=1)  # 6N x 21 x 3
            joints_pred['joints_mano_left'] = joints_pred['joints_mano_left'].view(cfg.dec_layers, -1, 21,
                                                                                     3).to(pose.device)  # 6 x N x 21 x 3

            joints_pred['joints_left'] = torch.matmul(self.ih26m_joint_regressor, joints_pred['mesh_left']).view(cfg.dec_layers, -1, 21,
                                                                                   3).to(pose.device)  # 6 x N x 21 x 3
            joints_pred['mesh_left'] = joints_pred['mesh_left'].view(cfg.dec_layers, -1,
                                                                     joints_pred['mesh_left'].shape[1],
                                                                     3).to(pose.device)  # 6 x N x 778 x 3
            joints_pred['joints2d_left'] = joints_pred['joints_left'][:, :, :, :2] * cam_param[:, :, :1].unsqueeze(
                2) + cam_param[:, :, 1:].unsqueeze(2)
        else:
            joints_pred['mesh_left'] = None

        return joints_pred

class Joints2p5dLoss(nn.Module):
    '''
    Loss for 2.5D pose. We regress 1D heatmaps instead of directly regressing the pose values and use L2 Loss on the heatmaps
    inspired by 'I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image',
    Gyeongsik Moon and Kyoung Mu Leem, ECCV'20
    '''
    def __init__(self):
        self.skeleton = load_skeleton(cfg.skeleton_file, 42)
        self.parent_inds = np.array([self.skeleton[i]['parent_id'] for i in range(42)])
        self.parent_inds = torch.from_numpy(self.parent_inds).to(torch.int64)
        self.joint_recon_order = [3, 2, 1, 0,
                                  7, 6, 5, 4,
                                  11, 10, 9, 8,
                                  15, 14, 13, 12,
                                  19, 18, 17, 16]
        super(Joints2p5dLoss, self).__init__()

    def get_1d_gaussian_heatmaps(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])
        x = x[None, None, None, :].cuda().float()
        hm_right = torch.exp(-((joint_coord[:, :21, :, None] - x) / cfg.sigma) ** 2 / 2)  # N x 21 x 3 x 128
        hm_left = torch.exp(-((joint_coord[:, 21:, :, None] - x) / cfg.sigma) ** 2 / 2)  # N x 21 x 3 x 128
        hm = torch.cat([hm_right, hm_left],dim=1) # N x 42 x 3 x 128
        return hm



    def forward(self, joints_2p5d_pred, joints_coord_gt, joints_valid):
        '''

        :param joints_2p5d_pred: 6 x 42 x N x 3 x 128
        :param joints_coord_gt: N x 42 x 3
        :param joints_valid: N x 42
        :return:
        '''
        bs = joints_coord_gt.shape[0]
        parent_inds = self.parent_inds.unsqueeze(1).unsqueeze(0).repeat(1, 1, 3).repeat(bs, 1, 1).to(joints_2p5d_pred.device)  # N x 42 x 3
        joint_valid_parent_right = torch.gather(joints_valid[:, :21], 1, parent_inds[:, :20, 0])
        parent_rel_valid_right = torch.logical_and(joint_valid_parent_right, joints_valid[:, :20]) # N x 20
        parent_rel_valid_right = torch.cat([parent_rel_valid_right,
                                            torch.zeros(parent_rel_valid_right.shape[0],1).to(parent_rel_valid_right.device)],dim=1) # N x 21
        joint_valid_parent_left = torch.gather(joints_valid[:, 21:], 1, parent_inds[:, 21:41, 0] - 21)  # N x 20
        parent_rel_valid_left = torch.logical_and(joint_valid_parent_left, joints_valid[:, 21:41])  # N x 20
        parent_rel_valid_left = torch.cat([parent_rel_valid_left,
                                            torch.zeros(parent_rel_valid_left.shape[0], 1).to(
                                                parent_rel_valid_left.device)], dim=1) # N x 21

        hm_gt = self.get_1d_gaussian_heatmaps(joints_coord_gt)

        loss = (joints_2p5d_pred.permute(0,2,1,3,4) - hm_gt.unsqueeze(0))**2 # 6 x N x 42 x 3 x 128
        loss[:, :, :21, :2] *= joints_valid[:, :21].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        loss[:, :, 21:, :2] *= joints_valid[:, 21:].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        loss[:, :, :21, 2:] *= parent_rel_valid_right[:, :21].unsqueeze(0).unsqueeze(3).unsqueeze(4)
        loss[:, :, 21:, 2:] *= parent_rel_valid_left[:, :21].unsqueeze(0).unsqueeze(3).unsqueeze(4)

        return loss

class JointVectorsLoss(nn.Module):
    '''
    Loss for 3D pose represetantion (parent-relative joint vectors)
    '''
    def __init__(self):
        super(JointVectorsLoss, self).__init__()
        self.skeleton = load_skeleton(cfg.skeleton_file, 42)
        self.parent_inds = np.array([self.skeleton[i]['parent_id'] for i in range(42)])
        self.parent_inds = torch.from_numpy(self.parent_inds).to(torch.int64)
        self.joint_recon_order = [3,2,1,0,
                                  7,6,5,4,
                                  11,10,9,8,
                                  15,14,13,12,
                                  19,18,17,16]

    def forward(self, joint_vec_preds_in, joints_gt_in, joint_valid_in, joint_loc_2d_gt, cam_param, rel_trans_pred):
        '''

        :param joint_vec_preds: 6 x N x 40 x 3
        :param joints_gt_in: N x 42 x 3
        :param joint_valid:  N x 42
        :param rel_trans_pred: 6 x N x 3
        :return:
        '''
        bs = joints_gt_in.shape[0]
        joints_gt = joints_gt_in.detach().clone()

        # get root relative pose
        joints_gt[:, :21] = joints_gt[:, :21] - joints_gt[:, 20:21]
        joints_gt[:, 21:] = joints_gt[:, 21:] - joints_gt[:, 41:]

        parent_inds = self.parent_inds.unsqueeze(1).unsqueeze(0).repeat(1,1,3).repeat(bs,1,1).to(joints_gt.device) # N x 42 x 3

        joint_vec_gt_right = joints_gt[:, :20] - torch.gather(joints_gt[:, :21], 1, parent_inds[:,:20]) # N x 20 x 3
        joint_vec_gt_left = joints_gt[:, 21:41] - torch.gather(joints_gt[:, 21:], 1, parent_inds[:,21:41]-21)  # N x 20 x 3
        joint_valid_parent_right = torch.gather(joint_valid_in[:, :21], 1, parent_inds[:,:20,0])
        joint_valid_parent_right = torch.logical_and(joint_valid_parent_right, joint_valid_in[:, :20])
        joint_valid_parent_left = torch.gather(joint_valid_in[:, 21:], 1, parent_inds[:, 21:41,0]-21) # N x 20
        joint_valid_parent_left = torch.logical_and(joint_valid_parent_left, joint_valid_in[:, 21:41]) # N x 20

        if cfg.hand_type == 'both':
            joint_vec_preds = joint_vec_preds_in.permute(0, 2, 1, 3)
            joint_vec_gt = torch.cat([joint_vec_gt_right, joint_vec_gt_left], dim=1)
            joint_vec_valid = torch.cat([joint_valid_parent_right, joint_valid_parent_left], dim=1)
        elif cfg.hand_type == 'right':
            joint_vec_preds = joint_vec_preds_in.permute(0, 2, 1, 3)[:,:,:20]
            joint_vec_gt = joint_vec_gt_right
            joint_vec_valid = joint_valid_parent_right
        elif cfg.hand_type == 'left':
            joint_vec_preds = joint_vec_preds_in.permute(0, 2, 1, 3)[:,:,:20]
            joint_vec_gt = joint_vec_gt_left
            joint_vec_valid = joint_valid_parent_left

        # Parent-relative joint vector loss
        loss_joint_vec = torch.abs(joint_vec_preds*1000 - joint_vec_gt.unsqueeze(0))*joint_vec_valid.unsqueeze(0).unsqueeze(3)


        # Reconstruct root-relative pose from parent-relative joint vectors
        joint_list_right = []
        joint_list_left = []
        for j in range(5):
            for i in range(4):
                if cfg.hand_type in ['both', 'right']:
                    if i == 0:
                        joint_3d_right = joint_vec_preds[:, :, self.joint_recon_order[j * 4 + i]] + 0.
                    else:
                        joint_3d_right = joint_vec_preds[:, :, self.joint_recon_order[j * 4 + i]] + joint_list_right[-1]
                    joint_list_right.append(joint_3d_right)
                if cfg.hand_type in ['both', 'left']:
                    offset = 20 if cfg.hand_type == 'both' else 0
                    if i == 0:
                        joint_3d_left = joint_vec_preds[:, :, self.joint_recon_order[j * 4 + i] + offset] + 0.
                    else:
                        joint_3d_left = joint_vec_preds[:, :, self.joint_recon_order[j * 4 + i] + offset] + joint_list_left[-1]
                    joint_list_left.append(joint_3d_left)
        joint_list_right = [tens.unsqueeze(2) for tens in joint_list_right]
        joint_list_left = [tens.unsqueeze(2) for tens in joint_list_left]



        joint_recon_order = torch.tensor(self.joint_recon_order).to(joints_gt.device)
        if cfg.hand_type == 'both':
            joint_3d_pred_right = torch.cat(joint_list_right, dim=2) # 6 x N x 20 x 3
            joint_3d_pred_left = torch.cat(joint_list_left, dim=2) # 6 x N x 20 x 3
            joint_3d_preds = torch.cat([joint_3d_pred_right, joint_3d_pred_left], dim=2) # 6 x N x 40 x 3

            joint_3d_gt_right = torch.index_select(joints_gt[:, :20], 1, (joint_recon_order))
            joint_3d_gt_left = torch.index_select(joints_gt[:, 21:41], 1, (joint_recon_order))
            joint_3d_gt = torch.cat([joint_3d_gt_right, joint_3d_gt_left], dim=1) # N x 40 x 3

            joint_3d_valid_right = torch.index_select(joint_valid_in[:, :20], 1, (joint_recon_order))
            joint_3d_valid_left = torch.index_select(joint_valid_in[:, 21:41], 1, (joint_recon_order))
            joint_3d_valid = torch.cat([joint_3d_valid_right,
                                        joint_3d_valid_left], dim=1)  # N x 40

            joint_2d_gt_right = torch.index_select(joint_loc_2d_gt[:, :20], 1, (joint_recon_order))
            joint_2d_gt_left = torch.index_select(joint_loc_2d_gt[:, 21:41], 1, (joint_recon_order))
            joint_2d_gt = torch.cat([joint_2d_gt_right, joint_2d_gt_left], dim=1)  # N x 40 x 2
        elif cfg.hand_type == 'right':
            joint_3d_preds = torch.cat(joint_list_right, dim=2)  # 6 x N x 20 x 3
            joint_3d_gt = torch.index_select(joints_gt[:, :20], 1, (joint_recon_order))
            joint_3d_valid = torch.index_select(joint_valid_in[:, :20], 1, (joint_recon_order))
            joint_2d_gt = torch.index_select(joint_loc_2d_gt[:, :20], 1, (joint_recon_order))
        elif cfg.hand_type == 'left':
            joint_3d_preds = torch.cat(joint_list_left, dim=2) # 6 x N x 20 x 3
            joint_3d_gt = torch.index_select(joints_gt[:, 21:41], 1, (joint_recon_order))
            joint_3d_valid = torch.index_select(joint_valid_in[:, 21:41], 1, (joint_recon_order))
            joint_2d_gt = torch.index_select(joint_loc_2d_gt[:, 21:41], 1, (joint_recon_order))


        # Loss on the root-relative poses
        loss_joint_3d = torch.abs(joint_3d_preds*1000 - joint_3d_gt.unsqueeze(0)) * joint_3d_valid.unsqueeze(0).unsqueeze(3)
        loss_joint_3d[:, :, :20] *= joint_valid_in[:, 20:21].unsqueeze(0).unsqueeze(3)
        loss_joint_3d[:, :, 20:] *= joint_valid_in[:, 41:42].unsqueeze(0).unsqueeze(3)


        zero_trans = torch.zeros((rel_trans_pred.shape[0], rel_trans_pred.shape[1], 20, 3)).to(rel_trans_pred.device)
        trans = torch.cat([zero_trans, rel_trans_pred.unsqueeze(2).repeat([1,1,20,1])],dim=2)


        # Loss on 2D poses after orthographic projection
        if cfg.use_2D_loss:
            joints_2d_pred = (joint_3d_preds+trans)[:, :, :, :2] * cam_param[:, :, :1].unsqueeze(2) + cam_param[:, :, 1:].unsqueeze(2)
            loss_joint_2d = torch.abs(joints_2d_pred  - joint_2d_gt.unsqueeze(0)) * joint_3d_valid.unsqueeze(0).unsqueeze(3)
        else:
            loss_joint_2d = torch.zeros((3,)).to(cam_param.device)

        if cfg.hand_type == 'both':
            joint_3d_preds_orig_order_right = torch.index_select(joint_3d_preds[:, :, :20], 2, (joint_recon_order))
            joint_3d_preds_orig_order_left = torch.index_select(joint_3d_preds[:, :, 20:], 2,
                                                                 (joint_recon_order))
        elif cfg.hand_type == 'right':
            joint_3d_preds_orig_order_right = torch.index_select(joint_3d_preds[:, :, :20], 2, (joint_recon_order))
            joint_3d_preds_orig_order_left = None
        elif cfg.hand_type == 'left':
            joint_3d_preds_orig_order_left = torch.index_select(joint_3d_preds[:, :, 20:], 2, (joint_recon_order))
            joint_3d_preds_orig_order_right = None

        return loss_joint_vec, loss_joint_3d, loss_joint_2d, joint_3d_preds_orig_order_right, joint_3d_preds_orig_order_left





class JointLoss(nn.Module):
    '''
    3D joint position loss. Used when outputing MANO joing angle representation.
    Also uses 2D loss after projecting to image using orthographic projection
    '''
    def __init__(self):


        self.ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
        self.ih26m_joint_regressor = torch.FloatTensor(self.ih26m_joint_regressor).unsqueeze(0) # 1 x 21 x 778

        super(JointLoss, self).__init__()

    def forward(self, rel_trans_in, joints_gt_in, joint_valid, joint_loc_2d_gt, rel_trans_valid, joints_pred, cam_param=None):

        bs = joints_gt_in.shape[0]

        # get only root-relative joint locations
        joints_right = joints_pred['joints_right'] - joints_pred['joints_right'][:,:,20:21]
        joints_left = joints_pred['joints_left'] - joints_pred['joints_left'][:,:,20:21]

        joints_gt_right = joints_gt_in[:,:21] - joints_gt_in[:,20:21]
        joints_gt_left = joints_gt_in[:, 21:] - joints_gt_in[:, 41:42]
        joints_gt = torch.cat([joints_gt_right, joints_gt_left], dim=1)

        if cfg.hand_type == 'both':
            joints_out = torch.cat([joints_right, joints_left], dim=2) # 6 x N x 42 x 3
            joints_gt1 = joints_gt.unsqueeze(0)
            joints2d_gt1 = joint_loc_2d_gt.unsqueeze(0)
            joint_valid1 = joint_valid.unsqueeze(0).unsqueeze(3)

            joints_2d_out = torch.cat([joints_pred['joints2d_right'], joints_pred['joints2d_left']], dim=2)
            # when its just left hand image, let the trans be as it is and let the 2d loss take care of it. There wont be any
            # direct loss on trans when its single left hand image
            # joints_left_for2d = joints_out[:,:,21:]# + (rel_trans_in.detach()*(rel_trans_valid.unsqueeze(0)==0)).unsqueeze(2)
            # joints_out_for2d = torch.cat([joints_out[:,:,:21], joints_left_for2d], dim=2)
        elif cfg.hand_type == 'right':
            joints_out = joints_right
            joints_gt1 = joints_gt[:,:21].unsqueeze(0)
            joints2d_gt1 = joint_loc_2d_gt[:,:21].unsqueeze(0)
            joint_valid1 = joint_valid[:,:21].unsqueeze(0).unsqueeze(3)
            joints_2d_out = joints_pred['joints2d_right']

        elif cfg.hand_type == 'left':
            joints_out = joints_left
            joints_gt1 = joints_gt[:, 21:].unsqueeze(0)
            joints2d_gt1 = joint_loc_2d_gt[:, 21:].unsqueeze(0)
            joint_valid1 = joint_valid[:, 21:].unsqueeze(0).unsqueeze(3)
            joints_2d_out = joints_pred['joints2d_left']
        else:
            raise NotImplementedError

        if cfg.use_2D_loss:
            # weak perspective projection
            # joints_2d_out = (joints_out_for2d[:,:,:,:2])*cam_param[:,:,:1].unsqueeze(2) + cam_param[:,:,1:].unsqueeze(2)
            loss_2d = torch.abs(joints_2d_out-joints2d_gt1) * joint_valid1
        else:
            loss_2d = torch.zeros((3,)).to(cam_param.device)


        loss = torch.abs(joints_out*1000 - joints_gt1) * joint_valid1 # 6 x N x 42 x 3

        return loss, loss_2d



class VertexLoss(nn.Module):
    '''
    MANO vertices error. Used when outputting MANO joint angle representation
    '''
    def __init__(self):

        super(VertexLoss, self).__init__()
        self.mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                           'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}

    def forward(self, verts_gt, mano_valid, joints_pred):
        # print(pose[-1].permute(1,0,2))
        bs = verts_gt.shape[0]
        mesh_right = joints_pred['mesh_right']
        mesh_left = joints_pred['mesh_left']


        # Hook for visualization
        if cfg.hand_type in ['right', 'both']:
            if False:
                # visualize mesh in open3d
                mesh_right_o3d = o3d.geometry.TriangleMesh()
                mesh_right_o3d.vertices = o3d.utility.Vector3dVector(mesh_right[-1,0].cpu().detach().numpy())
                mesh_right_o3d.triangles = o3d.utility.Vector3iVector(self.mano_layer['right'].faces)
                mesh_right_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy'))

                mesh_left_o3d = o3d.geometry.TriangleMesh()
                mesh_left_o3d.vertices = o3d.utility.Vector3dVector(verts_gt[0][:778].cpu().detach().numpy())
                mesh_left_o3d.triangles = o3d.utility.Vector3iVector(self.mano_layer['right'].faces)
                mesh_left_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy')[:,[2,1,0]])
                o3d.visualization.draw_geometries([mesh_right_o3d, mesh_left_o3d], mesh_show_back_face=True)

        if cfg.hand_type in ['left', 'both']:
            if False:
                # visualize mesh in open3d
                mesh_right_o3d = o3d.geometry.TriangleMesh()
                mesh_right_o3d.vertices = o3d.utility.Vector3dVector(mesh_left[-1,0].detach().cpu().numpy())
                mesh_right_o3d.triangles = o3d.utility.Vector3iVector(self.mano_layer['left'].faces)
                mesh_right_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy'))

                mesh_left_o3d = o3d.geometry.TriangleMesh()
                mesh_left_o3d.vertices = o3d.utility.Vector3dVector(verts_gt[0][778:].cpu().detach().numpy())
                mesh_left_o3d.triangles = o3d.utility.Vector3iVector(self.mano_layer['left'].faces)
                mesh_left_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy'))[:,[2,1,0]]
                o3d.visualization.draw_geometries([mesh_right_o3d, mesh_left_o3d], mesh_show_back_face=True)

        if cfg.hand_type == 'both':
            verts_out = torch.cat([mesh_right, mesh_left], dim=2) # 6 x N x 778*2 x 3
            verts_valid = torch.cat([torch.ones((bs, 778, 3), device=mesh_right.device) * mano_valid[:, :1].unsqueeze(2),
                                     torch.ones((bs, 778, 3), device=mesh_right.device) * mano_valid[:, 1:].unsqueeze(2)],
                                    dim=1).unsqueeze(0)  # 1 x N x 778*2 x 3
            verts_gt1 = verts_gt.unsqueeze(0)
        elif cfg.hand_type == 'right':
            verts_out = mesh_right
            verts_valid = torch.ones((bs, 778, 3), device=mesh_right.device) * mano_valid[:, :1].unsqueeze(2).unsqueeze(0)
            verts_gt1 = verts_gt.unsqueeze(0)[:,:,:778]
        elif cfg.hand_type == 'left':
            verts_out = mesh_left
            verts_valid = torch.ones((bs, 778, 3), device=mesh_right.device) * mano_valid[:, 1:].unsqueeze(2).unsqueeze(0)
            verts_gt1 = verts_gt.unsqueeze(0)[:, :, 778:]
        else:
            raise NotImplementedError


        loss = torch.abs((verts_out.to(mesh_right.device) - verts_gt1)*1000) * verts_valid # 6 x N x 778*2 x 3



        return loss


class JointClassificationLoss(nn.Module):
    '''
    Keypoint-Joint association Loss.
    '''
    def __init__(self):

        super(JointClassificationLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, joint_loc_pred_np, joint_loc_gt, mask, joint_class_pred, joint_valid, peak_joints_map_batch, joint_3d_gt,
                target_obj_kps_heatmap_np, obj_kps_coord_gt_np, obj_kps_3d_gt_np):
        '''

        :param joint_loc_pred_np: N x max_num_peaks x 2
        :param joint_loc_gt: N x 42 x 2
        :param mask: N x max_num_peaks
        :param joint_class_pred: 6 x N x max_num_peaks x (21+1)
        :param peak_joints_map_batch: N x max_num_peaks
        :return:
        '''
        min_dist_th = 2
        bs = joint_loc_pred_np.shape[0]

        if cfg.hand_type == 'right':
            joint_loc_gt1 = joint_loc_gt[:, :21]
            joint_valid1 = joint_valid[:,:21]
        elif cfg.hand_type == 'left':
            joint_loc_gt1 = joint_loc_gt[:, 21:]
            joint_valid1 = joint_valid[:, 21:]
        elif cfg.hand_type == 'both':
            joint_loc_gt1 = joint_loc_gt
            joint_valid1 = joint_valid
        joint_loc_gt_np = joint_loc_gt1.detach().cpu().numpy()
        joint_valid_np = joint_valid1.detach().cpu().numpy()
        peak_joints_map_batch_np = peak_joints_map_batch.detach().cpu().numpy()


        gt_inds_batch, row_inds_batch_list, asso_inds_batch_list \
            = nearest_match_2djoints(joint_loc_gt_np, joint_loc_pred_np, joint_valid_np, mask, joint_class_pred,
                                     joint_3d_gt.detach().cpu().numpy(), peak_joints_map_batch_np, target_obj_kps_heatmap_np,
                                     obj_kps_coord_gt_np, obj_kps_3d_gt_np)


        mask1 = torch.from_numpy(mask).to(joint_class_pred.device)  # N x max_num_peaks
        mask1 = mask1.unsqueeze(0).repeat([joint_class_pred.shape[0], 1, 1]) # 6 x N x max_num_peaks

        joint_class_pred1 = joint_class_pred.reshape(-1, joint_class_pred.shape[-1]).masked_select(mask1.reshape(-1).unsqueeze(1))
        joint_class_pred1 = joint_class_pred1.reshape(-1, joint_class_pred.shape[-1])

        ce_loss = self.loss(joint_class_pred1, torch.from_numpy(gt_inds_batch).to(joint_class_pred.device))

        return ce_loss, row_inds_batch_list, asso_inds_batch_list







