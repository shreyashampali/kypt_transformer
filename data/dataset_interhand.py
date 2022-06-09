# Copyright (c) 2020 Graz University of Technology All rights reserved.


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import torch
import torch.utils.data
import cv2
from glob import glob
import os.path as osp
from main.config import cfg
from common.utils.preprocessing import load_img, load_skeleton, get_bbox, process_bbox, augmentation, transform_input_to_output_space, trans_point2d
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, transformManoParamsToCam
from common.utils.vis import vis_keypoints, vis_3d_keypoints
from PIL import Image, ImageDraw
import random
import json
import math
from pycocotools.coco import COCO
import scipy.io as sio
import matplotlib.pyplot as plt
import smplx
import pyrender
import trimesh
import open3d as o3d
from tqdm import tqdm
import pickle
from common.utils.misc import get_root_rel_from_parent_rel_depths
ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
from common.utils.misc import my_print
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, annot_subset, capture_test=None, camera_test=None, seq_name_test=None):
        self.mode = mode # train, test, val
        self.annot_subset = annot_subset # all, human_annot, machine_annot
        self.img_path = cfg.interhand_images_path
        self.annot_path = cfg.interhand_anno_dir
        if self.annot_subset == 'machine_annot' and self.mode == 'val':
            self.rootnet_output_path = os.path.join(cfg.root_net_output_path, 'rootnet_interhand2.6m_output_machine_annot_val.json')
        else:
            self.rootnet_output_path = os.path.join(cfg.root_net_output_path, 'rootnet_interhand2.6m_output_all_test.json')
        self.transform = transform
        self.joint_num = 21 # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0,self.joint_num), 'left': np.arange(self.joint_num,self.joint_num*2)}
        self.skeleton = load_skeleton(cfg.skeleton_file, self.joint_num*2)
        
        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []
        
        # load annotation
        print("Load annotation from  " + osp.join(self.annot_path, self.annot_subset))
        db = COCO(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_data.json'))
        with open(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_camera.json')) as f:
            self.cameras = json.load(f)
        with open(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.annot_path, self.annot_subset, 'InterHand2.6M_' + self.mode + '_MANO.json')) as f:
            mano_params = json.load(f)


        if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
            assert os.path.exists(self.rootnet_output_path), 'Rootnet not available in %s'%self.rootnet_output_path
            print("Get bbox and root depth from " + self.rootnet_output_path)
            rootnet_result = {}
            with open(self.rootnet_output_path) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                rootnet_result[str(annot[i]['annot_id'])] = annot[i]
        else:
            print("Get bbox and root depth from groundtruth annotation")

        self.mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                           'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1
        
        for aid in tqdm(list(db.anns.keys())[::100]):
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
 
            capture_id = img['capture']
            seq_name = img['seq_name']
            cam = img['camera']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.img_path, self.mode, img['file_name'])

            if capture_test is not None:
                if int(capture_test) != int(capture_id):
                    continue
            if seq_name_test is not None:
                if seq_name_test != seq_name:
                    continue
            if camera_test is not None:
                if camera_test != cam:
                    continue


            if str(frame_idx) not in mano_params[str(capture_id)].keys():
                print('Frame %s in capture Id %s does not have MANO model (skipping)...'%(str(frame_idx), str(capture_id)))
                continue
            
            campos, camrot = np.array(self.cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(self.cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
            focal, princpt = np.array(self.cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(self.cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)
            joint_cam = world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0)
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            hand_type = ann['hand_type']
            joint_valid = np.array(ann['joint_valid'], dtype=np.float32).reshape(self.joint_num * 2)

            # get mano params in current cam
            mano_hand_type = ['right', 'left']
            mano_pose = []
            mano_trans = []
            mano_shape = []
            mano_valid = np.array([True, True]) # right, left valid
            for ii, ht in enumerate(mano_hand_type):
                mano_param = mano_params[str(capture_id)][str(frame_idx)][ht]
                if mano_param is not None and np.sum(joint_valid[self.joint_type[ht]]) > 0:
                    pose, trans, shape = transformManoParamsToCam(np.array(mano_param['pose']),
                                                                       np.array(mano_param['trans']),
                                                                       np.array(mano_param['shape']),
                                                                       cv2.Rodrigues(camrot.reshape(3,3))[0],
                                                                       -np.dot(camrot, campos.reshape(3, 1)).reshape(3), # -Rt -> t
                                                                       ht)
                    assert np.sum(joint_valid[self.joint_type[ht]]) > 0

                else:
                    mano_valid[ii] = False
                    pose = np.zeros((48,))*1.0
                    trans = np.zeros((3,))*1.0
                    shape = np.zeros((10,))*1.0

                mano_pose.append(pose)
                mano_trans.append(trans.squeeze())
                mano_shape.append(shape)
            mano_pose = np.concatenate(mano_pose, axis=0)
            mano_trans = np.concatenate(mano_trans, axis=0)
            mano_shape = np.concatenate(mano_shape, axis=0)




            # Exclude some images with annotations not suitable for training
            if True:
                # if root is not valid then root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
                if self.mode == 'test':
                    joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
                    joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]


                # if both hands not in the image, skip it
                if np.sum(mano_valid) == 0 and cfg.predict_type=='angles':
                    continue


                if cfg.hand_type == 'right':
                    if mano_valid[0] == 0 or mano_valid[1] == 1:
                        continue
                elif cfg.hand_type == 'left':
                    if mano_valid[1] == 0 or mano_valid[0] == 1:
                        continue
                else:
                    if not self.mode == 'test':
                        if hand_type == 'interacting' and cfg.predict_type=='angles':
                            if mano_valid[0] == 0:
                                if np.sum(joint_valid[self.joint_type['right']]) != 0 and np.sum(
                                        joint_valid[self.joint_type['left']]) != 0:
                                    # when interacting hands and mano_valid[0]==0, then relative translation is invalid.
                                    continue

            hand_type_valid = np.array((ann['hand_type_valid']), dtype=np.float32)
            

            if (self.mode == 'val' or self.mode == 'test') and cfg.trans_test == 'rootnet':
                bbox = np.array(rootnet_result[str(aid)]['bbox'],dtype=np.float32)
                abs_depth = {'right': rootnet_result[str(aid)]['abs_depth'][0], 'left': rootnet_result[str(aid)]['abs_depth'][1]}
            else:
                img_width, img_height = img['width'], img['height']
                bbox = np.array(ann['bbox'],dtype=np.float32) # x,y,w,h
                bbox = process_bbox(bbox, (img_height, img_width))
                abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam, 'img_coord': joint_img, 'valid': joint_valid}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param,
                    'bbox': bbox, 'joint': joint, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid,
                    'abs_depth': abs_depth, 'file_name': img['file_name'], 'capture': capture_id, 'cam': cam,
                    'frame': frame_idx, 'mano_pose': mano_pose, 'mano_trans': mano_trans, 'mano_shape': mano_shape,
                    'mano_valid': mano_valid}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_ih + self.datalist_sh
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def __len__(self):
        return len(self.datalist)

    def get_mano_vertices(self, pose, shape, transl, hand_type):
        mesh = self.mano_layer[hand_type](global_orient=torch.from_numpy(pose[:3]).float().unsqueeze(0),
                                              hand_pose=torch.from_numpy(pose[3:]).float().unsqueeze(0),
                                              betas=torch.from_numpy(shape).float().unsqueeze(0),
                                              transl=torch.from_numpy(transl).float().unsqueeze(0))

        return mesh.vertices[0].numpy(), self.mano_layer[hand_type].faces

    def restore_coord_cam_from_img(self, pred_joint_coord_img, inv_trans, data, do_flip):
        pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        for j in range(self.joint_num * 2):
            pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans)
        # restore depth to original camera space

        pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (
                    cfg.bbox_3d_size / 2)


        if cfg.dep_rel_to == 'parent':
            pred_joint_coord_img[:21, 2] = get_root_rel_from_parent_rel_depths(pred_joint_coord_img[:21,2])
            pred_joint_coord_img[21:, 2] = get_root_rel_from_parent_rel_depths(pred_joint_coord_img[21:, 2])

        # add root joint depth
        if do_flip:
            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['left']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['right']
        else:
            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']


        # back project to camera coordinate system
        pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, data['cam_param']['focal'], data['cam_param']['princpt'])

        return pred_joint_coord_cam



    
    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['hand_type'], data['hand_type_valid']
        mano_pose, mano_trans, mano_shape, mano_valid = data['mano_pose'].copy(), data['mano_trans'].copy(), data['mano_shape'].copy(), data['mano_valid'].copy()
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy();
        seq_name = data['seq_name']
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None].copy()),1)

        contact_vis_np = np.zeros((32, 2)).astype(np.float32)


        # image load
        img = load_img(img_path)

        # augmentation
        img, joint_coord, joint_valid, hand_type, inv_trans, inv_trans_no_rot, do_flip, mano_pose, mano_trans, joint_cam, _, _, _ =\
            augmentation(img, bbox, joint_coord, joint_valid, hand_type, self.mode, self.joint_type, mano_pose, mano_trans, mano_shape, joint_cam)

        if do_flip:
            mano_valid = mano_valid[[1, 0]]


        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)

        if cfg.predict_type == 'vectors':
            root_valid =  np.array([joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']]])*1.0
        elif cfg.predict_type == 'angles':
            root_valid = np.array([mano_valid[1]*mano_valid[0]], dtype=np.float32)*1.0

        # transform to output heatmap space
        joint_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_coord,
                                                                                               joint_valid,
                                                                                               rel_root_depth,
                                                                                               root_valid,
                                                                                               self.root_joint_idx,
                                                                                               self.joint_type, self.skeleton)
        img = self.transform(img.astype(np.float32)) / 255.




        if cfg.predict_type == 'angles':
            rel_trans_hands_rTol = mano_trans[3:] - mano_trans[:3]
            if root_valid and (np.sum(joint_valid[:21]) == 0):  # when its only left hand image
                # when its only left hand, shift it closer to right hand
                rel_trans_hands_rTol = np.array([0.2, 0., 0.]).astype(np.float32)

            joint_cam_no_trans = joint_cam.copy().astype(np.float32)
            joint_cam_no_trans[self.joint_type['right']] -= mano_trans[:3] * 1000  # because joints in mm
            joint_cam_no_trans[self.joint_type['left']] -= mano_trans[3:] * 1000
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol*1000


        elif cfg.predict_type == 'vectors':
            # when the root joints of left and right hand are not valid, joint_cam_no_trans are not correct. Joint_loss will be made
            # zero in this in the loss function.
            if self.mode == 'train':
                joint_cam_no_trans = self.restore_coord_cam_from_img(joint_coord[:42].copy(), inv_trans_no_rot, data, do_flip)
            else:
                joint_cam_no_trans = joint_cam.copy().astype(np.float32)

            rel_trans_hands_rTol = (joint_cam_no_trans[self.root_joint_idx['left']] - joint_cam_no_trans[
                self.root_joint_idx['right']]) / 1000


            joint_cam_no_trans[self.joint_type['right']] -= joint_cam_no_trans[self.root_joint_idx['right']]
            joint_cam_no_trans[self.joint_type['left']] -= joint_cam_no_trans[self.root_joint_idx['left']]
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol * 1000

        else:
            raise NotImplementedError

        interest_vert_coord = np.zeros((32,2,3)).astype(np.float32)
        interset_vert_valid = np.zeros((32,2)).astype(np.float32)


        # use zero mask for now. Later if required put ones along padded pixels
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))


        if cfg.predict_type == 'angles' and self.mode=='test':
            # vertices needed only in test mode to get the mano_joints, while training setting vertex loss to 0
            # get the mano vertices
            verts_right, faces_right = self.get_mano_vertices(mano_pose[:48], mano_shape[:10], np.zeros((3,)), 'right')
            verts_left, faces_left = self.get_mano_vertices(mano_pose[48:], mano_shape[10:], rel_trans_hands_rTol, 'left') # add the rel trans in loss.py
            verts = np.concatenate([verts_right, verts_left], axis=0)
        else:
            verts = np.zeros((1,3))



        # Some images are blank
        if np.sum(img.numpy()) == 0:
            mano_valid *= False
            joint_valid *= 0
            root_valid *= 0
            hand_type_valid *= 0
            contact_vis_np *= 0


        # Experimental Stuff
        contact_tri_inds_gt = (np.zeros((32, 2)) + cfg.num_faces_mano).astype(np.float32)
        contact_valid_gt = np.zeros((32, 2)).astype(np.float32)
        contact_verts_gt = np.zeros((32,2,3)).astype(np.float32)


        # heatmap valid flag. If its interacting and only one of the hand annotations is valid, set hm_valid=0
        if (np.sum(hand_type) == 2 and np.sum(joint_valid[:42])!=42) or ((hand_type[0]==1 and hand_type[1]==0) and np.sum(joint_valid[:21])!=21)\
                or  ((hand_type[0]==0 and hand_type[1]==1) and np.sum(joint_valid[21:])!=21):
            hm_valid = np.array([0.]).astype(np.float32)
        else:
            hm_valid = np.array([1.]).astype(np.float32)


        inputs = {'img': img, 'mask': mask}

        targets = {'joint_coord': joint_coord[:42], 'rel_trans_hands_rTol': rel_trans_hands_rTol, 'hand_type': hand_type,
                   'mano_pose':mano_pose, 'mano_shape': mano_shape, 'joint_cam_no_trans': joint_cam_no_trans, 'verts': verts,
                   'contact_tri_verts_inds': contact_verts_gt, 'contact_tri_inds': contact_tri_inds_gt, 'contact_vert_coord': interest_vert_coord,
                   'contact_vis': contact_vis_np}

        meta_info = {'joint_valid': joint_valid[:42], 'root_valid': root_valid, 'hand_type_valid': hand_type_valid,
                     'mano_valid': mano_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']),
                     'frame': int(data['frame']), 'seq_id': (seq_name[:9]), 'contact_valid': contact_valid_gt, 'contact_vert_valid':interset_vert_valid,
                     'hm_valid': hm_valid, 'abs_depth_left': data['abs_depth']['left'],
                     'abs_depth_right': data['abs_depth']['right'], 'focal': data['cam_param']['focal'],
                     'princpt': data['cam_param']['princpt']
                     }


        return inputs, targets, meta_info

    def dump_results(self, preds, dump_dir):
        for i, frame in enumerate(preds['frame']):
            hm = preds['heatmaps'][i]
            hm = (hm-np.min(hm))/(np.max(hm)-np.min(hm))
            hm = (hm*255).astype(np.uint8)
            hm = cv2.resize(hm, (256,256))
            cv2.imwrite(osp.join(dump_dir, 'patch_' + str(frame) + '.jpg'),
                        (np.transpose(preds['inputs'][i], [1, 2, 0])*255).astype(np.uint8)[:,:,[2,1,0]])
            cv2.imwrite(osp.join(dump_dir, 'hm_' + str(frame) + '.jpg'), hm)


            if False:
                # render the mesh
                mano_layer = preds['mano_mesh']
                cam_param = self.cameras[str(preds['capture'][i])]
                prev_depth = None
                for hand_type in ('right', 'left'):
                    # mesh
                    mesh = trimesh.Trimesh(mano_layer[hand_type].vertices.numpy(), mano_layer[hand_type].faces.numpy())
                    rot = trimesh.transformations.rotation_matrix(
                        np.radians(180), [1, 0, 0])
                    mesh.apply_transform(rot)
                    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
                    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
                    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
                    scene.add(mesh, 'mesh')

                    # add camera intrinsics
                    focal = np.array(cam_param['focal'][str(preds['cam'][i])], dtype=np.float32).reshape(2)
                    princpt = np.array(cam_param['princpt'][str(preds['cam'][i])], dtype=np.float32).reshape(2)
                    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
                    scene.add(camera)

                    # renderer
                    renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height, point_size=1.0)

                    # light
                    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
                    light_pose = np.eye(4)
                    light_pose[:3, 3] = np.array([0, -1, 1])
                    scene.add(light, pose=light_pose)
                    light_pose[:3, 3] = np.array([0, 1, 1])
                    scene.add(light, pose=light_pose)
                    light_pose[:3, 3] = np.array([1, 1, 2])
                    scene.add(light, pose=light_pose)

                    # render
                    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                    rgb = rgb[:, :, :3].astype(np.float32)
                    depth = depth[:, :, None]
                    valid_mask = (depth > 0)
                    if prev_depth is None:
                        render_mask = valid_mask
                        img = rgb * render_mask + img * (1 - render_mask)
                        prev_depth = depth
                    else:
                        render_mask = valid_mask * np.logical_or(depth < prev_depth, prev_depth == 0)
                        img = rgb * render_mask + img * (1 - render_mask)
                        prev_depth = depth * render_mask + prev_depth * (1 - render_mask)

    def print_results(self, mpjpe_sh, mpjpe_ih, mrrpe, f):
        if len(mrrpe) > 0: my_print('MRRPE: %f'%(sum(mrrpe) / len(mrrpe)), f)

        tot_err = []
        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            if len(mpjpe_sh[j])>0 and len(mpjpe_ih[j])>0:
                tot_err_j = np.mean(np.concatenate((np.stack(mpjpe_sh[j]), np.stack(mpjpe_ih[j]))))
            else:
                tot_err_j = np.nan
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % tot_err_j)
            tot_err.append(tot_err_j)
        my_print(eval_summary, f)
        my_print('MPJPE for all hand sequences: %.2f' % (np.nanmean(tot_err)), f)
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            if len(mpjpe_sh[j]) > 0:
                mpjpe_sh[j] = np.mean(np.stack(mpjpe_sh[j]))
            else:
                mpjpe_sh[j] = np.nan
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_sh[j])
        my_print(eval_summary, f)
        my_print('MPJPE for single hand sequences: %.2f' % (np.nanmean(mpjpe_sh)), f)
        print()

        eval_summary = 'MPJPE for each joint: \n'
        for j in range(self.joint_num * 2):
            if len(mpjpe_ih[j]) > 0:
                mpjpe_ih[j] = np.mean(np.stack(mpjpe_ih[j]))
            else:
                mpjpe_ih[j] = np.nan
            joint_name = self.skeleton[j]['name']
            eval_summary += (joint_name + ': %.2f, ' % mpjpe_ih[j])
        my_print(eval_summary, f)
        my_print('MPJPE for interacting hand sequences: %.2f' % (np.nanmean(mpjpe_ih)), f)

    def evaluate(self, preds, gt, ckpt_path, annot_subset):

        gt_joints, joint_valid_all, gt_hand_type_all = gt['joints'], gt['joint_valid'], gt['hand_type']
        mano_gt_joints, gt_rel_trans = gt['mano_joints'], gt['rel_trans']


        pred_joints,  pred_hand_type_all, pred_rel_trans = preds['joints'], preds['hand_type'], preds['rel_trans'],


        num_samples = pred_joints.shape[0]
        mpjpe_sh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_ih = [[] for _ in range(self.joint_num * 2)]
        mpjpe_mano_sh = [[] for _ in range(self.joint_num * 2)]
        mpjpe_mano_ih = [[] for _ in range(self.joint_num * 2)]
        frame_name_sh  = []
        frame_name_ih = []
        frame_name_mano_sh = []
        frame_name_mano_ih = []
        mrrpe = []

        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        ckpt_dir = os.path.dirname(ckpt_path)

        f_log = open(osp.join(ckpt_dir, '%s_%s_6layers.txt'%(ckpt_name, annot_subset)), 'w')

        my_print('num samples: %d'% num_samples, f_log)
        wrong_hand_type_pred_cnt = 0
        for i in range(num_samples):
            pred_joint_coord_cam = pred_joints[i]
            gt_joint_coord_cam = gt_joints[i]
            joint_valid = joint_valid_all[i]
            gt_hand_type = gt_hand_type_all[i]
            if cfg.predict_type == 'angles':
                mano_gt_joint_coord_cam = mano_gt_joints[i].copy()

            if np.sum(np.logical_xor(pred_hand_type_all[i]>0.5,gt_hand_type)) > 0:
                hand_type_wrong = True
                wrong_hand_type_pred_cnt += 1
            else:
                hand_type_wrong = False

            # mrrpe
            if (np.sum(gt_hand_type)==2) and joint_valid[self.root_joint_idx['left']] and joint_valid[
                self.root_joint_idx['right']]:
                gt_rel_root = gt_rel_trans[i]
                mrrpe.append(float(np.sqrt(np.sum((pred_rel_trans[i] - gt_rel_root) ** 2))))

            if cfg.hand_type == 'both':
                hands_list = ['right', 'left']
            else:
                hands_list = [cfg.hand_type]

            # root joint alignment
            for h in hands_list:
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[
                                                                                                      self.root_joint_idx[
                                                                                                          h], None, :]
                gt_joint_coord_cam[self.joint_type[h]] = gt_joint_coord_cam[self.joint_type[h]] - gt_joint_coord_cam[
                                                                                      self.root_joint_idx[h], None, :]

                if cfg.predict_type == 'angles':
                    mano_gt_joint_coord_cam[self.joint_type[h]] = mano_gt_joint_coord_cam[self.joint_type[h]] - mano_gt_joint_coord_cam[
                                                                                                      self.root_joint_idx[
                                                                                                          h], None, :]


            frame_name = str(preds['capture'][i])+'/'+preds['seq_id'][i]+'/'+str(preds['cam'][i])+'/'+str(preds['frame'][i])

            err = 0
            cnt = 0

            # mpjpe
            for j in range(self.joint_num * 2):
                if joint_valid[j]:
                    err += np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord_cam[j]) ** 2))
                    cnt += 1
                    if np.sum(gt_hand_type) == 1:
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord_cam[j]) ** 2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord_cam[j]) ** 2)))


            if np.sum(gt_hand_type) == 1:
                frame_name_sh.append([frame_name, hand_type_wrong, err/(cnt+1e-8)])
            else:
                frame_name_ih.append([frame_name, hand_type_wrong, err /(cnt+1e-8)])

            if cfg.predict_type == 'angles':
                err = 0
                cnt = 0
                # mpjpe
                for j in range(self.joint_num * 2):
                    if joint_valid[j]:
                        err += np.sqrt(np.sum((pred_joint_coord_cam[j] - mano_gt_joint_coord_cam[j]) ** 2))
                        cnt += 1
                        if np.sum(gt_hand_type) == 1:  # == 'right' or gt_hand_type == 'left':
                            mpjpe_mano_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - mano_gt_joint_coord_cam[j]) ** 2)))
                        else:
                            mpjpe_mano_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - mano_gt_joint_coord_cam[j]) ** 2)))


                if np.sum(gt_hand_type) == 1:
                    frame_name_mano_sh.append([frame_name, hand_type_wrong, err / (cnt + 1e-8)])
                else:
                    frame_name_mano_ih.append([frame_name, hand_type_wrong, err / (cnt + 1e-8)])

        with open(osp.join(cfg.model_dir, 'results_%s.pickle'%(ckpt_name)), 'wb') as f:
            pickle.dump({'sh':mpjpe_sh, 'ih':mpjpe_ih, 'fname_sh':frame_name_sh, 'fname_ih':frame_name_ih,
                         'sh_mano': mpjpe_mano_sh, 'ih_mano': mpjpe_mano_ih, 'fname_mano_sh': frame_name_mano_sh, 'fname_mano_ih': frame_name_mano_ih}, f)


        self.print_results(mpjpe_sh, mpjpe_ih, mrrpe, f_log)
        if cfg.predict_type == 'angles' and False:
            my_print('\n', f_log)
            my_print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', f_log)
            my_print('Below results used fitted MANO joints as ground-truths', f_log)
            my_print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', f_log)
            my_print('\n', f_log)
            self.print_results(mpjpe_mano_sh, mpjpe_mano_ih, mrrpe, f_log)


    def evaluate_2p5d(self, preds, gt, ckpt_path, annot_subset):

        print() 
        print('Evaluation start...')

        preds_joint_coord, preds_rel_trans, preds_hand_type, inv_trans = preds['joint_coord'], preds['rel_trans'], preds['hand_type'], preds['inv_trans']
        sample_num = len(preds_joint_coord)


        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        ckpt_dir = os.path.dirname(ckpt_path)
        f = open(osp.join(ckpt_dir, '%s_%s.txt' % (ckpt_name, annot_subset)), 'w')
        
        mpjpe_sh = [[] for _ in range(self.joint_num*2)]
        mpjpe_ih = [[] for _ in range(self.joint_num*2)]
        mrrpe = []
        frame_name_sh = []
        frame_name_ih = []
        acc_hand_cls = 0; hand_cls_cnt = 0;
        for n in range(sample_num):
            gt_joint_coord = gt['joints'][n]
            joint_valid = gt['joint_valid'][n]
            gt_hand_type = gt['hand_type'][n]

            if np.sum(gt_hand_type) == 2:
                gt_hand_type = 'interacting'
            elif gt_hand_type[0] == 1:
                gt_hand_type = 'right'
            elif gt_hand_type[0] == 0:
                gt_hand_type = 'left'

            hand_type_valid = gt['hand_type_valid'][n]

            data = {'abs_depth':{'left': preds['abs_depth_left'][n], 'right': preds['abs_depth_right'][n]},
                    'cam_param':{'princpt': gt['princpt'][n], 'focal':gt['focal'][n]}}

            preds_joint_coord_curr = preds_joint_coord[n].astype(np.float)
            pred_joint_coord_cam = self.restore_coord_cam_from_img(preds_joint_coord_curr.copy(), inv_trans[n], data, False)

            # mrrpe
            if gt_hand_type == 'interacting' and joint_valid[self.root_joint_idx['left']] and joint_valid[
                self.root_joint_idx['right']]:
                gt_rel_root = gt_joint_coord[self.root_joint_idx['left']] - gt_joint_coord[self.root_joint_idx['right']]
                mrrpe.append(float(np.sqrt(np.sum((preds['rel_trans'][n]*1000 - gt_rel_root) ** 2))))
            
            # root joint alignment
            for h in ('right', 'left'):
                pred_joint_coord_cam[self.joint_type[h]] = pred_joint_coord_cam[self.joint_type[h]] - pred_joint_coord_cam[self.root_joint_idx[h],None,:]
                gt_joint_coord[self.joint_type[h]] = gt_joint_coord[self.joint_type[h]] - gt_joint_coord[self.root_joint_idx[h],None,:]
            
            # mpjpe
            err = 0
            cnt = 0
            for j in range(self.joint_num*2):
                if joint_valid[j]:
                    err += np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j]) ** 2))
                    cnt += 1
                    if gt_hand_type == 'right' or gt_hand_type == 'left':
                        mpjpe_sh[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))
                    else:
                        mpjpe_ih[j].append(np.sqrt(np.sum((pred_joint_coord_cam[j] - gt_joint_coord[j])**2)))

            # handedness accuray
            hand_type_wrong = True
            if hand_type_valid:
                if gt_hand_type == 'right' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] < 0.5:
                    acc_hand_cls += 1
                    hand_type_wrong = False
                elif gt_hand_type == 'left' and preds_hand_type[n][0] < 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                    hand_type_wrong = False
                elif gt_hand_type == 'interacting' and preds_hand_type[n][0] > 0.5 and preds_hand_type[n][1] > 0.5:
                    acc_hand_cls += 1
                    hand_type_wrong = False
                hand_cls_cnt += 1

            if gt_hand_type == 'right' or gt_hand_type == 'left':
                frame_name_sh.append([n, hand_type_wrong, err/(cnt+1e-8)])
            else:
                frame_name_ih.append([n, hand_type_wrong, err /(cnt+1e-8)])

            vis = False
            if vis:
                filename = 'out_' + str(n) + '_3d.jpg'
                vis_3d_keypoints(pred_joint_coord_cam, joint_valid, self.skeleton, filename)

        with open(osp.join(ckpt_dir, 'results_%s.pickle'%(ckpt_name)), 'wb') as f1:
            pickle.dump({'sh': mpjpe_sh, 'ih': mpjpe_ih, 'fname_sh': frame_name_sh, 'fname_ih': frame_name_ih}, f1)
            
        # if hand_cls_cnt > 0: print('Handedness accuracy: ' + str(acc_hand_cls / hand_cls_cnt))

        self.print_results(mpjpe_sh, mpjpe_ih, mrrpe, f)
 



