# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch
import torch.utils.data
from common.utils.preprocessing import *
from common.utils.transforms import world2cam, cam2pixel, pixel2cam, transformManoParamsToCam, convert_pose_to_opencv
import smplx
from tqdm import tqdm
from common.utils.misc import get_root_rel_from_parent_rel_depths
ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)


jointsMapManoToDefault = [
                         16, 15, 14, 13,
                         17, 3, 2, 1,
                         18, 6, 5, 4,
                         19, 12, 11, 10,
                         20, 9, 8, 7,
                         0]




class Dataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, annot_subset, capture=None, camera=None, seq_name_test=None):
        self.mode = mode  # train, test, val
        if mode == 'test':
            self.mode = 'evaluation' # train, test, val

        self.dataset_path_h2o3d = cfg.h2o3d_anno_dir
        self.dataset_path_ho3d = cfg.ho3d_anno_dir
        self.obj_kps_dir = cfg.obj_kps_dir

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
        with open(osp.join(self.dataset_path_h2o3d, self.mode+'.txt'), 'r') as f:
            self.filelist_h2o3d = f.readlines()
        with open(osp.join(self.dataset_path_ho3d, self.mode+'.txt'), 'r') as f:
            self.filelist_ho3d = f.readlines()
        self.filelist_h2o3d = [f.strip() for f in self.filelist_h2o3d]
        self.filelist_ho3d = [f.strip() for f in self.filelist_ho3d]

        self.dataset_name = ['h2o3d']*len(self.filelist_h2o3d) + ['ho3d']*len(self.filelist_ho3d)
        self.filelist = self.filelist_h2o3d + self.filelist_ho3d
        assert len(self.filelist) == len(self.dataset_name)
        
        self.mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
                           'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1


        for fid in tqdm(range(len(self.filelist))):
            fname = self.filelist[fid]
            ds_curr = self.dataset_name[fid]
            ds_path_curr = self.dataset_path_h2o3d if ds_curr == 'h2o3d' else self.dataset_path_ho3d

            seq_name = fname.split('/')[0]
            frame_idx = fname.split('/')[1]

            if ds_curr == 'h2o3d':
                img_path = osp.join(ds_path_curr, self.mode, seq_name, 'rgb', frame_idx + '.png')
                seg_path = osp.join(ds_path_curr, self.mode, seq_name, 'segr', frame_idx+'.png')
                obj_seg_channel = 1 # in rgb order
            else:
                img_path = osp.join(ds_path_curr, self.mode, seq_name, 'rgb', frame_idx + '.jpg')
                seg_path = osp.join(ds_path_curr, 'segr', self.mode, seq_name, 'seg', frame_idx+'.png')
                obj_seg_channel = 1  # in rgb order
            anno_path = osp.join(ds_path_curr, self.mode, seq_name, 'meta', frame_idx+'.pkl')

            if seq_name_test is not None:
                if seq_name_test != seq_name:
                    continue


            anno = load_pickle_data(anno_path)

            if ds_curr == 'ho3d':
                anno['rightHandJoints3D'] = anno['handJoints3D']
                anno['leftHandJoints3D'] = np.zeros_like(anno['handJoints3D'])
                anno['rightHandPose'] = anno['handPose']
                anno['rightHandTrans'] = anno['handTrans']
                hand_type = 'right'
                joint_valid = np.ones((self.joint_num * 2))
                joint_valid[21:] = 0.
                mano_valid = np.array([True, False])  # right, left valid
            else:
                hand_type = 'interacting'
                if 'jointValidRight' in anno.keys():
                    joint_valid = np.concatenate([anno['jointValidRight'], anno['jointValidLeft']], axis=0)
                    mano_valid = np.array([np.all(anno['poseValidRight']), np.all(anno['poseValidLeft'])])  # right, left valid
                else:
                    joint_valid = np.ones((self.joint_num * 2))
                    mano_valid = np.array([True, True])  # right, left valid


            focal, princpt = np.array([anno['camMat'][0,0], anno['camMat'][1,1]], dtype=np.float32), np.array([anno['camMat'][0,2], anno['camMat'][1,2]], dtype=np.float32)
            if self.mode == 'evaluation':
                anno_hand_joints_3d = np.concatenate([anno['rightHandJoints3D'][jointsMapManoToDefault],
                                                      anno['leftHandJoints3D'][jointsMapManoToDefault]])
                anno['handPose'] = np.zeros((48,)) * 1.0
                anno['handTrans'] = np.zeros((3,)) * 1.0
                anno['handBeta'] = np.zeros((10,)) * 1.0
            else:
                anno_hand_joints_3d = np.concatenate([anno['rightHandJoints3D'][jointsMapManoToDefault], anno['leftHandJoints3D'][jointsMapManoToDefault]])
            joint_cam = swap_coord_sys(anno_hand_joints_3d)*1000
            joint_img = cam2pixel(joint_cam, focal, princpt)[:,:2]

            obj_cam = swap_coord_sys(anno['objCorners3D'])*1000
            obj_img = cam2pixel(swap_coord_sys(anno['objCorners3D']), focal, princpt)[:,:2]


            obj_rot, obj_trans = convert_pose_to_opencv(anno['objRot'].squeeze(), anno['objTrans'])

            # get mano params in current cam
            mano_hand_type = ['right', 'left']
            mano_pose = []
            mano_trans = []
            mano_shape = []

            coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            for ii, ht in enumerate(mano_hand_type):
                if mano_valid[ii] == 1:
                    if ht == 'right':
                        pose, trans, _ = transformManoParamsToCam(anno['rightHandPose'].squeeze(), anno['rightHandTrans'].squeeze(),anno['handBeta'].squeeze(),
                                                               cv2.Rodrigues(coordChangMat)[0].squeeze(), np.zeros((3,)), 'right')
                        shape = anno['handBeta'].squeeze()
                        assert np.sum(joint_valid[self.joint_type[ht]]) > 0
                    else:
                        pose, trans, _ = transformManoParamsToCam(anno['leftHandPose'].squeeze(),
                                                                  anno['leftHandTrans'].squeeze(),
                                                                  anno['handBeta'].squeeze(),
                                                                  cv2.Rodrigues(coordChangMat)[0].squeeze(), np.zeros((3,)),
                                                                  'left')
                        shape = anno['handBeta'].squeeze()
                        assert np.sum(joint_valid[self.joint_type[ht]]) > 0
                else:
                    pose = np.zeros((48,)) * 1.0
                    trans = np.zeros((3,)) * 1.0
                    shape = np.zeros((10,)) * 1.0

                mano_pose.append(pose)
                mano_trans.append(trans.squeeze())
                mano_shape.append(shape)
            mano_pose = np.concatenate(mano_pose, axis=0)
            mano_trans = np.concatenate(mano_trans, axis=0)
            mano_shape = np.concatenate(mano_shape, axis=0)

            if True:
                # if both hands not in the image, skip it
                if np.sum(mano_valid) == 0:
                    continue

                if cfg.hand_type == 'right':
                    if mano_valid[0] == 0 or mano_valid[1] == 1:
                        continue
                elif cfg.hand_type == 'left':
                    if mano_valid[1] == 0 or mano_valid[0] == 1:
                        continue

            hand_type_valid = np.array([1.], dtype=np.float32)
            

            img_width, img_height = 640, 480
            if self.mode == 'evaluation' and ds_curr == 'ho3d':
                hand_bb = np.array([[anno['handBoundingBox'][0], anno['handBoundingBox'][1]], [anno['handBoundingBox'][2], anno['handBoundingBox'][3]]])
                tl = np.min(np.concatenate([hand_bb, obj_img], axis=0), axis=0)
                br = np.max(np.concatenate([hand_bb, obj_img], axis=0), axis=0)
            else:
                tl = np.min(np.concatenate([joint_img[joint_valid==1], obj_img], axis=0), axis=0)
                br = np.max(np.concatenate([joint_img[joint_valid==1], obj_img], axis=0), axis=0)
            box_size = br - tl
            bbox = np.concatenate([tl-10, box_size+20],axis=0)
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'],2], 'left': joint_cam[self.root_joint_idx['left'],2]}

            obj_bb_rest = anno['objCorners3DRest']


            obj_kps_3d_rest = np.load(osp.join(self.obj_kps_dir, '%s.npy' % (anno['objName'])))
            obj_kps_3d = obj_kps_3d_rest.dot(cv2.Rodrigues(obj_rot)[0].T) + obj_trans
            obj_kps_2d = cam2pixel(obj_kps_3d, focal, princpt)[:,:2]

            obj_pose_valid = 1.

            if anno['objName'] == '024_bowl':
                obj_bb_rest[:,:2] = 0


            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam.astype(np.float32), 'img_coord': joint_img.astype(np.float32), 'valid': joint_valid.astype(np.float32)}
            object = {'cam_coord': obj_cam.astype(np.float32), 'img_coord': obj_img.astype(np.float32),
                      'obj_bb_rest': obj_bb_rest.astype(np.float32), 'obj_kps_2d':obj_kps_2d,
                      'obj_kps_3d':obj_kps_3d, 'obj_id': int(anno['objName'][:3]), 'obj_kps_3d_rest': obj_kps_3d_rest}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param,
                    'bbox': bbox, 'joint': joint, 'object': object, 'hand_type': hand_type, 'hand_type_valid': hand_type_valid,
                    'abs_depth': abs_depth, 'file_name': frame_idx+'.png', 'capture': 0, 'cam': 0,
                    'frame': frame_idx, 'mano_pose': mano_pose, 'mano_trans': mano_trans, 'mano_shape': mano_shape,
                    'mano_valid': mano_valid, 'obj_rot': obj_rot, 'obj_trans': obj_trans, 'obj_pose_valid': obj_pose_valid, 'seg_path':seg_path,
                    'obj_seg_channel': obj_seg_channel}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)


        self.datalist = self.datalist_sh + self.datalist_ih
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

        return mesh.vertices[0].numpy(), self.mano_layer[hand_type].faces, mesh.joints[0].numpy()

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
        img_path, bbox, joint, object, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data['object'], data['hand_type'], data['hand_type_valid']
        seg_path = data['seg_path']
        obj_seg_channel = data['obj_seg_channel']
        mano_pose, mano_trans, mano_shape, mano_valid = data['mano_pose'].copy(), data['mano_trans'].copy(), data['mano_shape'].copy(), data['mano_valid'].copy()
        obj_rot, obj_trans = data['obj_rot'].copy(), data['obj_trans'].copy()
        joint_cam = joint['cam_coord'].copy(); joint_img = joint['img_coord'].copy(); joint_valid = joint['valid'].copy().astype(np.float32);
        obj_cam = object['cam_coord'].copy(); obj_img = object['img_coord'].copy(); obj_bb_rest = object['obj_bb_rest'];
        obj_kps_img = object['obj_kps_2d'].copy(); obj_kps_cam = object['obj_kps_3d'].copy(); obj_kps_rest = object['obj_kps_3d_rest'].copy()
        seq_name = data['seq_name']
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:,2,None].copy()),1)
        obj_coord = np.concatenate((obj_img, obj_cam[:,2,None].copy()),1)
        obj_kps_coord = np.concatenate((obj_kps_img, obj_kps_cam[:,2,None].copy()),1)

        obj_pose_valid = data['obj_pose_valid']

        # make the obj trans relative to hand, so that when augmentation is done its all good
        if cfg.predict_type == 'angles':
            obj_trans = obj_trans - mano_trans[:3]
        elif cfg.predict_type == 'vectors':
            obj_trans = obj_trans - joint_cam[self.root_joint_idx['right']]/1000

        num_kps = obj_kps_coord.shape[0]
        num_corners = 8

        joint_obj_coord = np.concatenate([joint_coord, obj_kps_coord, obj_coord],axis=0)
        joint_obj_cam = np.concatenate([joint_cam, obj_kps_cam, obj_cam],axis=0)

        # image load
        img = load_img(img_path)
        obj_seg = load_img(seg_path)
        obj_seg = cv2.resize(obj_seg.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # augmentation
        img, joint_obj_coord, joint_valid, hand_type, inv_trans, inv_trans_no_rot, do_flip, mano_pose,\
        mano_trans, joint_obj_cam, obj_seg, obj_rot, obj_trans =\
            augmentation(img, bbox, joint_obj_coord, joint_valid, hand_type, self.mode, self.joint_type,
                         mano_pose, mano_trans, mano_shape, joint_obj_cam, obj_seg, obj_rot, obj_trans)

        obj_seg = cv2.resize(obj_seg, (cfg.output_hm_shape[1], cfg.output_hm_shape[2]), interpolation=cv2.INTER_NEAREST)


        if do_flip:
            obj_pose_valid *= 0
            mano_valid = mano_valid[[1, 0]]


        rel_root_depth = np.array([joint_coord[self.root_joint_idx['left'],2] - joint_coord[self.root_joint_idx['right'],2]],dtype=np.float32).reshape(1)

        obj_root_valid = np.array([1]) * 1.0
        if cfg.predict_type == 'vectors':
            root_valid = np.array([np.sum(joint_valid[21:])>0])*1.0
            if (mano_valid[0]==1) and (mano_valid[1]==1): # if interacting hand
                root_valid *=  (joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']])
        elif cfg.predict_type == 'angles':
            root_valid = np.array([mano_valid[1]], dtype=np.float32)

        # transform to output heatmap space
        joint_obj_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_obj_coord,
                                                                                               joint_valid,
                                                                                               rel_root_depth,
                                                                                               root_valid,
                                                                                               self.root_joint_idx,
                                                                                               self.joint_type, self.skeleton)

        joint_cam = joint_obj_cam[:42]
        obj_cam = joint_obj_cam[42:42+num_kps]
        obj_corners_cam = joint_obj_cam[42+num_kps:]
        joint_coord = joint_obj_coord[:42].astype(np.float32)
        obj_coord = joint_obj_coord[42:42+num_kps].astype(np.float32)
        obj_corners_coord = joint_obj_coord[42+num_kps:]

        # fill some dummy values in obj_coord
        dummy = np.zeros((30-obj_coord.shape[0],3)) + np.array([-2000, -2000, -2000])
        obj_coord = np.concatenate([obj_coord, dummy],axis=0).astype(np.float32)

        obj_coord[:,2] = obj_coord[:,2] - obj_coord[0,2]# rel depths for now, not used anywhere yet!

        img = self.transform(img.astype(np.float32)) / 255.
        obj_seg = (obj_seg[:,:,obj_seg_channel]>200)*255.
        obj_seg = self.transform(obj_seg.astype(np.float32))[0]



        if cfg.predict_type == 'angles':
            rel_trans_hands_rTol = mano_trans[3:] - mano_trans[:3]
            rel_trans_obj = obj_trans
            if root_valid and (np.sum(joint_valid[:21]) == 0):  # when its only left hand image
                # when its only left hand, shift it closer to right hand
                rel_trans_hands_rTol = np.array([0.2, 0., 0.]).astype(np.float32)


            joint_cam_no_trans = joint_cam.copy().astype(np.float32)
            joint_cam_no_trans[self.joint_type['right']] -= mano_trans[:3] * 1000  # because joints in mm
            joint_cam_no_trans[self.joint_type['left']] -= mano_trans[3:] * 1000
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol*1000


        elif cfg.predict_type == 'vectors':
            joint_cam_no_trans = self.restore_coord_cam_from_img(joint_coord.copy(), inv_trans_no_rot, data, do_flip)

            rel_trans_hands_rTol = (joint_cam_no_trans[self.root_joint_idx['left']] - joint_cam_no_trans[
                self.root_joint_idx['right']]) / 1000
            rel_trans_obj = obj_trans
            if root_valid and (np.sum(joint_valid[:21]) == 0):  # when its only left hand image
                # when its only left hand, shift it closer to right hand
                rel_trans_hands_rTol = np.array([0.2, 0., 0.]).astype(np.float32)

            right_root_loc = joint_cam_no_trans[self.root_joint_idx['right']].copy()
            joint_cam_no_trans[self.joint_type['right']] -= joint_cam_no_trans[self.root_joint_idx['right']]
            joint_cam_no_trans[self.joint_type['left']] -= joint_cam_no_trans[self.root_joint_idx['left']]
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol * 1000



        obj_kps_3d = obj_kps_rest.dot(cv2.Rodrigues(obj_rot)[0].T) + rel_trans_obj


        # use zero mask for now. Later if required put ones along padded pixels
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))

        if cfg.predict_type == 'angles':
            # get the mano vertices
            verts_right, faces, mano_joints_right = self.get_mano_vertices(mano_pose[:48], mano_shape[:10], np.zeros((3,)), 'right')
            verts_left, faces, mano_joints_left = self.get_mano_vertices(mano_pose[48:], mano_shape[10:], rel_trans_hands_rTol, 'left') # add the rel trans in loss.py
            verts = np.concatenate([verts_right, verts_left], axis=0)
        else:
            verts = np.zeros((1,3))



        if np.sum(obj_seg.cpu().numpy())/128/128/255 < 0.02:
            obj_pose_valid *= 0.
        else:
            obj_pose_valid *= 1.



        # heatmap valid flag. If its interacting and only one of the hand annotations is valid, set hm_valid=0
        if (np.sum(hand_type) == 2 and np.sum(joint_valid[:21]) < 21) or (
                np.sum(hand_type) == 2 and np.sum(joint_valid[21:]) < 21):
            hm_valid = np.array([0.]).astype(np.float32)
        else:
            hm_valid = np.array([1.]).astype(np.float32)

        if np.sum(joint_valid[:21]) < 21:
            mano_valid[0] = 0

        if np.sum(joint_valid[21:]) < 21:
            mano_valid[1] = 0

        if np.sum(img.numpy()) == 0:
            mano_valid *= False
            joint_valid *= 0
            root_valid *= 0
            hand_type_valid *= 0
            obj_pose_valid *= 0

        inputs = {'img': img, 'mask': mask}
        targets = {'joint_coord': joint_coord, 'rel_trans_hands_rTol': rel_trans_hands_rTol, 'hand_type': hand_type,
                   'mano_pose':mano_pose, 'mano_shape': mano_shape, 'joint_cam_no_trans': joint_cam_no_trans, 'verts': verts,
                   'obj_rot': obj_rot.astype(np.float32), 'rel_obj_trans': rel_trans_obj.astype(np.float32),
                   'obj_kps_coord':obj_coord, 'obj_seg': obj_seg, 'obj_corners_coord': obj_corners_coord,
                   'obj_kps_3d': obj_kps_3d}

        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid, 'obj_root_valid': obj_root_valid,
                     'mano_valid': mano_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']), 'cam': int(data['cam']),
                     'frame': int(data['frame']), 'seq_id': (seq_name[:9]), 'obj_bb_rest': obj_bb_rest.astype(np.float32),
                     'obj_pose_valid':obj_pose_valid, 'focal': data['cam_param']['focal'], 'princpt': data['cam_param']['princpt'],
                     'obj_id': data['object']['obj_id'], 'hm_valid': hm_valid}
        return inputs, targets, meta_info



