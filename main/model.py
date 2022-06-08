# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch.nn.functional as F
from common.nets.layer import MLP
from common.nets.module import BackboneNet, DecoderNet, DecoderNet_big
from common.nets.transformer import Transformer
from common.nets.loss import *
from common.nets.position_encoding import build_position_encoding
from common.utils.preprocessing import PeakDetector
from common.utils.misc import *
import numpy as np
import time

class Model(nn.Module):
    def __init__(self, backbone_net, decoder_net, transformer, position_embedding):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.decoder_net = decoder_net
        self.transformer = transformer
        self.position_embedding = position_embedding

        start = time.time()
        self.peak_detector = PeakDetector()
        self.obj_peak_detector = PeakDetector(nearest_neighbor_th=5)
        print('Init of peak detector took %f s'%(time.time()-start))



        output_dim = cfg.hidden_dim

        if cfg.position_embedding == 'simpleCat':
            output_dim = output_dim - 32


        # MLP for converting concatenated image features to 256-D features
        self.norm1 = nn.LayerNorm(cfg.mutliscale_dim)
        self.linear1 = MLP(input_dim=cfg.mutliscale_dim, hidden_dim=[1024, 512, 256],
                           output_dim=output_dim,
                           num_layers=4, is_activation_last=True)
        self.activation = nn.functional.relu

        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.shape_query_embed = None



        # MLPs for keypoint Classification
        if cfg.has_object:
            self.linear_class = MLP(cfg.hidden_dim, cfg.hidden_dim,
                                    (42 + 1 + 1) if cfg.hand_type == 'both' else (21 + 1 + 1),
                                    4)
        else:
            self.linear_class = MLP(cfg.hidden_dim, cfg.hidden_dim,
                                    (42+1) if cfg.hand_type=='both' else (21+1),
                                    4)



        # Pose Regression MLPs
        pose_fan_out = 3
        if cfg.predict_type == 'angles':
            self.linear_pose = MLP(cfg.hidden_dim, cfg.hidden_dim, pose_fan_out, 3)
        elif cfg.predict_type == 'vectors':
            if not cfg.predict_2p5d :
                self.linear_joint_vecs =  MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)
            else:
                self.linear_joint_2p5d = {}
                self.linear_joint_2p5d_px = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.output_hm_shape[1], 3)
                self.linear_joint_2p5d_py = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.output_hm_shape[2], 3)
                self.linear_joint_2p5d_dep = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.output_hm_shape[0], 3)
                self.softmax = nn.Softmax(dim=3)

        self.linear_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)

        if cfg.has_object:
            self.linear_obj_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)
            if cfg.predict_obj_left_hand_trans:
                self.linear_obj_left_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3,
                                                3)
            self.linear_obj_corner_proj = MLP(cfg.hidden_dim, cfg.hidden_dim, 16, 3)
            if cfg.use_obj_rot_parameterization:
                self.linear_obj_rot = MLP(cfg.hidden_dim, cfg.hidden_dim, 6, 3)
            else:
                self.linear_obj_rot = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)


            if not cfg.use_big_decoder:
                self.linear1_obj = MLP(input_dim=(512+256), hidden_dim=[512, 256, 256],
                                   output_dim=output_dim,
                                   num_layers=4, is_activation_last=True)
            else:
                self.linear1_obj = MLP(input_dim=3072, hidden_dim=[1024, 512, 256],
                                       output_dim=output_dim,
                                       num_layers=4, is_activation_last=True)



        # MLPs for predicting Hand shape and camera parameters
        if cfg.use_2D_loss:
            self.linear_shape = MLP(cfg.hidden_dim, cfg.hidden_dim, 10, 3)
            self.linear_cam = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)

        else:
            self.linear_shape = MLP(cfg.hidden_dim, cfg.hidden_dim, 10, 3)



        # MLP for predicting the hand type after the U-Net decoder
        if cfg.use_bottleneck_hand_type:
            if cfg.resnet_type >= 50:
                self.linear_bottleneck_hand_type = MLP(2048, 512, 2, 2)
            else:
                self.linear_bottleneck_hand_type = MLP(512, 512, 2, 2)

        if cfg.hand_type == 'both':
            self.linear_hand_type = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 2)
            self.hand_type_query_embed = None
        else:
            self.hand_type_query_embed = None



        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.obj_seg_loss = ObjSegLoss()
        self.pose_loss = PoseLoss()
        self.rel_trans_loss = RelTransLoss()
        self.joints_loss = JointLoss()
        self.vertex_loss = VertexLoss()
        self.shape_reg = ShapeRegularize()
        self.joint_class_loss = JointClassificationLoss()
        self.hand_type_loss = HandTypeLoss()
        self.cam_param_loss = CameraParamLoss()
        self.shape_loss = ShapeLoss()
        if cfg.use_bottleneck_hand_type:
            self.bottleneck_hand_type_loss = BottleNeckHandTypeLoss()
        self.mano_mesh = ManoMesh()
        if cfg.predict_type == 'vectors':
            self.joint_vecs_loss = JointVectorsLoss()
            if cfg.predict_2p5d:
                self.joint_2p5d_loss = Joints2p5dLoss()

        self.obj_pose_loss = ObjPoseLoss()



        # Freeze batch norm layers
        self.freeze_stages()
     

    def freeze_stages(self):

        for name, param in self.backbone_net.named_parameters():
            if 'bn' in name:
                param.requires_grad = False

    def render_gaussian_heatmap(self, joint_coord, joint_valid ):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy,xx = torch.meshgrid(y,x)
        xx = xx[None,None,:,:].cuda().float(); yy = yy[None,None,:,:].cuda().float()
        
        if cfg.hand_type == 'both':
            joint_coord1 = joint_coord # N x 42 x 3
            joint_valid1 = joint_valid
        elif cfg.hand_type == 'right':
            joint_coord1 = joint_coord[:,:21]  # N x 21 x 3
            joint_valid1 = joint_valid[:,:21]
        elif cfg.hand_type == 'left':
            joint_coord1 = joint_coord[:,21:]  # N x 21 x 3
            joint_valid1 = joint_valid[:, 21:]

        x = joint_coord1[:,:,0,None,None]; y = joint_coord1[:,:,1,None,None]
        heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2) * joint_valid1[:,:,None,None] # N x 42 x h x w
        heatmap = torch.sum(heatmap, 1)
        heatmap = heatmap * 255

        return heatmap

    def sample_obj_seg(self, obj_seg):
        total_area = np.sum(obj_seg>cfg.intensity_th)
        nms_th = np.sqrt(total_area/(np.pi*cfg.num_obj_samples))
        peaks, peaks_ind_list = self.obj_peak_detector.detect_peaks_nms(obj_seg, cfg.num_obj_samples,
                                                                        intensity_th=15)#, nms_th)
        return peaks_ind_list

    def get_input_seq(self, joint_heatmap_out, obj_seg_out, feature_pyramid, pos_embed, meta_info, joint_coord, joint_valid, obj_seg_gt, epoch_cnt):
        heatmap_np = joint_heatmap_out.detach().cpu().numpy()
        if cfg.has_object:
            obj_seg_np = obj_seg_out.detach().cpu().numpy()

        if epoch_cnt<cfg.num_epochs_gt_peak_locs:
            use_gt_peak_locs = True
        else:
            use_gt_peak_locs = False

        grids = []
        masks = []
        peak_joints_map_batch = []
        normalizer = np.array([cfg.output_hm_shape[1] - 1, cfg.output_hm_shape[2] - 1]) / 2
        for ii in range(heatmap_np.shape[0]):
            if use_gt_peak_locs:
                peaks_ind_list = joint_coord[ii,:,[1,0]].cpu().numpy()
                peak_joints_map = np.arange(0, peaks_ind_list.shape[0]) + 1
                mask1 = np.logical_or(peaks_ind_list[:, 0] < 0, peaks_ind_list[:, 1] < 0)
                mask2 = np.logical_or(peaks_ind_list[:, 0] > cfg.output_hm_shape[1]-1, peaks_ind_list[:, 1] > cfg.output_hm_shape[2]-1)
                mask = np.logical_not(np.logical_or(mask1, mask2))
                mask = np.logical_and(mask, joint_valid[ii].cpu().numpy())
                peaks_ind_list = peaks_ind_list[mask]
                peak_joints_map = peak_joints_map[mask]

                if cfg.has_object:
                    obj_peaks_ind_list = self.sample_obj_seg(obj_seg_gt[ii].cpu().numpy())

                    if len(obj_peaks_ind_list) == 0:
                        # print('Found %d object peaks for %s/%s/' % (len(obj_peaks_ind_list),
                        #                                           str(meta_info['seq_id'][ii]),
                        #                                           str(meta_info['frame'][ii])))
                        meta_info['obj_pose_valid'][ii] *= 0

                    if len(obj_peaks_ind_list) > 0:
                        obj_peaks_ind_list = np.stack(obj_peaks_ind_list, axis=0)
                        mask1 = np.logical_or(obj_peaks_ind_list[:, 0] < 0, obj_peaks_ind_list[:, 1] < 0)
                        mask2 = np.logical_or(obj_peaks_ind_list[:, 0] > cfg.output_hm_shape[1] - 1,
                                              obj_peaks_ind_list[:, 1] > cfg.output_hm_shape[2] - 1)
                        mask = np.logical_not(np.logical_or(mask1, mask2))
                        obj_peaks_ind_list = obj_peaks_ind_list[mask]

                        obj_peak_map = np.zeros((len(obj_peaks_ind_list))) + cfg.obj_cls_index
                        peaks_ind_list = np.concatenate([peaks_ind_list, np.array(obj_peaks_ind_list)], axis=0)
                        peak_joints_map = np.concatenate([peak_joints_map, obj_peak_map], axis=0)
            else:
                peaks, peaks_ind_list = self.peak_detector.detect_peaks_nms(heatmap_np[ii],
                                                                            (cfg.max_num_peaks-cfg.num_obj_samples) if cfg.has_object else cfg.max_num_peaks)
                peak_joints_map = np.zeros((len(peaks_ind_list)), dtype=np.int)+1
                if cfg.has_object:
                    obj_peaks_ind_list = self.sample_obj_seg(obj_seg_np[ii])
                    if len(obj_peaks_ind_list) > 0:
                        obj_peak_map = np.zeros((len(obj_peaks_ind_list))) + cfg.obj_cls_index
                        if len(peaks_ind_list) > 0:
                            peaks_ind_list = np.concatenate([peaks_ind_list, np.array(obj_peaks_ind_list)], axis=0)
                            peak_joints_map = np.concatenate([peak_joints_map, obj_peak_map], axis=0)
                        else:
                            peaks_ind_list = np.array(obj_peaks_ind_list)
                            peak_joints_map = obj_peak_map
                    else:
                        # Corner case when the object is heavily occluded
                        # print('Found %d object peaks for %s/%s/' % (len(obj_peaks_ind_list),
                        #                                      str(meta_info['seq_id'][ii]),
                        #                                      str(meta_info['frame'][ii])))
                        meta_info['obj_pose_valid'][ii] *= 0

            if len(peak_joints_map) != len(peaks_ind_list):
                print(len(peak_joints_map), len(peaks_ind_list))
                assert len(peak_joints_map) == len(peaks_ind_list)

            if len(peaks_ind_list) == 0:
                # Corner case when the object and hand is heavily occluded in the image

                # print('Found %d peaks for %s/%s/%s/%s'%(len(peaks_ind_list), str(meta_info['capture'][ii]),
                #                                      str(meta_info['cam'][ii]), str(meta_info['seq_id'][ii]), str(meta_info['frame'][ii])))
                meta_info['mano_valid'][ii] = False
                meta_info['joint_valid'][ii] *= 0
                meta_info['hand_type_valid'][ii] *= 0
                peaks_pixel_locs_normalized = np.tile(np.array([[-1, -1]]), (cfg.max_num_peaks,1))
                mask = np.ones((cfg.max_num_peaks), dtype=np.bool)
                peak_joints_map = np.zeros((cfg.max_num_peaks,), dtype=np.int)
            else:
                peaks_ind_normalized = (np.array(peaks_ind_list) - normalizer) / normalizer
                assert np.sum(peaks_ind_normalized < -1) == 0 and np.sum(peaks_ind_normalized > 1) == 0

                peaks_pixel_locs_normalized = peaks_ind_normalized[:, [1, 0]]  # in pixel coordinates
                mask = np.ones((peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)

                # fill up the empty slots with some dummy values
                if peaks_pixel_locs_normalized.shape[0] < cfg.max_num_peaks:
                    dummy_peaks = np.tile(np.array([[-1, -1]]), (cfg.max_num_peaks - peaks_pixel_locs_normalized.shape[0],1))
                    invalid_mask = np.zeros((cfg.max_num_peaks - peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)
                    peak_joints_map = np.concatenate([peak_joints_map, invalid_mask.astype(np.int)], axis=0)

                    peaks_pixel_locs_normalized = np.concatenate([peaks_pixel_locs_normalized, dummy_peaks], axis=0)
                    mask = np.concatenate([mask, invalid_mask], axis=0)


            grids.append(peaks_pixel_locs_normalized)
            masks.append(mask)
            peak_joints_map_batch.append(peak_joints_map)

        peak_joints_map_batch = torch.from_numpy(np.stack(peak_joints_map_batch, 0)).to(pos_embed.device) # N x max_num_peaks
        grids = np.stack(grids, 0)  # N x max_num_peaks x 2
        grids_unnormalized_np = grids*normalizer[[1,0]] + normalizer[[1,0]] # in pixel coordinates space
        masks_np = np.stack(masks, 0)  # N x max_num_peaks
        masks = torch.from_numpy(masks_np).bool().to(pos_embed.device) # N x max_num_peaks


        # Get the positional embeddings
        positions = nn.functional.grid_sample(pos_embed,
                                              torch.from_numpy(np.expand_dims(grids, 1)).float().to(pos_embed.device),
                                              mode='nearest', align_corners=True).squeeze(2) # N x hidden_dim x max_num_peaks


        # Sample the CNN features
        multiscale_features = []
        grids_tensor = torch.from_numpy(np.expand_dims(grids, 1)).float().to(feature_pyramid[cfg.mutliscale_layers[0]].device)
        for layer_name in cfg.mutliscale_layers:
            # N x C x 1 x max_num_peaks
            multiscale_features.append(torch.nn.functional.grid_sample(feature_pyramid[layer_name],
                                                                       grids_tensor,
                                                                         align_corners=True))

        multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2) # N x C1 x  max_num_peaks
        multiscale_features = multiscale_features.permute(0, 2, 1) # N x max_num_peaks x C1


        if cfg.has_object:
            input_seq_hands = self.linear1(multiscale_features)  # N x max_num_peaks x hidden_dim

            # For object use CNN features from deeper layers only as they have higer receptive field
            if not cfg.use_big_decoder:
                input_seq_object = self.linear1_obj(multiscale_features[:,:,-(512+256):])  # N x max_num_peaks x hidden_dim
            else:
                input_seq_object = self.linear1_obj(multiscale_features[:, :, -3072:])  # N x max_num_peaks x hidden_dim

            input_seq = input_seq_hands*(peak_joints_map_batch.unsqueeze(2)!=cfg.obj_cls_index)\
                        + input_seq_object*(peak_joints_map_batch.unsqueeze(2)==cfg.obj_cls_index)
            input_seq = input_seq.permute(0, 2, 1) # N x hidden_dim x max_num_peaks
        else:
            input_seq = self.linear1(multiscale_features).permute(0, 2, 1) # N x hidden_dim x max_num_peaks

        return input_seq, masks, positions, grids_unnormalized_np, masks_np, peak_joints_map_batch

    def forward(self, inputs, targets, meta_info, mode, epoch_cnt=1e8):
        input_img = inputs['img']
        input_mask = inputs['mask']
        batch_size = input_img.shape[0]

        img_feat, enc_skip_conn_layers = self.backbone_net(input_img)
        feature_pyramid, decoder_out = self.decoder_net(img_feat, enc_skip_conn_layers)


        joint_heatmap_out = decoder_out[:,0]


        if cfg.has_object:
            obj_seg_out = decoder_out[:,1]
            obj_seg_gt = targets['obj_seg']
            obj_kps_coord_gt = targets['obj_kps_coord']
        else:
            obj_seg_out = None
            obj_seg_gt = None
            obj_kps_coord_gt = None

        # Get the positional embeddings
        pos = self.position_embedding(nn.functional.interpolate(input_img, (cfg.output_hm_shape[2], cfg.output_hm_shape[1])),
                                nn.functional.interpolate(input_mask, (cfg.output_hm_shape[2], cfg.output_hm_shape[1])))

        # Get the input tokens
        input_seq, masks, positions, joint_loc_pred_np, mask_np, peak_joints_map_batch \
            = self.get_input_seq(joint_heatmap_out, obj_seg_out, feature_pyramid, pos, meta_info,
                                 targets['joint_coord'], meta_info['joint_valid'], obj_seg_gt, epoch_cnt)

        if cfg.use_bottleneck_hand_type:
            bottleneck_hand_type_feat = F.avg_pool2d(img_feat, (img_feat.shape[2], img_feat.shape[3])).view(-1, img_feat.shape[1])
            bottleneck_hand_type = torch.sigmoid(self.linear_bottleneck_hand_type(bottleneck_hand_type_feat))



        # Concatenate positional and appearance embeddings
        if cfg.position_embedding == 'simpleCat':
            input_seq = torch.cat([input_seq, positions], dim=1)
            positions = torch.zeros_like(input_seq).to(input_seq.device)



        # Define attention masks
        tgt_key_padding_mask = None
        if cfg.hand_type == 'both':
            # define attention masks on the queries. This is irrelevant when using only 1 cross-attention layer.
            tgt_mask = get_tgt_mask().to(input_seq.device)

        if cfg.has_object:
            _, memory_mask = get_src_memory_mask(peak_joints_map_batch)
            memory_mask = memory_mask.to(input_seq.device)
            src_mask = None
        else:
            src_mask = None
            memory_mask = None


        transformer_out, hand_type, memory, encoder_out, attn_wts_all_layers = self.transformer(src=input_seq, mask=torch.logical_not(masks),
                                                                                       query_embed=self.query_embed.weight,
                                                                                       pos_embed=positions,
                                                                                       tgt_mask=tgt_mask if cfg.use_tgt_mask else None,
                                                                                       tgt_key_padding_mask = tgt_key_padding_mask,
                                                                                       src_mask=src_mask,
                                                                                       memory_mask=memory_mask)


        # Make all the predictions
        if cfg.hand_type == 'both':
            if cfg.predict_type == 'angles':
                pose = self.linear_pose(transformer_out[:, :(cfg.num_joint_queries_per_hand*2)])  # 6 x 32 x N x 3(9)
                shape = self.linear_shape(transformer_out[:, cfg.shape_indx])  # 6 x N x 10

            elif cfg.predict_type == 'vectors':
                if cfg.predict_2p5d:
                    joint_px_hm = self.linear_joint_2p5d_px(transformer_out[:, :cfg.num_joint_queries_per_hand*2]) # 6 x 42 x N x 128
                    joint_py_hm = self.linear_joint_2p5d_py(transformer_out[:, :cfg.num_joint_queries_per_hand * 2])  # 6 x 42 x N x 128
                    joint_dep_hm = self.linear_joint_2p5d_dep(transformer_out[:, :cfg.num_joint_queries_per_hand * 2])  # 6 x 42 x N x 128
                    joint_2p5d_hm = torch.cat([joint_px_hm.unsqueeze(3),
                                               joint_py_hm.unsqueeze(3), joint_dep_hm.unsqueeze(3)], dim=3) # 6 x 42 x N x 3 x 128
                else:
                    joint_vecs = self.linear_joint_vecs(transformer_out[:, :cfg.num_joint_queries_per_hand*2]) # 6 x 40 x N x 3
            else:
                raise NotImplementedError

            rel_trans = self.linear_rel_trans(transformer_out[:, cfg.shape_indx])

            if cfg.has_object:
                obj_rot = self.linear_obj_rot(transformer_out[:, cfg.obj_rot_indx])  # 6 x N x 3
                obj_trans = self.linear_obj_rel_trans(transformer_out[:, cfg.obj_trans_indx])
                if cfg.predict_obj_left_hand_trans:
                    obj_trans_left = self.linear_obj_left_rel_trans(transformer_out[:, cfg.obj_trans_indx])
                else:
                    obj_trans_left = None
                obj_corner_proj = self.linear_obj_corner_proj(transformer_out[:, cfg.obj_trans_indx]) # 6 x N x 16


        if cfg.use_2D_loss:
            cam_param = self.linear_cam(transformer_out[:, cfg.shape_indx]) # 6 x N x 3
        else:
            cam_param = None

        if cfg.enc_layers>0:
            joint_class = self.linear_class(encoder_out) # 6 x max_num_peaks x N x 22(43)


        # Put all the outputs in a dict
        out = {}
        out['rel_trans_out'] = rel_trans[-1]
        out['attn_weights_out'] = attn_wts_all_layers[0]
        if 'inv_trans' in meta_info:
            out['inv_trans_out'] = meta_info['inv_trans']
        out['joint_heatmap_out'] = joint_heatmap_out
        if cfg.predict_type == 'angles':
            out['pose_out'] = pose
            out['shape_out'] = shape

        if cfg.enc_layers > 0:
            out['joint_class_out'] = joint_class[-1].permute(1,0,2)# N x max_num_peaks
        out['seq_mask_out'] = torch.from_numpy(mask_np).to(transformer_out.device)
        out['joint_loc_pred_out'] = torch.from_numpy(joint_loc_pred_np).to(transformer_out.device)
        out['cam_param_out'] = cam_param[-1]
        if cfg.hand_type == 'both':
            if cfg.use_bottleneck_hand_type:
                out['hand_type_out'] = bottleneck_hand_type  # N x 2
            else:
                out['hand_type_out'] = torch.argmax(hand_type[-1,0], dim=1) # N

        if cfg.has_object:
            out['obj_rot_out'] = obj_rot[-1]
            out['obj_trans_out'] = obj_trans[-1]
            out['obj_corner_proj_out'] = obj_corner_proj[-1]
            out['obj_seg_gt_out'] = obj_seg_gt
            out['obj_seg_pred_out'] = obj_seg_out




        # Get all the losses for all the predictions
        loss = {}
        if cfg.predict_type == 'angles':
            joints_pred = self.mano_mesh.get_mano_mesh(pose, shape, rel_trans, meta_info['root_valid'], cam_param)
        elif cfg.predict_type == 'vectors':
            joints_pred = {}
            if cfg.predict_2p5d:
                loss['joint_2p5d_hm'] = self.joint_2p5d_loss(joint_2p5d_hm, targets['joint_coord'], meta_info['joint_valid'])
                out['joint_2p5d_out'] = torch.argmax(joint_2p5d_hm[-1], dim=-1).permute(1,0,2) # N x 42 x 3
            else:
                loss['joint_vec'], loss['joints_loss'], loss['joints2d_loss'], joints_pred['joints_right'], joints_pred['joints_left'] = \
                self.joint_vecs_loss(joint_vecs, targets['joint_cam_no_trans'], meta_info['joint_valid'], targets['joint_coord'][:,:,:2],
                                     cam_param, rel_trans)
                out['joint_3d_right_out'] = joints_pred['joints_right']
                out['joint_3d_left_out'] = joints_pred['joints_left']
                joints_pred['joints2d_right'] = joints_pred['joints_right'][:, :, :, :2] * cam_param[:, :, :1].unsqueeze(2) + cam_param[:, :,
                                                                                                  1:].unsqueeze(2)
                joints_pred['joints2d_left'] = joints_pred['joints_left'][:, :, :, :2] * cam_param[:, :, :1].unsqueeze(2) + cam_param[:, :,
                                                                                                    1:].unsqueeze(2)


        target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'], meta_info['joint_valid'])
        loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap, meta_info['joint_valid'], meta_info['hm_valid'])

        if cfg.has_object:
            target_obj_kps_heatmap = targets['obj_seg']
            out['obj_kps_gt_out'] = target_obj_kps_heatmap
            loss['obj_seg'] = self.obj_seg_loss(obj_seg_out, target_obj_kps_heatmap)
            target_obj_kps_3d = targets['obj_kps_3d']
        else:
            target_obj_kps_heatmap = None
            target_obj_kps_3d = None

        if mode != 'train':
            out1 = {**loss, **out}
            return out1

        if cfg.enc_layers>0:
            loss['cls'], row_inds_batch_list, asso_inds_batch_list \
                = self.joint_class_loss(joint_loc_pred_np, targets['joint_coord'][:,:,:2],
                                        mask_np, joint_class.permute(0,2,1,3), meta_info['joint_valid'], peak_joints_map_batch,
                                        targets['joint_cam_no_trans'],
                                        target_obj_kps_heatmap.cpu().numpy() if cfg.has_object else None,
                                        obj_kps_coord_gt.cpu().numpy() if cfg.has_object else None,
                                        target_obj_kps_3d.cpu().numpy() if cfg.has_object else None)
        else:
            loss['cls'] = torch.zeros((1,)).to(transformer_out.device)


        if cfg.use_bottleneck_hand_type:
            loss['hand_type'] = self.bottleneck_hand_type_loss(bottleneck_hand_type, targets['hand_type'], meta_info['hand_type_valid'])
        else:
            if cfg.hand_type == 'both':
                loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'],
                                                            meta_info['hand_type_valid'])

        if cfg.predict_type == 'angles':
            loss['pose'] = self.pose_loss(pose, targets['mano_pose'], meta_info['mano_valid'])
            loss['shape_reg'] = self.shape_reg(shape)
            loss['shape_loss'] = self.shape_loss(targets['mano_shape'], shape, meta_info['mano_valid'])
            loss['joints_loss'], loss['joints2d_loss'] = self.joints_loss(rel_trans, targets['joint_cam_no_trans'],
                                                   meta_info['joint_valid'], targets['joint_coord'][:,:,:2], meta_info['root_valid'],
                                                                          joints_pred, cam_param)
            loss['vertex_loss'] = torch.zeros((1,)).to(pose.device)


        if cfg.hand_type == 'both':
            loss['rel_trans'] = self.rel_trans_loss(rel_trans, targets['rel_trans_hands_rTol'],
                                                    meta_info['root_valid'])



        if cfg.use_2D_loss and ((not cfg.predict_2p5d) or (cfg.predict_type=='angles')) :
            loss['cam_scale'], loss['cam_trans'], cam_param_gt = self.cam_param_loss(targets['joint_cam_no_trans'], meta_info['joint_valid'],
                                                                       targets['joint_coord'][:, :, :2], cam_param,
                                                                       meta_info['root_valid'], rel_trans)

        if cfg.has_object:
            loss['obj_corners'], loss['obj_rot'], loss['obj_trans'],\
            loss['obj_corners_proj'], loss['obj_weak_proj'] = self.obj_pose_loss(obj_rot, obj_trans, obj_trans_left, rel_trans, targets['obj_rot'],
                                                                                         targets['rel_obj_trans'],
                                                                                         targets['rel_trans_hands_rTol'],
                                                                                         meta_info['obj_bb_rest'],
                                                                                         meta_info['obj_pose_valid'],
                                                                                         obj_corner_proj,
                                                                                         targets['obj_corners_coord'],
                                                                                         meta_info['obj_id'], meta_info['root_valid'], cam_param, cam_param_gt[0])





        out1 = {**loss, **out}
        return out1



def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode, joint_num):
    backbone_net = BackboneNet()

    if cfg.use_big_decoder:
        decoder_net = DecoderNet_big()
    else:
        decoder_net = DecoderNet()

    transformer = Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

    print('BackboneNet No. of Params = %d'%(sum(p.numel() for p in backbone_net.parameters() if p.requires_grad)))
    print('decoder_net No. of Params = %d' % (sum(p.numel() for p in decoder_net.parameters() if p.requires_grad)))
    print('transformer No. of Params = %d' % (sum(p.numel() for p in transformer.parameters() if p.requires_grad)))


    position_embedding = build_position_encoding(cfg)

    if mode == 'train':
        backbone_net.init_weights()
        decoder_net.apply(init_weights)

    model = Model(backbone_net, decoder_net, transformer, position_embedding)
    print('Total No. of Params = %d' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    return model

