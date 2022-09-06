# Copyright (c) 2020 Graz University of Technology All rights reserved.

import os
import os.path as osp
import sys
from common.utils.dir import add_pypath, make_folder
from datetime import datetime
import torch
import random
import numpy as np

def fix_seeds(random_seed):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class Config:
    # ~~~~~~~~~~~~~~~~~~~~~~Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    dataset = 'InterHand2.6M'  # InterHand2.6M, ho3d, h2o3d, ho3d_h2o3d
    pose_representation = 'angles' #3D, 2p5D, angles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    #~~~~~~~~~~~~~~~~~~~~~~SMPLX path~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    smplx_path = '/home/shreyas/Documents/opendr/mano/models/' # Path to MANO model files
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # ~~~~~~~~~~~~~~~~~~~~~~InterHand2.6M paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    interhand_anno_dir = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/networks/InterHand2.6M/data/InterHand2.6M/annotations'
    interhand_images_path = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/networks/InterHand2.6M/data/InterHand2.6M/images'

    # RootNet output path
    root_net_output_path = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/networks/InterHand2.6M/data/InterHand2.6M/RootNet.output.on.InterHand2.6M'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # ~~~~~~~~~~~~~~~~~~~~~~HO-3D paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ho3d_anno_dir = '/media/shreyas/ssd2/Dataset/HO3D_Release_Final'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # ~~~~~~~~~~~~~~~~~~~~~~H2O-3D paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    h2o3d_anno_dir = '/media/shreyas/4aa82be1-14a8-47f7-93a7-171e3ebac2b0/Datasets/dataset_h2o3d_final/dataset_h2o3d_final'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # ~~~~~~~~~~~~~~~~~~~~~~Object Model paths~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    object_models_dir = '/media/shreyas/ssd2/Networks/handobjectconsist/models'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # ~~~~~~~~~~~~~~~~~~~~~~Training Setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # number of epochs for which keypoints are obtained from ground-truth heatmaps and ground-truth object segmentation map
    num_epochs_gt_peak_locs = 5 if dataset == 'InterHand2.6M' else 25
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



    # Path to J_regressor_mano_ih26m.py in InterHand2.6M dataset
    joint_regr_np_path = '../tool/MANO_joint_regressor/J_regressor_mano_ih26m.npy'
    assert os.path.exists(joint_regr_np_path), 'Unable to find J_regressor_mano_ih26m.npy'

    skeleton_file = '../tool/hand_skeleton/skeleton.txt'
    assert os.path.exists(skeleton_file), 'Unable to find skeleton.txt'

    obj_kps_dir = '../tool/ho3d_object_kps'

    # Some internal conversions for pose representation setting
    if pose_representation == '3D':
        # 3D pose
        predict_type = 'vectors'  # 'angles', 'vectors'
        predict_2p5d = False
    elif pose_representation == '2p5D':
        # 2.5 pose
        predict_type = 'vectors'  # 'angles', 'vectors'
        predict_2p5d = True
    elif pose_representation == 'angles':
        # MANO Joint Angles
        predict_type = 'angles'  # 'angles', 'vectors'
        predict_2p5d = False
    else:
        raise NotImplementedError

    ## model
    use_big_decoder = False
    resnet_type = 50  # 18, 34, 50, 101, 152
    mutliscale_layers = ['stride2', 'stride4', 'stride8', 'stride16', 'stride32']

    def calc_mutliscale_dim(self, use_big_decoder_l, resnet_type_l):
        if use_big_decoder_l:
            self.mutliscale_dim = 128 + 256 + 512 + 1024 + 2048
        else:
            if resnet_type_l >= 50:
                self.mutliscale_dim = 32 + 64 + 128 + 256 + 512
            else:
                self.mutliscale_dim = 32 + 64 + 128 + 256 + 512


    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (128, 128, 128)  # (depth, height, width)
    sigma = 2.5 / 2
    bbox_3d_size = 300  # depth axis
    num_faces_mano = 1538

    # ~~~~~~~~~~~Some Frozen Configurations~~~~~~~~~~~~~~~~~~~~~#
    position_embedding = 'simpleCat' #'sine' #'convLearned'  # learned
    use_tgt_mask = True
    use_2D_loss = True
    use_bottleneck_hand_type = True
    dep_rel_to = 'parent' # parent, root
    use_obj_rot_parameterization = True
    predict_obj_left_hand_trans = True
    if dataset == 'ho3d':
        predict_obj_left_hand_trans = False

    if dataset in ['ho3d', 'h2o3d', 'ho3d_h2o3d']:
        has_object = True
    else:
        has_object = False

    hand_type = 'both'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


    # Some more hyper-parameters
    max_num_peaks = 100 # number of peaks to extract after 1st stage
    num_obj_samples = 50
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 6
    dec_layers = 6
    pre_norm = False

    obj_cls_index = 43 if hand_type == 'both' else 22

    # peak selection config
    intensity_th = 20
    nearest_neighbor_th = 5


    # Queries config
    num_queries = 15 + 1 + 1
    num_joint_queries_per_hand = 16

    if hand_type in ['right', 'left']:
        shape_indx = num_joint_queries_per_hand
    elif hand_type == 'both':
        num_queries = 2*num_queries # joints+shape+trans
        shape_indx = num_joint_queries_per_hand*2
    else:
        raise NotImplementedError


    if predict_type == 'vectors':
        num_joint_queries_per_hand = 20
        num_queries = 20
        if predict_2p5d:
            num_queries += 1 # for the root joint
            num_joint_queries_per_hand += 1
        if hand_type == 'both':
            num_queries = 2 * num_queries
        num_queries += 1 # for cam_param and trans
        shape_indx = num_queries - 1

    if has_object:
        num_queries += 2 # one for rot and one for trans
        obj_rot_indx = num_queries - 2
        obj_trans_indx = num_queries - 1


    ## optimization config
    lr_dec_epoch = [15, 17] if dataset == 'InterHand2.6M' else [45,47]
    end_epoch = 30 if dataset == 'InterHand2.6M' else 150
    lr = 1e-4
    lr_dec_factor = 2
    train_batch_size = 90
    lr_drop = 200

    ## weights
    hm_weight = 100/10000
    pose_weight = 0.07
    rel_trans_weight = 100/1
    shape_reg_weight = 0*2/10
    joint_weight = 1/10
    joint_2d_weight = 1/100
    vertex_weight = 0*1/100
    cls_weight = 1/1
    hand_type_weight = 0*100
    cam_trans_weight = 10/100/2
    cam_scale_weight = 1/100/2
    shape_weight = 5
    inter_joint_weight = 0*1/100
    joint_vec_weight = 1
    joint_2p5d_weight = 1000
    contact_pos_weight = 10
    contact_vis_weight = 10
    contact_attract_weight = 20

    obj_hm_weight = 1 / 1000
    obj_rot_weight = 0.7
    obj_trans_weight = 100/1
    obj_corner_weight = 1/10
    obj_corner_proj_weight = 0*1
    obj_weak_proj_weight = 0*1/10


    ## testing config
    test_batch_size = 32
    trans_test = 'gt' # gt, rootnet
    if predict_2p5d:
        if root_net_output_path is None:
            trans_test = 'gt'
        else:
            trans_test = 'rootnet'

    ## directory setup
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')

    def setup_out_dirs(self, model_dir_name):
        self.model_dir = osp.join(self.output_dir, 'model_dump', model_dir_name)
        self.tensorboard_dir = osp.join(self.output_dir, 'tensorboard', model_dir_name)


    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = True

    # date_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # model_dir = "{}_{}".format(model_dir, date_time)
    # tensorboard_dir = "{}_{}".format(tensorboard_dir, date_time)

    def set_args(self, gpu_ids, model_dir_name, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        self.model_dir_name = model_dir_name
        self.setup_out_dirs(model_dir_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

    def create_run_dirs(self):
        make_folder(cfg.model_dir)
        make_folder(cfg.tensorboard_dir)


cfg = Config()



sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.log_dir)
# fix_seeds(0)

