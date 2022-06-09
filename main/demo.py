# Copyright (c) 2020 Graz University of Technology All rights reserved.


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from common.base import Tester


from common.utils.preprocessing import PeakDetector, load_skeleton
from common.utils.transforms import rot_param_rot_mat, rot_param_rot_mat_np
import smplx
from common.utils.vis import *
import configargparse


jointsMapManoToDefault = [
                         16, 15, 14, 13,
                         17, 3, 2, 1,
                         18, 6, 5, 4,
                         19, 12, 11, 10,
                         20, 9, 8, 7,
                         0]

VIS_ATTN = False

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--annot_subset', type=str, dest='annot_subset', default='all',
                        help='all/human_annot/machine_annot')
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


    mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
                       'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(
            mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        mano_layer['left'].shapedirs[:, 0, :] *= -1

    ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
    ih26m_joint_regressor = torch.FloatTensor(ih26m_joint_regressor).unsqueeze(0)  # 1 x 21 x 778

    ortho_rend = OrthographicRender()
    skeleton = load_skeleton(cfg.skeleton_file, 42)
    peak_detector = PeakDetector()
    if cfg.has_object:
        ycb_objs = H2O3DObjects()



    fig, ax = plt.subplots(nrows=2, ncols=2)
    # fig.set_figheight(15)
    # fig.set_figwidth(15)
    plt.waitforbuttonpress(0.01)
    with torch.no_grad():
       for itr, (inputs, targets, meta_info) in enumerate(tester.batch_generator):

            # forward
            model_out = tester.model(inputs, targets, meta_info, 'test', epoch_cnt=1e8)

            out = {k[:-4]: model_out[k] for k in model_out.keys() if '_out' in k}


            # Get all the outputs from the model
            heatmap_np = out['joint_heatmap'].cpu().numpy()
            hand_type_np = targets['hand_type'].cpu().numpy() # N

            if 'cam_param' in out.keys():
                cam_param = out['cam_param']

            rel_trans_np = out['rel_trans'].detach().cpu().numpy()

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
                joints_2d_right = joints_right[:,:,:2] * cam_param[:, :1].unsqueeze(1).cpu() + cam_param[:, 1:].unsqueeze(1).cpu()
                mano_joints_right_gt = torch.matmul(ih26m_joint_regressor, targets['verts'][:,:778])

                if cfg.hand_type in ['left', 'both']:
                    mesh_left = mano_layer['left'](
                        global_orient=out['pose'][-1].permute(1, 0, 2)[:, start_indx].to(torch.device('cpu')),  # N x 32 x 3
                        hand_pose=out['pose'][-1].permute(1, 0, 2)[:, (start_indx+1):].reshape(-1, 45).to(torch.device('cpu')),  # N x 32 x 3
                        betas=out['shape'][-1].to(torch.device('cpu')),  # N x 10
                        transl=out['rel_trans'].to(torch.device('cpu')))#*meta_info['root_valid']) # N x 3
                    joints_left = torch.matmul(ih26m_joint_regressor, mesh_left.vertices)  # N x 21 x 3
                else:
                    joints_left = torch.zeros((inputs['img'].shape[0], 21, 3))
                joints_2d_left = joints_left[:,:,:2] * cam_param[:, :1].unsqueeze(1).cpu() + cam_param[:, 1:].unsqueeze(1).cpu()
                mano_joints_left_gt = torch.matmul(ih26m_joint_regressor, targets['verts'][:,778:])

            elif cfg.predict_type == 'vectors':
                if not cfg.predict_2p5d:
                    root_joint = torch.zeros((out['joint_3d_right'].shape[1],1,3)).to(out['joint_3d_right'].device)
                    if cfg.hand_type in ['right', 'both']:
                        joints_right = out['joint_3d_right'][-1]
                        joints_right = torch.cat([joints_right, root_joint], dim = 1)
                        joints_2d_right = joints_right[:,:,:2]*cam_param[:, :1].unsqueeze(1) + cam_param[:,1:].unsqueeze(1)

                    if cfg.hand_type in ['left', 'both']:
                        joints_left = out['joint_3d_left'][-1]
                        joints_left = torch.cat([joints_left, root_joint], dim=1)
                        joints_2d_left = (joints_left + out['rel_trans'].unsqueeze(1))[:,:,:2] * cam_param[:, :1].unsqueeze(1) + cam_param[:, 1:].unsqueeze(1)
                else:
                    joints_2d_right = out['joint_2p5d'][:, :21]
                    joints_2d_left = out['joint_2p5d'][:, 21:]

            if cfg.has_object:
                pred_obj_corners_all = []
                obj_corners_proj_pred_all = []
                for ii in range(heatmap_np.shape[0]):
                    if cfg.use_obj_rot_parameterization:
                        rot_mat = rot_param_rot_mat(out['obj_rot'][ii:ii + 1].reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
                    else:
                        rot_mat = cv2.Rodrigues(out['obj_rot'][ii].cpu().numpy())[0]
                    pred_obj_corners = meta_info['obj_bb_rest'][ii].cpu().numpy().dot(rot_mat.T) \
                                       + out['obj_trans'][ii].cpu().numpy()
                    obj_corners_proj_pred = pred_obj_corners[:, :2] * cam_param[ii:ii + 1, :1].cpu().numpy() \
                                            + cam_param[ii:ii + 1, 1:].cpu().numpy()
                    pred_obj_corners_all.append(pred_obj_corners)
                    obj_corners_proj_pred_all.append(obj_corners_proj_pred)
                pred_obj_corners_all = np.stack(pred_obj_corners_all, axis=0)
                obj_corners_proj_pred_all = np.stack(obj_corners_proj_pred_all, axis=0)



            # Start visualizing each image in the batch
            for ii in range(heatmap_np.shape[0]):

                for r in range(2):
                    for c in range(2):
                        ax[r,c].clear()

                # Display the predicted heatmap
                ax[0, 0].imshow(heatmap_np[ii])
                ax[0, 0].set_title('Predicted Heatmap')
                ax[0, 0].set_xticks([])
                ax[0, 0].set_yticks([])

                # Display the peaks selected from the heatmap after NMS
                peaks, _ = peak_detector.detect_peaks_nms(heatmap_np[ii], cfg.max_num_peaks)
                ax[0, 1].imshow(peaks)
                ax[0, 1].set_title('Peaks Selected by the NMS')
                ax[0, 1].set_xticks([])
                ax[0, 1].set_yticks([])

                # Get the object corner projections if object is in the image
                img = (np.transpose(inputs['img'][ii].cpu().numpy(), [1, 2, 0]) * 255).astype(np.uint8)
                img = cv2.resize(img, (128, 128))
                if cfg.has_object:
                    img_2d = vis_2d_obj_corners(img, obj_corners_proj_pred_all[ii], lineThickness=1)
                else:
                    img_2d = img.copy()

                # Get the joint projections for both the hands
                if hand_type_np[ii, 0] > 0.5:
                    img_2d = vis_keypoints_new(img_2d, joints_2d_right.cpu().numpy()[ii], np.ones((21)), skeleton, line_width=1, circle_rad=1.5, hand_type='right')
                else:
                    img_2d = img.copy()
                if hand_type_np[ii,1]>0.5 and cfg.dataset != 'ho3d':
                    img_2d = vis_keypoints_new(img_2d, joints_2d_left.cpu().numpy()[ii], np.ones((21)), skeleton, line_width=1, circle_rad=1.5, hand_type='left')

                # Display the projected joint locations
                ax[1, 0].imshow(img_2d)
                ax[1, 0].set_title('Predicted Joint Projections')
                ax[1, 0].set_xticks([])
                ax[1, 0].set_yticks([])



                if cfg.predict_type == 'angles':

                    # Get the left and right hand predicted and GT MANO meshes
                    mesh_out_list = []
                    if hand_type_np[ii,0]>0.5:
                        # visualize mesh in open3d
                        mesh_right_o3d = o3d.geometry.TriangleMesh()
                        mesh_right_o3d.vertices = o3d.utility.Vector3dVector(mesh_right.vertices[ii].numpy())
                        mesh_right_o3d.triangles = o3d.utility.Vector3iVector(mano_layer['right'].faces)
                        mesh_out_list.append(mesh_right_o3d)

                        mesh_right_gt_o3d = o3d.geometry.TriangleMesh()
                        mesh_right_gt_o3d.vertices = o3d.utility.Vector3dVector(targets['verts'][ii][:778].cpu().detach().numpy())
                        mesh_right_gt_o3d.triangles = o3d.utility.Vector3iVector(mano_layer['right'].faces)


                    if hand_type_np[ii, 1] > 0.5:
                        mesh_left_o3d = o3d.geometry.TriangleMesh()

                        mesh_left_o3d.vertices = o3d.utility.Vector3dVector(mesh_left.vertices[ii].numpy())

                        mesh_left_o3d.triangles = o3d.utility.Vector3iVector(mano_layer['left'].faces)
                        mesh_out_list.append(mesh_left_o3d)

                        mesh_left_gt_o3d = o3d.geometry.TriangleMesh()
                        mesh_left_gt_o3d.vertices = o3d.utility.Vector3dVector(
                            targets['verts'][ii][778:].cpu().detach().numpy())
                        mesh_left_gt_o3d.triangles = o3d.utility.Vector3iVector(mano_layer['left'].faces)

                    mesh_out = mesh_out_list[0]
                    for mesh in mesh_out_list[1:]:
                        mesh_out = mesh_out + mesh

                    # Orthographic projection of the meshed onto the image using the predicted camera parameters
                    render, mask = ortho_rend.render(out['cam_param'][ii].cpu().detach().numpy(), mesh_out)
                    mask = np.expand_dims(mask, 2)
                    img_overlay = img * np.logical_not(mask) + render * mask

                    # Display the mesh overlayed image
                    ax[1, 1].imshow(img_overlay)
                    ax[1, 1].set_title('Predicted Meshes')
                    ax[1, 1].set_xticks([])
                    ax[1, 1].set_yticks([])
                elif cfg.predict_type == 'vectors' and not cfg.predict_2p5d and False:
                    # This part of the code visualizes the joints and object corners in 3D. Disable the other matplotlib window
                    # to use this
                    joints_3d_list = []
                    if hand_type_np[ii, 0] > 0.5:
                        joints_3d_list.append(joints_right[ii].cpu().numpy())
                    if hand_type_np[ii, 1] > 0.5:
                        joints_3d_list.append(joints_left[ii].cpu().numpy()+rel_trans_np[ii])
                    joints_3d = np.concatenate(joints_3d_list, axis=0)

                    # Visualize the joint location in 3D
                    if len(joints_3d) == 21:
                        skeleton_vis = skeleton[:21]
                    else:
                        skeleton_vis = skeleton
                    ax_3d = vis_3d_keypoints_new(joints_3d, np.ones((joints_3d.shape[0])), skeleton_vis, plot=False)
                    if cfg.has_object:
                        pred_obj_corners = pred_obj_corners_all[ii]

                        gt_obj_corners = meta_info['obj_bb_rest'][ii].cpu().numpy().dot(cv2.Rodrigues(targets['obj_rot'][ii].cpu().numpy())[0].T)\
                                         + targets['rel_obj_trans'][ii].cpu().numpy()

                        # Visualize the object corners in 3D
                        vis_3d_obj_corners([pred_obj_corners], ax=ax_3d)

                # Show the predicted object seg map
                if cfg.has_object:
                    ax[1, 1].imshow(out['obj_seg_pred'][ii].cpu().numpy())
                    ax[1, 1].set_title('Predicted Object Seg ')
                    ax[1, 1].set_xticks([])
                    ax[1, 1].set_yticks([])

                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('Seq name: %s, Frame ID: %d'%(meta_info['seq_id'][ii], meta_info['frame'][ii].cpu().numpy()))
                print('Press \'N\' on the matplotlib window')

                bpress = plt.waitforbuttonpress()
                while (not bpress):
                    bpress = plt.waitforbuttonpress()


                # Get the object mesh
                obj_mesh_list = []
                if cfg.has_object:
                    rot_z_mat = np.eye(4)
                    rot_z_mat[:3,:3] = cv2.Rodrigues(np.array([0,1,0])*np.pi)[0]

                    obj_id = int(meta_info['obj_id'][ii].cpu().numpy())
                    obj_mesh = deepcopy(ycb_objs.obj_id_to_mesh[obj_id])
                    if cfg.use_obj_rot_parameterization:
                        rot_mat = rot_param_rot_mat(out['obj_rot'][ii:ii + 1].reshape(-1, 6))[0].cpu().numpy()  # 3 x 3
                    else:
                        rot_mat = cv2.Rodrigues(out['obj_rot'][ii].cpu().numpy())[0]

                    trans_mat = np.eye(4)
                    trans_mat[:3,:3] = rot_mat#.dot(aa)
                    trans_mat[:3,3] = out['obj_trans'][ii].cpu().numpy()
                    obj_mesh.transform(rot_z_mat)
                    obj_mesh.transform(trans_mat)
                    obj_mesh_list.append(obj_mesh)



                if cfg.predict_type == 'angles':
                    print('Showing Open3d Vis.....')
                    print('Press \'Q\' on the open3d window to go to the next image...')
                    o3d.visualization.draw_geometries([
                                                       mesh_right_o3d,
                                                       # mesh_right_gt_o3d,
                                                       mesh_left_o3d,
                                                       # mesh_left_gt_o3d,
                                                       ]+obj_mesh_list,
                                                      mesh_show_back_face=True)











if __name__ == "__main__":
    main()
