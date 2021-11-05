import open3d as o3d
import argparse
import cv2 
import numpy as np
import glob
import copy
import matplotlib.pyplot as plt
from scipy.spatial.qhull import Voronoi
from scipy.spatial.transform import Rotation as R
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# set intrincsic function
WIDTH = 512
HEIGHT = 512
FOV = 0.5 * np.pi 
FX = WIDTH / 2 * np.tan(FOV / 2)
FY = HEIGHT / 2 * np.tan(FOV / 2)
CX = WIDTH / 2
CY = HEIGHT / 2
INTRINSIC = [WIDTH, HEIGHT, FX, FY, CX, CY]

# util function
def get_transform_by_quaternion_and_pos(pose):

    # extract info
    [x, y, z, rw, rx, ry, rz] = pose

    # convert quaternion to rotation
    m00 = 1 - 2 * (ry * ry + rz * rz)
    m01 = 2 * (rx * ry - rw * rz)
    m02 = 2 * (rx * rz + rw * ry)

    m10 = 2 * (rx * ry + rw * rz)
    m11 = 1 - 2 * (rx * rx + rz * rz)
    m12 = 2 * (ry * rz - rw * rx)

    m20 = 2 * (rx * rz - rw * ry)
    m21 = 2 * (ry * rz + rw * rx)
    m22 = 1 - 2 * (rx * rx + ry * ry)

    # transform matrix
    transform = np.array([[m00, m01, m02,   x],
                          [m10, m11, m12,   y],
                          [m20, m21, m22,   z],
                          [  0,   0,   0,   1]]).astype(np.float64)

    return transform

# major function
def depth_image_to_point_cloud(rgb, depth, intrinsic):

    adjust = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0,-1, 0],
                    [0, 0, 0, 1]])

    [w, h, fx, fy, cx, cy] = intrinsic

    ix, iy =  np.meshgrid(range(w), range(h))
    
    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel()
    x = z * x_ratio
    y = z * y_ratio

    points = np.vstack((x, y, z)).T
    colors = np.reshape(rgb,(512 * 512, 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(adjust)

    return pcd

def get_all_pcds(num=-1, semantic_gt_path=None, semantic_other_path=None, semantic_apartment0_path=None):
    # get point clouds
    num = len(glob.glob('{}/*.png'.format(semantic_gt_path))) if num == -1 else num
    pose_file = open(pose_path, 'r')
    pcds_gt = []
    pcds_other = []
    pcds_apartment0 = []
    poses = []
    for id in range(num):

        path_id = 1 + id

        # get pose info from corresponding image in file
        data_raw = pose_file.readline()
        pose_raw = np.asarray([np.float64(x) for x in str(data_raw).split('\n')[0].split()])
        pose = get_transform_by_quaternion_and_pos(pose_raw)

        # read image file and get point cloud
        demo_color_gt = cv2.imread('{}{}.png'.format(semantic_gt_path, path_id ))
        demo_color_other = cv2.imread('{}{}.png'.format(semantic_other_path, path_id ))
        demo_color_apartment0 = cv2.imread('{}{}.png'.format(semantic_apartment0_path, path_id ))
        demo_depth = np.array(Image.open('{}{}.png'.format(depth_path, path_id ))).astype(np.float32) / 65535.0 * 10.0
        pcd_gt = depth_image_to_point_cloud(demo_color_gt, demo_depth, [WIDTH, HEIGHT, FX, FY, CX, CY])
        pcd_other = depth_image_to_point_cloud(demo_color_other, demo_depth, [WIDTH, HEIGHT, FX, FY, CX, CY])
        pcd_apartment0 = depth_image_to_point_cloud(demo_color_apartment0, demo_depth, [WIDTH, HEIGHT, FX, FY, CX, CY])

        print('{}{}.png'.format(semantic_gt_path, path_id), pcd_gt)
        print('{}{}.png'.format(semantic_other_path, path_id), pcd_other)
        print('{}{}.png'.format(semantic_apartment0_path, path_id), pcd_apartment0)
        
        # write to buffer
        poses.append(pose)
        pcds_gt.append(pcd_gt)
        pcds_other.append(pcd_other)
        pcds_apartment0.append(pcd_apartment0)

    return pcds_gt, pcds_other, pcds_apartment0, poses

def voxel_down_sample(pcd, voxel_size):
    
    pts = np.asarray(pcd.points)
    clrs = np.asarray(pcd.colors)

    pts_round = np.round(np.array(pts / voxel_size)).astype(int)
    pts_down = np.unique(pts_round, axis=0) *  voxel_size

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(pts)
    _, idx = nn.kneighbors(pts_down)
    clrs_down = clrs[idx]
    clrs_down = np.reshape(clrs_down, (clrs_down.shape[0], clrs_down.shape[2]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_down)
    pcd.colors = o3d.utility.Vector3dVector(clrs_down)

    return pcd

# constant config
parser = argparse.ArgumentParser()
parser.add_argument('--floor', '-f', required=True, type=int)
args = parser.parse_args()

assert args.floor == 1 or args.floor == 2

FLOOR = args.floor - 1 # 0: first, 1: second
CURRENT_DIR = ['first_floor', 'second_floor',]

pose_path = 'Data_collection/{}/GT_Pose.txt'.format(CURRENT_DIR[FLOOR])
semantic_gt_path = 'Data_collection/{}/semantic_gt/'.format(CURRENT_DIR[FLOOR])
semantic_other_path = 'Data_collection/{}/semantic_1/'.format(CURRENT_DIR[FLOOR])
semantic_apartment0_path = 'Data_collection/{}/semantic_2/'.format(CURRENT_DIR[FLOOR])
depth_path = 'Data_collection/{}/depth/'.format(CURRENT_DIR[FLOOR])

VOXEL_SIZE = 0.05

# get all point cloud
pcds_gt, pcds_other, pcds_apartment0, transforms = get_all_pcds(-1, semantic_gt_path, semantic_other_path, semantic_apartment0_path)

print("Full registration ...")

height_thresh_list = [0.75, 0.75]

# multiway registration
for point_id in range(len(pcds_gt)):

    # remove roof
    print('processing {} ...'.format(point_id))
    height_thresh = height_thresh_list[FLOOR]
    # gt
    pcds_gt[point_id] = voxel_down_sample(pcds_gt[point_id], voxel_size=VOXEL_SIZE)
    points_o3d = np.array(pcds_gt[point_id].points)
    colors_o3d = np.array(pcds_gt[point_id].colors)
    cond_o3d = np.where(points_o3d[:,1] > height_thresh)
    pcd_without_roof = np.delete(points_o3d, cond_o3d, axis=0)
    pcd_colors = np.delete(colors_o3d, cond_o3d, axis=0)
    pcds_gt[point_id].points = o3d.utility.Vector3dVector(pcd_without_roof)
    pcds_gt[point_id].colors = o3d.utility.Vector3dVector(pcd_colors)
    pcds_gt[point_id].transform(transforms[point_id])
    # other
    pcds_other[point_id] = voxel_down_sample(pcds_other[point_id], voxel_size=VOXEL_SIZE)
    points_o3d = np.array(pcds_other[point_id].points)
    colors_o3d = np.array(pcds_other[point_id].colors)
    cond_o3d = np.where(points_o3d[:,1] > height_thresh)
    pcd_without_roof = np.delete(points_o3d, cond_o3d, axis=0)
    pcd_colors = np.delete(colors_o3d, cond_o3d, axis=0)
    pcds_other[point_id].points = o3d.utility.Vector3dVector(pcd_without_roof)
    pcds_other[point_id].colors = o3d.utility.Vector3dVector(pcd_colors)
    pcds_other[point_id].transform(transforms[point_id])
    # apartment0
    pcds_apartment0[point_id] = voxel_down_sample(pcds_apartment0[point_id], voxel_size=VOXEL_SIZE)
    points_o3d = np.array(pcds_apartment0[point_id].points)
    colors_o3d = np.array(pcds_apartment0[point_id].colors)
    cond_o3d = np.where(points_o3d[:,1] > height_thresh)
    pcd_without_roof = np.delete(points_o3d, cond_o3d, axis=0)
    pcd_colors = np.delete(colors_o3d, cond_o3d, axis=0)
    pcds_apartment0[point_id].points = o3d.utility.Vector3dVector(pcd_without_roof)
    pcds_apartment0[point_id].colors = o3d.utility.Vector3dVector(pcd_colors)
    pcds_apartment0[point_id].transform(transforms[point_id])


o3d.visualization.draw_geometries(pcds_gt, zoom=0.3412, front=[0, 1, 0], lookat=[0, 10, 5], up=[1, 0, 0])
o3d.visualization.draw_geometries(pcds_other, zoom=0.3412, front=[0, 1, 0], lookat=[0, 10, 5], up=[1, 0, 0])
o3d.visualization.draw_geometries(pcds_apartment0, zoom=0.3412, front=[0, 1, 0], lookat=[0, 10, 5], up=[1, 0, 0])
print('process completed.')
