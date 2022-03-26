{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import numpy as np\
import load_data as ld\
from SLAM_functions import *\
from copy import deepcopy\
\
\
#\
\
data_idx = 3 # change data_id for different datasets\
def read_data(idx):\
    joint = 'data/train_joint' + str(idx)\
    lidar = 'data/train_lidar' + str(idx)\
    joint_dict = ld.load_joint_data(joint)\
    lidar_dict = ld.load_lidar_data(lidar)\
\
    return lidar_dict,joint_dict\
#\
def map_t():\
    grid = \{\}\
    resolution=0.05\
    grid['resolution'] = 0.05\
    grid['size'] = 40  # m\
    xmin,xmax = -20,20\
    ymin,ymax = -20,20\
    szx = int(np.ceil((xmax - xmin) / resolution + 1))\
    szy = int(np.ceil((ymax - ymin) / resolution + 1))\
\
    cells = np.zeros((szx, szy), dtype=np.int8)\
    log_odds = np.zeros(cells.shape, dtype=np.float64)\
\
    grid['log_odds'] = log_odds\
    print(grid['log_odds'].shape,log_odds.shape)\
\
\
    occupied_prob_thresh = 0.6  #\
    grid['occ_d'] = np.log(occupied_prob_thresh / (1 - occupied_prob_thresh))\
    grid['free_d'] = np.log((1 - occupied_prob_thresh) / occupied_prob_thresh) * .5\
\
    grid['lidar_log_odds_occ'] = np.log(9)\
    grid['free_thres'] = np.log(1/9)\
    grid['bound'] = 5e6  # allow log odds recovery\
\
\
\
\
    return grid\
\
def init_sensor_model(lidar_data):\
\
\
    # get lidar angles\
    \
    # dmin is the minimum reading of the LiDAR, dmax is the maximum reading\
    lidar_dmin = 1e-3\
    lidar_dmax = 30\
    lidar_angular_resolution = 0.25\
    # these are the angles of the rays of the Hokuyo\
    lidar_angles = np.arange(-135, 135 + lidar_angular_resolution,\
                               lidar_angular_resolution) * np.pi / 180.0\
    lidar_angles=lidar_angles.reshape(1,-1)\
\
\
\
\
\
    return lidar_angles,lidar_dmin,lidar_dmax\
\
\
def init_particles(p=None,w=None):\
\
\
    n = 100\
    p = deepcopy(p) if p is not None else np.zeros((3, n), dtype=np.float64)\
    w = deepcopy(w) if w is not None else np.ones(n) / float(n)\
\
    particles = \{\}\
    particles['n'] = 100\
    particles['w'] = deepcopy(w) if w is not None else np.ones(n) / float(n)\
    particles['p'] =deepcopy(p) if p is not None else np.zeros((3, n), dtype=np.float64)\
    particles['Q'] = np.diag([2e-4,2e-4,1e-4])\
    particles['n_eff'] = .1 * particles['n']\
\
    return particles\
\
\
#\
\
\
def main():\
\
\
\
    lidar_data, joint_data = read_data(data_idx)\
    path = []\
\
    grid = map_t()\
\
    Particles = init_particles()\
\
    lidar_angles,lidar_dmin,lidar_dmax = init_sensor_model(lidar_data)\
\
\
\
\
\
\
    height, width = grid['log_odds'].shape\
\
    output = np.zeros((height,width,3),np.uint8)\
\
\
\
    for lidar_idx in range(0, len(lidar_data)):\
        Pose = Particles['p'][:, np.argmax(Particles['w'])]\
        path.append(np.copy(Pose))\
\
\
\
\
\
        joint_idx = np.argmin(np.abs(joint_data['ts']-lidar_data[lidar_idx]['t']))\
        joint_angles = joint_data['head_angles'][:,joint_idx]\
        scan_values = lidar_data[lidar_idx]['scan']\
        good_range = np.logical_and(scan_values > lidar_dmin, scan_values < lidar_dmax)\
        PH = to_grid(scan_values[good_range], lidar_angles[good_range])  # 3*n\
\
\
        lidar_hits = rays2world(PH, joint_angles, lidar_data[lidar_idx]['rpy'][0,:], pose=Pose)\
\
\
        grid_update(lidar_hits[:2], Pose[:2], grid)\
\
\
\
        if lidar_idx == 0:\
            continue\
\
\
        dynamic_step_from_odometry(Particles, lidar_data[lidar_idx]['pose'][0,:2], lidar_data[lidar_idx]['rpy'][0,2],\
                     lidar_data[lidar_idx-1]['pose'][0,:2], lidar_data[lidar_idx-1]['rpy'][0,2])\
\
\
        oberservation_step_update(Particles, grid, PH, joint_angles, lidar_data[lidar_idx]['rpy'][0,:])\
\
\
\
        if lidar_idx >= len(lidar_data)-1:\
            show_plots(grid, path, lidar_hits, output, idx=data_idx)\
\
        if lidar_idx % 100 == 0:\
            show_plots(grid, path, lidar_hits, output)\
\
\
def data_preprocess(joint_dir, lidar_dir):\
    joint_data = ld.load_joint_data(joint_dir)\
    lidar_data = ld.load_lidar_data(lidar_dir)\
\
\
    num_beams = lidar_data[0]['scan'].shape[1]\
    lidar_angles = np.linspace(start=-135*np.pi/180, stop=135*np.pi/180, num=num_beams).reshape(1,-1)\
\
\
    yaw_bias = lidar_data[0]['rpy'][0,2]\
    pose_bias = lidar_data[0]['pose'][0,:2]\
    for i in range(len(lidar_data)):\
        lidar_data[i]['rpy'][0,2] -= yaw_bias\
        lidar_data[i]['pose'][0,:2] -= pose_bias\
\
    return joint_data, lidar_data, lidar_angles\
\
\
\
if __name__ == '__main__':\
    main()\
\
\
\
\
}