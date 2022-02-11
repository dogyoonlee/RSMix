
import os
import sys
import numpy as np
import argparse
import random


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


# def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
def random_scale_point_cloud(batch_data, scale_low=2./3., scale_high=3./2.):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        # dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        dropout_ratio = 0.95
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data_mixed', help='mixed data dir [default: ./data_mixed]')

    args = parser.parse_args()

    np.random.seed(100)
    # c_background = (255, 255, 255)
    point_set = np.loadtxt(args.path ,delimiter=',').astype(np.float32)
    point_set = np.expand_dims(point_set[:,:3], 0)
    point_set_drop = point_set.copy()
    point_set_jit = point_set.copy()
    
    point_set_drop = random_point_dropout(point_set_drop)
    point_set_drop = np.squeeze(point_set_drop)
    filename = 'data_drop.txt'
    save_file_path = os.path.join('./data', filename)
    np.savetxt(save_file_path, point_set_drop, fmt='%.6f', delimiter=',') 
    
    point_set_jit = jitter_point_cloud(point_set_jit)
    point_set_jit = np.squeeze(point_set_jit)
    filename = 'data_jit.txt'
    save_file_path = os.path.join('./data',filename)
    np.savetxt(save_file_path, point_set_jit, fmt='%.6f', delimiter=',') 
    