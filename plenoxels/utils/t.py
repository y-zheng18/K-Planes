import json
import numpy as np
import cv2
import os
import imageio

f = json.load(open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_train.json', 'r'))

f_new = f.copy()
# new_trasnform = [f['frames'][i]['transform_matrix'] for i in [0, 1, 2, 16]]
fluid_poses = np.load('/Users/yangzheng/code/project/smoke/data/poses_bounds.npy')
fluid_poses = fluid_poses[:, :-2].reshape((-1, 3, 5))
fluid_poses = fluid_poses[:, :3, :4]
fourth = np.array([0.,0,0,1])
fourth_tiled = np.tile(fourth, [len(fluid_poses),1,1])
fluid_poses = np.concatenate([fluid_poses, fourth_tiled], axis=1)
fluid_poses[:, :3, 3] = fluid_poses[:, :3, 3] * 3.5

scratch_dir = '/Users/yangzheng/code/project/smoke/fluid_debug'
# train set
new_transform = fluid_poses[1:].tolist()
f_new['frames'] = []
os.makedirs(os.path.join(scratch_dir, 'train'), exist_ok=True)
for i in range(4):
    video_path = os.path.join(scratch_dir, 'cam{}.mp4'.format(str(i + 1).zfill(2)))
    print('reading video from {}'.format(video_path))
    v = cv2.VideoCapture(video_path)

    # read video frames
    frames = []
    while (v.isOpened()):
        ret, frame = v.read()
        if not ret:
            break
        frames.append(frame)
    print('reading video done, {} frames'.format(len(frames)))
    for j in range(len(frames)):
        new_frame = f['frames'][0].copy()
        new_frame['transform_matrix'] = new_transform[i]
        new_frame['file_path'] = './train/r_{}'.format(str(j + i * 120).zfill(3))
        f_new['frames'].append(new_frame)
        cv2.imwrite(os.path.join(scratch_dir, 'train', 'r_{}'.format(str(j + i * 120).zfill(3)) + '.png'), frames[j])

json.dump(f_new, open(os.path.join(scratch_dir, 'transforms_train.json'), 'w'), indent=4)


# test set
new_transform = fluid_poses[:1].tolist()
f_new['frames'] = []
os.makedirs(os.path.join(scratch_dir, 'test'), exist_ok=True)
for i in range(1):
    video_path = os.path.join(scratch_dir, 'cam{}.mp4'.format(str(i).zfill(2)))
    v = cv2.VideoCapture(video_path)

    # read video frames
    frames = []
    while (v.isOpened()):
        ret, frame = v.read()
        if not ret:
            break
        frames.append(frame)
    assert len(frames) == 120
    for j in range(len(frames)):
        new_frame = f['frames'][0].copy()
        new_frame['transform_matrix'] = new_transform[i]
        new_frame['file_path'] = './test/r_{}'.format(str(j + i * 120).zfill(3))
        f_new['frames'].append(new_frame)
        cv2.imwrite(os.path.join(scratch_dir, 'test', 'r_{}'.format(str(j + i * 120).zfill(3)) + '.png'), frames[j])

json.dump(f_new, open(os.path.join(scratch_dir, 'transforms_test.json'), 'w'), indent=4)