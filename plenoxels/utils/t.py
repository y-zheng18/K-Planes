import json
import numpy as np


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

new_transform = fluid_poses[1:].tolist()
f_new['frames'] = []
for i in range(4):
    for j in range(100):
        new_frame = f['frames'][j].copy()
        new_frame['transform_matrix'] = new_transform[i]
        new_frame['file_path'] = './train/r_{}'.format(str(j + i * 100).zfill(3))
        f_new['frames'].append(new_frame)

json.dump(f_new, open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_train_fluid.json', 'w'), indent=4)

#
f_val = json.load(open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_val.json', 'r'))
new_transform = fluid_poses[:1].tolist()

f_val_new = f_val.copy()
f_val_new['frames'] = []
for i in range(1):
    for j in range(100):
        new_frame = f['frames'][j].copy()
        new_frame['transform_matrix'] = new_transform[i]
        new_frame['file_path'] = './val/r_{}'.format(str(j + i * 100).zfill(3))
        f_val_new['frames'].append(new_frame)
json.dump(f_val_new, open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_val_fluid.json', 'w'), indent=4)
#
#
# f_test = json.load(open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_test.json', 'r'))
# new_transform = [f_test['frames'][i]['transform_matrix'] for i in range(1)]
#
# f_test_new = f_test.copy()
# f_test_new['frames'] = []
# for i in range(1):
#     for j in range(100):
#         new_frame = f['frames'][j].copy()
#         new_frame['transform_matrix'] = new_transform[i]
#         new_frame['file_path'] = './val/r_{}'.format(str(j + i * 100).zfill(3))
#         f_test_new['frames'].append(new_frame)
#
# json.dump(f_test_new, open('/Users/yangzheng/code/project/smoke/test_fluid/transforms_test_1.json', 'w'), indent=4)