import json
import numpy as np


f = json.load(open('/Users/yangzheng/Downloads/data/hellwarrior/transforms_train.json', 'r'))
print(len(f['frames']))
rt = f['frames'][0]['transform_matrix']
print(rt)
poses = np.stack([np.array(f['frames'][i]['transform_matrix']) for i in range(len(f['frames']))])
print(poses.shape)