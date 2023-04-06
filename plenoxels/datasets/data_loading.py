from typing import Tuple, Optional, Dict, Any, List
import logging as log
import os
import resource

import torch
from torch.multiprocessing import Pool
import torchvision.transforms
from PIL import Image
import imageio.v3 as iio
import numpy as np
import json

from plenoxels.utils.my_tqdm import tqdm

pil2tensor = torchvision.transforms.ToTensor()
# increase ulimit -n (number of open files) otherwise parallel loading might fail
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (16192, rlimit[1]))


def _load_phototourism_image(idx: int,
                             paths: List[str],
                             out_h: List[int],
                             out_w: List[int]) -> torch.Tensor:
    f_path = paths[idx]
    img = Image.open(f_path).convert('RGB')
    img.resize((out_w[idx], out_h[idx]), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]
    return img


def _parallel_loader_phototourism_image(args):
    torch.set_num_threads(1)
    return _load_phototourism_image(**args)


def _load_llff_image(idx: int,
                     paths: List[str],
                     data_dir: str,
                     out_h: int,
                     out_w: int,
                     ) -> torch.Tensor:
    f_path = os.path.join(data_dir, paths[idx])
    img = Image.open(f_path).convert('RGB')

    img = img.resize((out_w, out_h), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]
    img = img.permute(1, 2, 0)  # [H, W, C]
    return img


def _parallel_loader_llff_image(args):
    torch.set_num_threads(1)
    return _load_llff_image(**args)


def _load_nerf_image_pose(idx: int,
                          frames: List[Dict[str, Any]],
                          data_dir: str,
                          out_h: Optional[int],
                          out_w: Optional[int],
                          downsample: float,
                          ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    # Fix file-path
    f_path = os.path.join(data_dir, frames[idx]['file_path'])
    if '.' not in os.path.basename(f_path):
        f_path += '.png'  # so silly...
    if not os.path.exists(f_path):  # there are non-exist paths in fox...
        return None
    img = Image.open(f_path)
    if out_h is None:
        out_h = int(img.size[0] / downsample)
    if out_w is None:
        out_w = int(img.size[1] / downsample)
    # Now we should downsample to out_h, out_w and low-pass filter to resolution * 2.
    # We only do the low-pass filtering if resolution * 2 is lower-res than out_h, out_w

    img = img.resize((out_h, out_w), Image.LANCZOS)
    img = pil2tensor(img)  # [C, H, W]

    img = img.permute(1, 2, 0)  # [H, W, C]
    if img.shape[0] > img.shape[1]:
        # zero pad to square
        pad = (img.shape[0] - img.shape[1]) // 2
        zeros = torch.zeros((img.shape[0], pad, img.shape[2]), dtype=torch.float32)
        img = torch.cat([zeros, img, zeros], dim=1)
    elif img.shape[1] > img.shape[0]:
        # zero pad to square
        pad = (img.shape[1] - img.shape[0]) // 2
        zeros = torch.zeros((pad, img.shape[1], img.shape[2]), dtype=torch.float32)
        img = torch.cat([zeros, img, zeros], dim=0)
    if img.shape[2] == 3:
        mask = img.sum(dim=2) > 0.08
        mask = mask.float()
        mask = mask.unsqueeze(2)
        # img = img * mask + (1 - mask) * 1
        img = torch.cat([img, mask], dim=2)
        # img = 1 - img
    assert img.shape[0] == img.shape[1]
    # assert img.shape[2] == 4
    pose = torch.tensor(frames[idx]['transform_matrix'], dtype=torch.float32)

    return (img, pose)

def load_pinf_frame_data(basedir, half_res=False, split='train'):
    # frame data
    all_imgs = []
    all_poses = []

    with open(os.path.join(basedir, 'info.json'), 'r') as fp:
        # read render settings
        meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]],
                              axis=1)  # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'], [3])

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num

        video_list = meta[split + '_videos'] if (split + '_videos') in meta else meta['train_videos'][0:1]

        for video_id, train_video in enumerate(video_list):
            imgs = []

            f_name = os.path.join(basedir, train_video['file_name'])
            reader = imageio.get_reader(f_name, "ffmpeg")
            for frame_i in range(train_video['frame_num']):
                reader.set_image_index(frame_i)
                frame = reader.get_next_data()

                H, W = frame.shape[:2]
                camera_angle_x = float(train_video['camera_angle_x'])
                Focal = .5 * W / np.tan(.5 * camera_angle_x)
                imgs.append(frame)

            reader.close()
            imgs = (np.float32(imgs) / 255.)

            if half_res:
                H = H // 2
                W = W // 2
                Focal = Focal / 2.

                imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[-1]))
                for i, img in enumerate(imgs):
                    imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs = imgs_half_res

            all_imgs.append(imgs)
            all_poses.append(np.array(
                train_video['transform_matrix_list'][frame_i]
                if 'transform_matrix_list' in train_video else train_video['transform_matrix']
            ).astype(np.float32))

    imgs = np.stack(all_imgs, 0)  # [V, T, H, W, 3]
    imgs = np.transpose(imgs, [1, 0, 2, 3, 4])  # [T, V, H, W, 3]
    poses = np.stack(all_poses, 0)  # [V, 4, 4]
    hwf = np.float32([H, W, Focal])

    # set render settings:
    sp_n = 120  # an even number!
    sp_poses = [
        pose_spherical(angle, phi, radius, rotZ, r_center[0], r_center[1], r_center[2])
        for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
    ]
    render_poses = torch.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
    render_timesteps = np.arange(sp_n) / (sp_n - 1)

    return imgs, poses, hwf, render_poses, render_timesteps, voxel_tran, voxel_scale, near, far

def _parallel_loader_nerf_image_pose(args):
    torch.set_num_threads(1)
    return _load_nerf_image_pose(**args)


def _load_video_1cam(idx: int,
                     paths: List[str],
                     poses: torch.Tensor,
                     out_h: int,
                     out_w: int,
                     load_every: int = 1
                     ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    filters = [
        ("scale", f"w={out_w}:h={out_h}")
    ]
    all_frames = iio.imread(
        paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)
    imgs, timestamps = [], []
    for frame_idx, frame in enumerate(all_frames):
        if frame_idx % load_every != 0:
            continue
        if frame_idx >= 300:  # Only look at the first 10 seconds
            break
        # Frame is np.ndarray in uint8 dtype (H, W, C)
        imgs.append(
            torch.from_numpy(frame)
        )
        timestamps.append(frame_idx)
    imgs = torch.stack(imgs, 0)
    med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
    return (imgs,
            poses[idx].expand(len(timestamps), -1, -1),
            med_img,
            torch.tensor(timestamps, dtype=torch.int32))


def _parallel_loader_video(args):
    torch.set_num_threads(1)
    return _load_video_1cam(**args)


def parallel_load_images(tqdm_title,
                         dset_type: str,
                         num_images: int,
                         **kwargs) -> List[Any]:
    max_threads = 10
    if dset_type == 'llff':
        fn = _parallel_loader_llff_image
    elif dset_type == 'synthetic':
        fn = _parallel_loader_nerf_image_pose
    elif dset_type == 'phototourism':
        fn = _parallel_loader_phototourism_image
    elif dset_type == 'video':
        fn = _parallel_loader_video
        # giac: Can increase to e.g. 10 if loading 4x subsampled images. Otherwise OOM.
        max_threads = 8
    else:
        raise ValueError(dset_type)
    p = Pool(min(max_threads, num_images))

    iterator = p.imap(fn, [{"idx": i, **kwargs} for i in range(num_images)])
    outputs = []
    for _ in tqdm(range(num_images), desc=tqdm_title):
        out = next(iterator)
        if out is not None:
            outputs.append(out)
    return outputs
