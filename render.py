import numpy as np
import json
import os
import math
import argparse
from typing import Any, Dict, Optional, Sequence, Union
import mathutils
import bpy
import time
import sys
import glob
from mathutils import *
from math import *


RESULOUTION_X = 800
RESULOUTION_Y = 800


# randomize np seed using time
np.random.seed(int(time.time()))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Blender_render():
    def __init__(self,
                 scratch_dir=None,
                 render_engine='BLENDER_EEVEE',
                 split='train',
                 use_gpu: bool = False,
                 ):
        self.blender_scene = bpy.context.scene
        self.render_engine = render_engine
        self.use_gpu = use_gpu

        self.set_render_engine()

        self.scratch_dir = scratch_dir
        self.split = split
        self.meta = json.load(open(os.path.join(scratch_dir, 'transforms_{}.json').format(split), 'r'))
        custom_scene = os.path.join(scratch_dir, 'scene.blend')

        assert custom_scene is not None and os.path.exists(custom_scene)
        print("Loading scene from '%s'" % custom_scene)
        bpy.ops.wm.open_mainfile(filepath=custom_scene)
        self.setup_scene()


    def set_render_engine(self):
        bpy.context.scene.render.engine = self.render_engine
        print("Using render engine: {}".format(self.render_engine))
        if self.use_gpu:
            print("----------------------------------------------")
            print('setting up gpu ......')

            bpy.context.scene.cycles.device = "GPU"
            for scene in bpy.data.scenes:
                print(scene.name)
                scene.cycles.device = 'GPU'

            # if cuda arch use cuda, else use metal
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

            bpy.context.preferences.addons["cycles"].preferences.get_devices()
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d.use = True
                print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
            print('setting up gpu done')
            print("----------------------------------------------")

    def setup_scene(self):
        bpy.ops.object.camera_add()
        self.camera = bpy.data.objects["Camera"]

        self.camera.data.clip_end = 10000

        # setup scene
        bpy.context.scene.render.resolution_x = RESULOUTION_X
        bpy.context.scene.render.resolution_y = RESULOUTION_Y
        bpy.context.scene.render.resolution_percentage = 100

    def clear_scene(self):
        for k in bpy.data.objects.keys():
            bpy.data.objects[k].select_set(False)

    def render(self):
        """Renders all frames (or a subset) of the animation.
        """
        print("Using scratch rendering folder: '%s'" % self.scratch_dir)
        # setup rigid world cache

        # --- starts rendering
        camdata = self.camera.data
        focal = camdata.lens  # mm


        self.set_render_engine()
        self.clear_scene()

        bpy.context.scene.frame_end = 100

        absolute_path = os.path.abspath(self.scratch_dir)

        os.makedirs(os.path.join(self.scratch_dir, self.split), exist_ok=True)
        for i, frame_nr in enumerate(self.meta['frames']):
            timestamp = int(frame_nr['time'] * 99 + 1)
            bpy.context.scene.frame_set(timestamp)
            camera_transform = frame_nr['transform_matrix']
            print(camera_transform)
            mat = Matrix(camera_transform)
            print(mat)
            self.camera.matrix_world = mat
            print('setting camera for frame %d' % timestamp)
            print(os.path.join(
                self.scratch_dir, self.split, f"r_{i:03d}.png"))
            bpy.context.scene.render.filepath = os.path.join(
                self.scratch_dir, self.split, f"r_{i:03d}.png")

            bpy.ops.render.render(animation=False, write_still=True)

            print("Rendered frame '%s'" % bpy.context.scene.render.filepath)


if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render scene.')
    parser.add_argument('--scratch_dir', type=str, metavar='PATH', default='/Users/yangzheng/code/project/smoke/test_fluid_cam/')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--render_engine', type=str, default='BLENDER_EEVEE', choices=['BLENDER_EEVEE', 'CYCLES'])
    args = parser.parse_args(argv)
    print("args:{0}".format(args))


    renderer = Blender_render(scratch_dir=args.scratch_dir,
                             render_engine='BLENDER_EEVEE',
                             split=args.split)

    renderer.render()


