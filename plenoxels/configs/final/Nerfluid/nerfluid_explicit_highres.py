config = {
 'expname': 'explicit_fluid_highres_explicit',
 'logdir': './logs/fluid_debug',
 'device': 'cuda:0',

 'data_downsample': 2.0,
 'data_dirs': ['/viscam/u/yangzheng/fluid_debug_new'],
 'contract': False,
 'ndc': False,
 'isg': False,
 'isg_step': -1,
 'ist_step': 50000,
 'keyframes': False,
 'scene_bbox': [[0.5, 0.5, -0.2], [1.8, 1.8, 2]],

 # Optimization settings
 'num_steps': 120001,
 'batch_size': 4096,
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.01,

 # Regularization
 'distortion_loss_weight': 0.001,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0001,
 'l1_time_planes_proposal_net': 0.0001,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.001,
 'time_smoothness_weight_proposal_net': 1e-05,

 # Training settings
 'save_every': 10000,
 'valid_every': 10000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_samples': [256, 128],
 'num_proposal_iterations': 2,
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 150]},
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [256, 256, 256, 150]}
 ],

 # Model settings
 'concat_features_across_scales': True,
 'density_activation': 'trunc_exp',
 'linear_decoder': True,
 'linear_decoder_layers': 1,
 'multiscale_res': [1, 2, 4, 8],
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 32,
  'resolution': [64, 64, 64, 150]
 }],
}
