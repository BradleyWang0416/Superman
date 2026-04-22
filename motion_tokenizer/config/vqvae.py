from easydict import EasyDict as edict

vqvae_config = edict()

vqvae_config.encoder = edict(
    in_channels=3, 
    mid_channels=[128, 512], 
    out_channels=None, 
    downsample_time=[1, 2], 
    downsample_joint=[1, 1]
)

vqvae_config.vq = edict(
    nb_code=None, 
    code_dim=None,
)

vqvae_config.decoder = edict(
    in_channels=None, 
    mid_channels=[512, 128], 
    out_channels=3, 
    upsample_rate=2.0, 
    frame_upsample_rate=[2.0, 1.0], 
    joint_upsample_rate=[1.0, 1.0]
)