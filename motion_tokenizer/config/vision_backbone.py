from easydict import EasyDict as edict

config = edict()

# model definition
config.model = edict()
config.model.image_shape = [448, 448]
config.model.init_weights = True
config.model.checkpoint = None

config.model.backbone = edict()
config.model.backbone.type = 'hrnet_32'
config.model.backbone.num_final_layer_channel = 17
config.model.backbone.num_joints = 17
config.model.backbone.num_layers = 152
config.model.backbone.init_weights = True
config.model.backbone.fix_weights = True
config.model.backbone.checkpoint = "checkpoint/hrnet/pose_hrnet_w32_256x192.pth"

# pose_hrnet related params
config.model.backbone.NUM_JOINTS = 17
config.model.backbone.PRETRAINED_LAYERS = ['*']
config.model.backbone.STEM_INPLANES = 64
config.model.backbone.FINAL_CONV_KERNEL = 1

config.model.backbone.STAGE2 = edict()
config.model.backbone.STAGE2.NUM_MODULES = 1
config.model.backbone.STAGE2.NUM_BRANCHES = 2
config.model.backbone.STAGE2.NUM_BLOCKS = [4, 4]
config.model.backbone.STAGE2.NUM_CHANNELS = [32, 64]
config.model.backbone.STAGE2.BLOCK = 'BASIC'
config.model.backbone.STAGE2.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE3 = edict()
config.model.backbone.STAGE3.NUM_MODULES = 4
config.model.backbone.STAGE3.NUM_BRANCHES = 3
config.model.backbone.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.model.backbone.STAGE3.NUM_CHANNELS = [32, 64, 128]
config.model.backbone.STAGE3.BLOCK = 'BASIC'
config.model.backbone.STAGE3.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE4 = edict()
config.model.backbone.STAGE4.NUM_MODULES = 3
config.model.backbone.STAGE4.NUM_BRANCHES = 4
config.model.backbone.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.model.backbone.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
config.model.backbone.STAGE4.BLOCK = 'BASIC'
config.model.backbone.STAGE4.FUSE_METHOD = 'SUM'

# pose_resnet related params
config.model.backbone.NUM_LAYERS = 50
config.model.backbone.DECONV_WITH_BIAS = False
config.model.backbone.NUM_DECONV_LAYERS = 3
config.model.backbone.NUM_DECONV_FILTERS = [256, 256, 256]
config.model.backbone.NUM_DECONV_KERNELS = [4, 4, 4]
config.model.backbone.FINAL_CONV_KERNEL = 1
config.model.backbone.PRETRAINED_LAYERS = ['*']