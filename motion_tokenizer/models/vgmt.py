import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from .modules import Encoder, Decoder, VectorQuantizer, VisionEncoder
from .pose_hrnet import get_pose_net


class VisualSkeletonAttention(nn.Module):
    def __init__(self, feature_channels, num_sampling_points=4):
        super().__init__()
        self.num_sampling_points = num_sampling_points
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(feature_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_sampling_points * 2, kernel_size=1)
        )
        nn.init.constant_(self.offset_predictor[-1].weight, 0.)
        nn.init.constant_(self.offset_predictor[-1].bias, 0.)

    def forward(self, features, keypoint_coords):
        BT, C, H, W = features.shape
        J = keypoint_coords.shape[1]
        initial_grid = keypoint_coords.unsqueeze(-2)
        seed_features = F.grid_sample(features, initial_grid, align_corners=True)
        offsets = self.offset_predictor(seed_features)
        offsets = offsets.squeeze(-1).permute(0, 2, 1).reshape(BT, J, self.num_sampling_points, 2)
        norm_offsets = offsets.clone()
        norm_offsets[..., 0] = norm_offsets[..., 0] / (W - 1) * 2
        norm_offsets[..., 1] = norm_offsets[..., 1] / (H - 1) * 2
        new_grid = keypoint_coords.unsqueeze(-2) + norm_offsets
        sampled_features = F.grid_sample(features, new_grid, align_corners=True)
        sampled_features = sampled_features.permute(0, 2, 3, 1).reshape(BT, J, self.num_sampling_points * C)        
        return sampled_features


class VisionGuidedMotionTokenizer(nn.Module):
    def __init__(self, encoder, decoder, vq, vision_config):
        super(VisionGuidedMotionTokenizer, self).__init__()

        self.joint_data_type = 'joint3d_image_affined_normed'

        num_channels_list = vision_config.model.backbone.STAGE4.NUM_CHANNELS
        self.hrnet_output_level = [0,1,2,3]
        num_channels_list = [num_channels_list[i] for i in self.hrnet_output_level]
        self.num_vision_channels = sum(num_channels_list)
        
        code_dim_vision = int(vq.code_dim * 0.5)
        code_dim_skel = vq.code_dim - code_dim_vision
        self.code_dim_skel = code_dim_skel

        encoder.out_channels = code_dim_skel
        self.encoder = Encoder(**encoder)
        self.decoder = Decoder(**decoder)
        self.vq = VectorQuantizer(**vq)

        self.vision_backbone = get_pose_net(vision_config.model.backbone)

        self.vsa = nn.ModuleList()
        for channels in num_channels_list:
            self.vsa.append(VisualSkeletonAttention(feature_channels=channels, num_sampling_points=4))

        total_sampled_channels = sum([c * 4 for c in num_channels_list])

        self.vision_encoder = VisionEncoder(
            mid_channels=[total_sampled_channels, 512],
            out_channels=code_dim_vision,
            downsample_time=encoder.downsample_time,
            downsample_joint=[1, 1],
        )

    def load_model_weights(self, weight_path):
        safetensors_path = os.path.join(weight_path, "model.safetensors")
        pytorch_bin_path = os.path.join(weight_path, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            print(f"Loading model from {safetensors_path}")
            state_dict = load_safetensors(safetensors_path, device="cpu")
        elif os.path.exists(pytorch_bin_path):
            print(f"Loading model from {pytorch_bin_path}")
            state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {weight_path}")
        self.load_state_dict(state_dict, strict=True)
        return
        
    

    def get_vision_feats(self, joint3d_video, video_rgb):
        B, T = joint3d_video.shape[:2]
        video_rgb = video_rgb.permute(0, 1, 4, 2, 3).contiguous()
        images = video_rgb.reshape(-1, *video_rgb.shape[2:])
        
        image_feature_list = self.vision_backbone(images)
        image_feature_list = [image_feature_list[i] for i in self.hrnet_output_level]

        joint3d_images = joint3d_video.reshape(-1, *joint3d_video.shape[2:])
        
        grid_sample_ref_coords = joint3d_images[..., :2]
        features_ref_list = []
        for i, features in enumerate(image_feature_list):
            features_ref = self.vsa[i](features, grid_sample_ref_coords)
            features_ref_list.append(features_ref)
        video_ref_features = torch.cat(features_ref_list, dim=-1)
        
        video_ref_features = video_ref_features.reshape(B, T, *video_ref_features.shape[1:])

        return video_ref_features.permute(0, 3, 1, 2).contiguous()


    def forward(self, batch_dict, return_vq=False):
        joint_gt = batch_dict[self.joint_data_type].clone()
        joint3d_video = batch_dict[self.joint_data_type]


        video_rgb = batch_dict.video_rgb
        vision_feats = self.get_vision_feats(joint3d_video, video_rgb)


        joint_feats = joint3d_video.permute(0, 3, 1, 2)
        indices = None
        if not self.vq.is_train:
            joint_feats, loss, indices, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_gt.shape[1]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        else:
            tuple_return = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_gt.shape[1]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
            joint_feats, loss, perplexity, _ = tuple_return
        if return_vq:
            return joint_feats, loss
        joint_feats, _, _, _ = self.encdec_slice_frames(joint_feats, frame_batch_size=min(2, joint_gt.shape[1]), encdec=self.decoder, return_vq=return_vq, vision_feats=vision_feats)
        joint_feats = joint_feats.permute(0, 2, 3, 1)
        if self.vq.is_train:
            return joint_feats, loss, perplexity, joint_gt
        return joint_feats, loss, indices, joint_gt

    def encdec_slice_frames(self, joint_feats, frame_batch_size, encdec, return_vq, vision_feats=None):
        num_frames = joint_feats.shape[2]
        remaining_frames = num_frames % frame_batch_size
        joint_output = []

        for i in range(num_frames // frame_batch_size):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            joint_feats_intermediate = joint_feats[:, :, start_frame:end_frame]

            joint_feats_intermediate = encdec(joint_feats_intermediate)

            if encdec == self.encoder:
                vision_feats_intermediate = vision_feats[:, :, start_frame:end_frame]
                vision_encoded = self.vision_encoder(vision_feats_intermediate)

                feats_list = [joint_feats_intermediate, vision_encoded]
                joint_feats_intermediate = torch.cat(feats_list, dim=1)

            joint_output.append(joint_feats_intermediate)

        joint_concat = torch.cat(joint_output, dim=2)

        if encdec == self.encoder and self.vq is not None and not self.vq.is_train:
            joint_output, loss, indices = self.vq(joint_concat, return_vq=return_vq)
            tuple_return = (joint_output, loss, indices, joint_concat.shape)
            return tuple_return
        elif encdec == self.encoder and self.vq is not None and self.vq.is_train:
            joint_output, loss, preplexity = self.vq(joint_concat)
            tuple_return = (joint_output, loss, preplexity, joint_concat.shape)
            return tuple_return
        else:
            return joint_concat, None, None, joint_concat.shape

    def encode(self, joint3d_video, video_rgb=None, return_vq=False):
        vision_feats = self.get_vision_feats(joint3d_video, video_rgb)
        joint_feats = joint3d_video.permute(0, 3, 1, 2)
        _, _, indices, quant_shape = self.encdec_slice_frames(joint_feats, frame_batch_size=min(8, joint_feats.shape[-2]), encdec=self.encoder, return_vq=return_vq, vision_feats=vision_feats)
        return indices, quant_shape

    def get_code_from_indices(self, indices):
        flat_indices = indices.view(-1)
        dequantized_vectors = self.vq.dequantize(flat_indices)
        batch_size, t_quant, j_quant = indices.shape
        code_dim = dequantized_vectors.shape[-1]
        vectors_reshaped = dequantized_vectors.view(batch_size, t_quant, j_quant, code_dim)
        return vectors_reshaped.permute(0, 3, 1, 2).contiguous()

    def decode(self, indices: torch.Tensor):
        quantized_vectors = self.get_code_from_indices(indices)
        reconstructed_x, _, _, _ = self.encdec_slice_frames(
            quantized_vectors, 
            frame_batch_size=2,
            encdec=self.decoder, 
            return_vq=False
        )
        return reconstructed_x.permute(0, 2, 3, 1).contiguous()

    def decode_from_quantized(self, quantized: torch.Tensor) -> torch.Tensor:
        quantized_reshaped = quantized.permute(0, 3, 1, 2).contiguous()
        out = self.decoder(quantized_reshaped)
        return out