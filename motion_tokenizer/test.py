import os
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
import yaml
from easydict import EasyDict as edict
from accelerate import Accelerator

from config.vision_backbone import config as vision_config
from config.vqvae import vqvae_config
from models.vgmt import VisionGuidedMotionTokenizer
from lib.dataset import Multimodal_Mocap_Dataset


def update_dict(v, cfg):
    for kk, vv in v.items():
        if kk in cfg:
            if isinstance(vv, dict) and isinstance(cfg[kk], dict):
                update_dict(vv, cfg[kk])
            else:
                if vv is not None:
                    cfg[kk] = vv
        else:
            if vv is not None: 
                cfg[kk] = vv

def update_config(path, args):
    with open(path) as fin:
        exp_config = edict(yaml.safe_load(fin))
        update_dict(vars(args), exp_config)
        return exp_config
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/config.yaml', help="Path to config file")
    parser.add_argument('--resume_pth', type=str, required=True, help="Path to the trained VQVAE model checkpoint.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for testing.")
    
    parser.add_argument('--nb_code', type=int, default=8192, help="Number of vectors in the codebook.")
    parser.add_argument('--codebook_dim', type=int, default=3072, help="Dimension of each codebook vector.")

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run the test on.")

    parser.add_argument('--num_frames', type=int, default=None, help="Number of frames per sample.")
    parser.add_argument('--sample_stride', type=int, default=None)
    parser.add_argument('--data_stride', type=int, default=None)

    parser.add_argument('--load_data_file', type=str, default=None)
    parser.add_argument('--load_image_source_file', type=str, default=None)
    parser.add_argument('--load_bbox_file', type=str, default=None)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Unknown command-line args will be ignored: {unknown}")

    config = update_config(args.config, args)

    return config

def mpjpe_loss(pred, target):
    return torch.mean(torch.norm(pred - target, dim=-1))

def test_vqvae(args):
    accelerator = Accelerator()

    print("Loading model...")
    vqvae_config.encoder.out_channels = args.codebook_dim
    vqvae_config.vq.nb_code = args.nb_code
    vqvae_config.vq.code_dim = args.codebook_dim
    vqvae_config.decoder.in_channels = args.codebook_dim
    vqvae_config.vq.is_train = False
    vqvae = VisionGuidedMotionTokenizer(vqvae_config.encoder, vqvae_config.decoder, vqvae_config.vq, vision_config=vision_config)
    for p in vqvae.vision_backbone.parameters():
        p.requires_grad = False

    state_dict = {}
    safetensors_path = os.path.join(args.resume_pth, "model.safetensors")
    pytorch_bin_path = os.path.join(args.resume_pth, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        print(f"Loading model from {safetensors_path}")
        state_dict = load_safetensors(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        print(f"Loading model from {pytorch_bin_path}")
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Neither model.safetensors nor pytorch_model.bin found in {args.resume_pth}")

    unwrapped_state_dict = {}
    is_ddp_model = all(key.startswith('module.') for key in state_dict.keys())
    if is_ddp_model:
        print("Unwrapping model from DDP 'module.' prefix.")
        for k, v in state_dict.items():
            unwrapped_state_dict[k[7:]] = v
        state_dict = unwrapped_state_dict
    
    missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=True)

    vqvae = vqvae.to(args.device)
    vqvae.eval()
    print("Model loaded successfully.")

    print("Loading dataset...")
    dataset = Multimodal_Mocap_Dataset( num_frames=args.num_frames, sample_stride=args.sample_stride, data_stride=args.data_stride,
                                        designated_split='test',
                                        load_data_file=args.load_data_file,
                                        load_image_source_file=args.load_image_source_file,
                                        load_bbox_file=args.load_bbox_file,
                                        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        num_workers=8,
    )

    vqvae, dataloader = accelerator.prepare(vqvae, dataloader)
    print(f"Dataset loaded with {len(dataset)} samples.")

    total_err = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            if isinstance(batch, tuple):
                batch_dict = {}
                assert len(batch) == len(dataloader.dataset.get_item_list)
                for element_id, element in enumerate(dataloader.dataset.get_item_list):
                    batch_dict[element] = batch[element_id]
                batch = edict(batch_dict)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device)
            
            # Forward pass
            recon_data, _, indices, gt_data = vqvae(batch)

            recon_data_affined = (recon_data + batch.joint3d_image_affined_transl[..., None, :]) * batch.joint3d_image_affined_scale[..., None, :]
            recon_data_affined_xy = recon_data_affined[..., :2].clone()
            trans_inv = batch.affine_trans_inv
            recon_data_affined_xy1 = torch.cat([recon_data_affined_xy, torch.ones_like(recon_data_affined_xy[..., :1])], dim=-1)
            recon_data_3dimage_xy = torch.einsum('btij,btkj->btki', trans_inv, recon_data_affined_xy1)
            recon_data_3dimage = torch.cat([recon_data_3dimage_xy, recon_data_affined[..., 2:]], dim=-1)

            factor_2_5d = batch.factor_2_5d[..., None, None]

            recon_data_2_5dimage = recon_data_3dimage * factor_2_5d
            gt_data_2_5dimage = batch.joint_2_5d_image

            recon_data_2_5dimage_rootrel = recon_data_2_5dimage - recon_data_2_5dimage[..., 0:1, :]
            gt_data_2_5dimage_rootrel = gt_data_2_5dimage - gt_data_2_5dimage[..., 0:1, :]
            recon_data = recon_data_2_5dimage_rootrel
            gt_data = gt_data_2_5dimage_rootrel

            # Calculate loss
            min_len = min(gt_data.shape[1], recon_data.shape[1])
            reconstruction_loss = mpjpe_loss(recon_data[:, :min_len], gt_data[:, :min_len])
            total_err += reconstruction_loss.item()

    avg_err = total_err / len(dataloader)
    print(f"\n--- Test Results ---")
    print(f"Average Reconstruction Loss: {avg_err:.6f}")


if __name__ == '__main__':
    args = get_args()
    test_vqvae(args)
