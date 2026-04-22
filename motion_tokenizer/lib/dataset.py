import os.path as osp
import joblib
import numpy as np
import cv2
import torch
from collections import defaultdict


DATA_ROOT_PATH = 'data'

class Multimodal_Mocap_Dataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=16, sample_stride=1, data_stride=16, designated_split='train',
                 load_data_file=osp.join(DATA_ROOT_PATH, "h36m/h36m_sh_conf_cam_source_final.pkl"), 
                 load_image_source_file=osp.join(DATA_ROOT_PATH, "h36m/images_source.pkl"), 
                 load_bbox_file=osp.join(DATA_ROOT_PATH, "h36m/bboxes_xyxy.pkl"),
                 ):
        self.num_frames = num_frames
        self.get_item_list = ['factor_2_5d', 'video_rgb', 'joint3d_image_affined',
                              'joint3d_image_affined_normed', 'joint3d_image_affined_scale', 'joint3d_image_affined_transl',
                              'affine_trans_inv', 'joint_2_5d_image']

        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])

        data_dict = {}
        data_list = []
        for dt_file, img_src_file, bbox_file in zip(load_data_file.split(','), load_image_source_file.split(','), load_bbox_file.split(',')):
            img_list = joblib.load(img_src_file)[designated_split]
            valid_img_indices = []
            for frame_id, img_path in enumerate(img_list):
                if img_path is None:
                    continue
                img_list[frame_id] = osp.join(DATA_ROOT_PATH, img_path)
                valid_img_indices.append(frame_id)
            img_list = np.array(img_list)[valid_img_indices][::sample_stride]
            img_list = img_list.tolist()
            for frame_id, img_path in enumerate(img_list):
                if 'images_fps50' in img_path:
                    img_list[frame_id] = img_path.replace('images_fps50', f'images_fps50_cropped_448x448')
                elif 'imageFiles' in img_path:
                    img_list[frame_id] = img_path.replace('imageFiles', f'imageFiles_cropped_448x448')
                elif 'imageSequence' in img_path:
                    img_list[frame_id] = img_path.replace('imageSequence', f'imageSequence_cropped_448x448')
                else:
                    raise NotImplementedError
            img_list = np.array(img_list)
        

            datareader_config_unsplit = {'dt_file': dt_file,}
            datareader_config_split = {'chunk_len': num_frames, 'sample_stride': sample_stride, 'data_stride': data_stride, 'read_confidence': False}
            datareader_config = {**datareader_config_unsplit, **datareader_config_split}
            datareader = DataReader(**datareader_config)
            unsplit_data = DataReader.load_dataset_static(**datareader_config_unsplit)

            for data_mode in unsplit_data[designated_split].keys():
                if isinstance(unsplit_data[designated_split][data_mode], list):
                    unsplit_data[designated_split][data_mode] = np.array(unsplit_data[designated_split][data_mode])[valid_img_indices].tolist()
                else:
                    unsplit_data[designated_split][data_mode] = unsplit_data[designated_split][data_mode][valid_img_indices]

            data_dict[dt_file] = {}
            datareader.dt_dataset = unsplit_data


            joint3d_image = datareader.read_3d_image(designated_split=designated_split, do_screen_coordinate_normalize=False)
            data_sources = datareader.read_source(designated_split=designated_split)
            if designated_split == 'test':
                factor_2_5d = datareader.read_2_5d_factor(designated_split=designated_split)
                joint_2_5d_image = datareader.read_2_5d_image(designated_split=designated_split)
            else:
                factor_2_5d = np.zeros((joint3d_image.shape[0],), dtype=np.float32)
                joint_2_5d_image = np.zeros_like(joint3d_image)

            data_dict[dt_file]['sources'] = data_sources
            data_dict[dt_file]['2.5d_factor'] = factor_2_5d
            data_dict[dt_file]['joint_2.5d_image'] = joint_2_5d_image
            data_dict[dt_file]['image_sources'] = img_list


            img_ori_wh = datareader.read_hw(designated_split=designated_split)
            img_ori_w, img_ori_h = img_ori_wh[:, 0:1], img_ori_wh[:, 1:2]
            bboxes_xyxy = joblib.load(bbox_file)[designated_split][valid_img_indices][::sample_stride]
            AFFINE_TRANS_INV = []
            joint3d_image_affined = np.zeros_like(joint3d_image)
            for i in range(joint3d_image.shape[0]):
                bbox = bboxes_xyxy[i]
                center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
                scale = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                trans = get_affine_transform(center, scale, 0, [448, 448])

                pose_xy = joint3d_image[i, :, :2].copy()
                pose_xy1 = np.concatenate([pose_xy, np.ones((pose_xy.shape[0],1))], axis=1)
                pose_xy_affined = np.einsum('ij,kj->ik', pose_xy1, trans)

                trans_inv = get_affine_transform(center, scale, 0, [448, 448], inv=1)
                AFFINE_TRANS_INV.append(trans_inv)

                pose_z = joint3d_image[i, :, 2:3].copy()
                pose_z_affined = pose_z - pose_z[0:1, :]

                joint3d_image_affined[i, :, :2] = pose_xy_affined
                joint3d_image_affined[i, :, 2:3] = pose_z_affined            
            AFFINE_TRANS_INV = np.stack(AFFINE_TRANS_INV)
            joint3d_image_affined_scale = np.concatenate([np.array([[448 // 2]]).repeat(joint3d_image.shape[0], axis=0), 
                                                        np.array([[448 // 2]]).repeat(joint3d_image.shape[0], axis=0),
                                                        img_ori_w / 2,                                                                 
                                                        ], axis=-1)
            joint3d_image_affined_transl =np.array([[1, 1, 0]]).repeat(joint3d_image.shape[0], axis=0)

            data_dict[dt_file]['affine_trans_inv'] = AFFINE_TRANS_INV
            data_dict[dt_file]['joint3d_image_affined'] = joint3d_image_affined
            data_dict[dt_file]['joint3d_image_affined_scale'] = joint3d_image_affined_scale
            data_dict[dt_file]['joint3d_image_affined_transl'] = joint3d_image_affined_transl

            split_id = datareader.get_split_id(designated_split=designated_split, if_resample=True)

            data_list.extend(zip([dt_file]*len(split_id), split_id))

        self.data_dict = data_dict
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)        

    def __getitem__(self, idx):
        dt_file, slice_id = self.data_list[idx]

        joint_2_5d_image = self.data_dict[dt_file]['joint_2.5d_image'][slice_id]  # (num_frames, 17, 3)
        factor_2_5d = self.data_dict[dt_file]['2.5d_factor'][slice_id]
        sources = self.data_dict[dt_file]['sources'][slice_id]  # (num_frames, 2)
        affine_trans_inv = self.data_dict[dt_file]['affine_trans_inv'][slice_id]  # (num_frames, 3, 2)

        video_bgr = []
        image_sources = self.data_dict[dt_file]['image_sources'][slice_id]  # (num_frames,)
        for img_path in image_sources:
            image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image_bgr is None:
                raise FileNotFoundError(f"Failed to read image: {img_path}")
            video_bgr.append(image_bgr)
        video_bgr = np.stack(video_bgr, axis=0)  # (num_frames, H, W, 3), BGR order
        video_rgb = video_bgr[..., ::-1]  # Convert BGR to RGB
        video_rgb = (video_rgb / 255.0 - self.img_mean) / self.img_std   # to [0,1], then normalize

        joint3d_image_affined = self.data_dict[dt_file]['joint3d_image_affined'][slice_id]  # (num_frames, 17, 3)
        joint3d_image_affined_scale = self.data_dict[dt_file]['joint3d_image_affined_scale'][slice_id]  # (num_frames, 3)
        joint3d_image_affined_transl = self.data_dict[dt_file]['joint3d_image_affined_transl'][slice_id]
        joint3d_image_affined_normed = joint3d_image_affined / joint3d_image_affined_scale[..., None, :] - joint3d_image_affined_transl[..., None, :]

        slice_id = np.array(slice_id).astype(np.int64)

        return_dict = {}
        for get_item in self.get_item_list:
            item = locals()[get_item]
            try:
                item = torch.from_numpy(item)
                if item.dtype == torch.int64:
                    pass
                else:
                    item = item.float()
            except:
                pass
            return_dict[get_item] = item
        return return_dict


    def collate_fn(self, batch):
        return_dict = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                return_dict[k].append(v)
                
        for k, v in return_dict.items():
            try:
                return_dict[k] = torch.stack(v, dim=0)
            except:
                pass

        if len(return_dict) == 1:
            return return_dict[ list(return_dict.keys())[0] ]
        return_dict = tuple([v for k, v in return_dict.items()])
        return return_dict




class DataReader(object):
    fileName_to_h36mKey = {
        'factor_2_5d': '2.5d_factor', 
        'joint_2d_image': 'joint_2d', 
        'joint_3d_image': 'joint3d_image', 
        'source': 'source',
        'camera_name': 'camera_name',
        'joint_3d_cam': 'joint_3d_cam',
        'joint_3d_world': 'joint_3d_world',
    }
    def __init__(self, dt_file, chunk_len, sample_stride, data_stride,
                 read_confidence=False,
                 read_modality=['joint2d', 'joint3d'],
                 subdatasets=None,
                 **kwargs):
        
        self.dt_file = dt_file

        self.n_frames = chunk_len
        self.sample_stride = sample_stride
        self.data_stride = data_stride
        self.read_confidence = read_confidence

        self.read_modality = read_modality

        self.split_id = None
        self.dt_dataset = kwargs.get('dt_dataset', None)
        self.split = kwargs.get('split', None)

        self.subdatasets = subdatasets

    def get_num_clips(self, designated_split, **kwargs):
        return len(self.get_split_id(designated_split, **kwargs))

    def get_split_id(self, designated_split=None, data_stride=None, **kwargs):
        data_stride = data_stride if data_stride is not None else self.data_stride

        if designated_split is not None:
            vid_list = self.dt_dataset[designated_split]['source'][::self.sample_stride]
            split_id = split_clips(vid_list, self.n_frames, data_stride, if_resample=kwargs.get('if_resample', True))
            return split_id

        if self.split_id is not None:
            return self.split_id
        vid_list = self.dt_dataset[self.split]['source'][::self.sample_stride]
        self.split_id = split_clips(vid_list, self.n_frames, data_stride, if_resample=kwargs.get('if_resample', True))
        return self.split_id

    def read_2d(self, designated_split=None, do_screen_coordinate_normalize=True):
        split = designated_split if designated_split is not None else self.split

        joints_2d = self.dt_dataset[split]['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]

        if do_screen_coordinate_normalize:
            if 'camera_name' in self.dt_dataset[split]:
                for idx, camera_name in enumerate(self.dt_dataset[split]['camera_name'][::self.sample_stride]):
                    if camera_name == '54138969' or camera_name == '60457274':
                        res_w, res_h = 1000, 1002
                    elif camera_name == '55011271' or camera_name == '58860488':
                        res_w, res_h = 1000, 1000
                    else:
                        assert 0, '%d data item has an invalid camera name' % idx
                    joints_2d[idx, :, :] = joints_2d[idx, :, :] / res_w * 2 - [1, res_h / res_w]
            elif 'res' in self.dt_dataset[split]:
                res = self.dt_dataset[split]['res'][::self.sample_stride][:, None, :]   # (T,2) -> (T,1,2)
                res_w, res_h = res[..., 0:1], res[..., 1:2]
                denormalize_factor = np.concatenate((np.ones_like(res_w), res_h / res_w), axis=-1)  # (T,1,2)
                joints_2d[..., :2] = joints_2d[..., :2] / res_w * 2 - denormalize_factor
            
        if self.read_confidence:
            if 'confidence' in self.dt_dataset[split]:
                dataset_confidence = self.dt_dataset[split]['confidence'][::self.sample_stride].astype(np.float32)
            else:
                dataset_confidence = np.ones_like(joints_2d[..., :1])
            if len(dataset_confidence.shape)==2: 
                dataset_confidence = dataset_confidence[:,:,None]
            joints_2d = np.concatenate((joints_2d, dataset_confidence), axis=-1)  # [N, 17, 3]
        return joints_2d
    
    def read_3d_image(self, designated_split=None, do_screen_coordinate_normalize=True):
        split = designated_split if designated_split is not None else self.split

        joints_3d = self.dt_dataset[split]['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]

        if not do_screen_coordinate_normalize:
            return joints_3d

        if 'camera_name' in self.dt_dataset[split]:
            for idx, camera_name in enumerate(self.dt_dataset[split]['camera_name'][::self.sample_stride]):
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                joints_3d[idx, :, :2] = joints_3d[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
                joints_3d[idx, :, 2:] = joints_3d[idx, :, 2:] / res_w * 2
        elif 'res' in self.dt_dataset[split]:
            res = self.dt_dataset[split]['res'][::self.sample_stride][:, None, :]   # (T,2) -> (T,1,2)
            res_w, res_h = res[..., 0:1], res[..., 1:2]
            denormalize_factor = np.concatenate((np.ones_like(res_w), res_h / res_w), axis=-1)  # (T,1,2)
            joints_3d[..., :2] = joints_3d[..., :2] / res_w * 2 - denormalize_factor
            joints_3d[..., 2:] = joints_3d[..., 2:] / res_w * 2
        return joints_3d
    
    def read_2_5d_factor(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        assert split == 'test'
        factor_2_5d = np.array(self.dt_dataset[split]['2.5d_factor'][::self.sample_stride])
        return factor_2_5d
    
    def read_joint(self, key, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        joint_npy = np.array(self.dt_dataset[split][key][::self.sample_stride])
        return joint_npy
        
    def read_2_5d_image(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        assert split == 'test'
        if 'joints_2.5d_image' not in self.dt_dataset[split]:
            return None
        factor_2_5d = np.array(self.dt_dataset[split]['joints_2.5d_image'][::self.sample_stride])
        return factor_2_5d
    
    def read_action(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        assert split == 'test'
        if 'action' not in self.dt_dataset[split]:
            return None
        actions = np.array(self.dt_dataset[split]['action'][::self.sample_stride])
        return actions
    
    def read_source(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        sources = np.array(self.dt_dataset[split]['source'][::self.sample_stride])
        return sources
    
    def read_image_source(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        image_sources = np.array(self.dt_dataset[split]['img_path'][::self.sample_stride])
        return image_sources

    def read_hw(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split
        if 'camera_name' in self.dt_dataset[split]:
            camera_names = self.dt_dataset[split]['camera_name'][::self.sample_stride]
            test_hw = np.zeros((len(camera_names), 2))
            for idx, camera_name in enumerate(camera_names):
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                test_hw[idx] = res_w, res_h
        elif 'res' in self.dt_dataset[split]:
            test_hw = self.dt_dataset[split]['res'][::self.sample_stride]
        return test_hw
    
    def get_split_data(self, designated_split=None, **kwargs):
        if_resample = kwargs.get('if_resample', True)
        if self.dt_dataset is None:
            self.load_dataset()

        split_id = self.get_split_id(designated_split, **kwargs)

        data_dict = {}
        if 'joint2d' in self.read_modality:
            joints_2d = self.read_2d(designated_split)  # [N, 17, 3]
            joints_2d = joints_2d[split_id] if if_resample else [joints_2d[id] for id in split_id]
            if if_resample: assert (not self.read_confidence) == (joints_2d[..., -1] == 0).all()
            assert len(split_id) == len(joints_2d)
            data_dict['joint2d'] = joints_2d

        if 'joint3d' in self.read_modality:
            joints_3d = self.read_3d_image(designated_split)
            joints_3d = joints_3d[split_id] if if_resample else [joints_3d[id] for id in split_id]
            assert len(split_id) == len(joints_3d)
            data_dict['joint3d'] = joints_3d
        
        return data_dict
    
    def load_dataset(self):
        self.dt_dataset = self.load_dataset_static(self.dt_file)

    @staticmethod
    def load_dataset_static(dt_file):
        if dt_file.endswith('.pt') or dt_file.endswith('.pth'):
            dt_dataset = joblib.load(dt_file)
        elif dt_file.endswith('.pkl'):
            dt_dataset = joblib.load(dt_file)
        elif osp.isdir(dt_file):
            dt_dataset = {'train': {}, 'test': {}}
            for data_mode in ['factor_2_5d', 'joint_2d_image', 'joint_3d_image', 'source', 'camera_name', 'joint_3d_cam', 'joint_3d_world']:
                data_file = osp.join(dt_file, f'{data_mode}.pkl')
                if osp.exists(data_file):
                    data_dict = joblib.load(data_file)
                    for data_split in ['train', 'test']:
                        dt_dataset[data_split][DataReader.fileName_to_h36mKey.get(data_mode, data_mode)] = data_dict[data_split]
        return dt_dataset
    
    def denormalize(self, test_data):
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw)
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data


def split_clips(vid_list, n_frames, data_stride, if_resample=True):
    result = []
    n_clips = 0
    st = 0
    i = 0
    saved = set()
    while i<len(vid_list):
        i += 1
        if i-st == n_frames:
            result.append(range(st,i))
            saved.add(vid_list[i-1])
            st = st + data_stride
            n_clips += 1
        if i==len(vid_list):
            break
        if vid_list[i]!=vid_list[i-1]: 
            if not (vid_list[i-1] in saved):
                if if_resample:
                    resampled = resample(i-st, n_frames) + st
                else:
                    resampled = range(st, i)
                result.append(resampled)
                saved.add(vid_list[i-1])
            st = i
    return result


def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result
    

def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    center = np.array(center)
    scale = np.array(scale)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    src_dir = np.array([0, (src_w-1) * -0.5], np.float32)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)
