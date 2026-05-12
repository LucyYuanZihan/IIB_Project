import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from os.path import join, isdir
import os


synth_id_to_category = {
    '02691156': 'planes',  '02773838': 'bag',        '02801938': 'basket', #airplane = planes temporary
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'cars',        '03001627': 'chairs', #car=cars temporary chair=chairs
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone'
}

category_to_synth_id = {v: k for k, v in synth_id_to_category.items()}
synth_id_to_number = {k: i for i, k in enumerate(synth_id_to_category.keys())}


class NeRFShapeNetDataset(Dataset):
    def __init__(self, root_dir='/home/datasets/nerfdataset', shapenet_root_dir='/shared/sets/datasets/3D_points/ShapeNetCore.v2', classes=[],
                 transform=None, train=True):
        """
        Args:
            root_dir (string): Directory of structure: 
            >
                >classname1
                    >sampled
                        >train
                            >count_{name}.npz
                        >eval
                            >count_{name}.npz
                >classname2
                    ...
            
            where sampled has all the .NPZ of format: images : (n, W, H, channels), cam_poses (n, 4, 4), data :(N, 6)
            and shapenet is a shapenet directory for this class (contains .obj files).

            classes: list of class names

            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.shapenet_root_dir = shapenet_root_dir
        self.transform = transform

        self.classes = classes
        self.train = train
    
        self.data = []

        self._load()

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __getitem__(self, idx):
        data_files = self.train_data if self.train else self.test_data
        fn = data_files['sample_filename'][idx]
        class_name = data_files['class'][idx]
        obj_fn = data_files['obj_filename'][idx]

        with np.load(fn) as npz:
            # canonical keys expected by the trainer:
            if 'images' in npz and 'cam_poses' in npz and 'data' in npz:
                images = npz['images'].astype(np.float32)          # (V,H,W,3) in [0,1]
                cam_poses = npz['cam_poses'].astype(np.float32)    # (V,4,4)
                data = npz['data'].astype(np.float32)              # (K,6)

                sample = {
                    'images': images,
                    'cam_poses': cam_poses,
                    'data': data,
                }

                # >>> ADD THIS: load depths if present <<<
                if 'depths' in npz:
                    # (V,H,W) float32, meters, 0 = invalid
                    sample['depths'] = npz['depths'].astype(np.float32)

            else:
                # backward-compat for older generator output:
                images = npz['images']
                if images.shape[-1] == 4:
                    rgb = images[..., :3].astype(np.float32) / 255.0
                    a = images[..., 3] > 0
                    # white background where alpha==0
                    rgb[~a] = 1.0
                else:
                    rgb = images.astype(np.float32)
                    if rgb.dtype != np.float32:
                        rgb = rgb / 255.0

                cam_poses = npz['cam_poses'] if 'cam_poses' in npz else npz['poses']
                if 'data' in npz:
                    data = npz['data']
                else:
                    pts = npz['points']
                    cols = npz['colors']
                    data = np.concatenate([pts, cols], axis=1).astype(np.float32)

                sample = {
                    'images': rgb.astype(np.float32),
                    'cam_poses': cam_poses.astype(np.float32),
                    'data': data,
                }

                # >>> ALSO handle depths for backward-compat npz <<<
                if 'depths' in npz:
                    sample['depths'] = npz['depths'].astype(np.float32)


        if self.transform:
            sample = self.transform(sample)
        # return sample, class_name, obj_fn
        sample_id = os.path.splitext(os.path.basename(fn))[0]
        return sample, class_name, sample_id


    def _load(self):
        """
        Scan <root_dir>/<class>/sampled/*.npz, build a dataframe of samples,
        and split per-class into train/test (80/20) without assuming ShapeNet synsets.
        """
        import glob
        from os.path import join
        from pathlib import Path
        import pandas as pd

        print("Loading dataset:")
        self.train_data = pd.DataFrame(columns=['class', 'name', 'sample_filename', 'obj_filename'])
        self.test_data  = pd.DataFrame(columns=['class', 'name', 'sample_filename', 'obj_filename'])

        for data_class in self.classes:
            print(data_class)

            class_dir = join(self.root_dir, data_class, 'sampled')
            # NEW: subfolders
            train_dir = join(class_dir, 'train')
            eval_dir  = join(class_dir, 'eval')

            train_files = sorted(glob.glob(join(train_dir, '*.npz')))
            eval_files  = sorted(glob.glob(join(eval_dir,  '*.npz')))

            print("train:", len(train_files), "eval:", len(eval_files))

            def build_df(npz_files):
                rows = []
                for file in npz_files:
                    sample_name = Path(file).stem

                    syn = category_to_synth_id.get(data_class, None)
                    if syn is None:
                        obj_fn = ""
                    else:
                        obj_fn = join(self.shapenet_root_dir, syn, sample_name, 'models', 'model_normalized.obj')

                    rows.append({
                        'class': data_class,
                        'name': sample_name,
                        'sample_filename': file,
                        'obj_filename': obj_fn
                    })
                df = pd.DataFrame(rows, columns=['class', 'name', 'sample_filename', 'obj_filename'])
                if len(df) > 0:
                    df = df.sort_values(by=['name', 'sample_filename']).reset_index(drop=True)
                return df

            df_train = build_df(train_files)
            df_test  = build_df(eval_files)

            if len(df_train) == 0:
                print(f"Warning: no train .npz files found under {train_dir}")
            if len(df_test) == 0:
                print(f"Warning: no eval .npz files found under {eval_dir}")





            # Accumulate across classes
            self.train_data = pd.concat([self.train_data, df_train], ignore_index=True)
            self.test_data  = pd.concat([self.test_data,  df_test],  ignore_index=True)

        # Final reindex
        self.train_data = self.train_data.reset_index(drop=True)
        self.test_data  = self.test_data.reset_index(drop=True)


        print("Loaded train data:", len(self.train_data), "samples")
        print("Loaded test data:", len(self.test_data), "samples")