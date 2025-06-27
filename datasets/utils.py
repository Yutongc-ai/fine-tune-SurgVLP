import os
import random
import os.path as osp
from collections import defaultdict
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

def preload_local_features(cfg, split, model, loader):
    if not cfg.preload_local_features:
        batch_save_size = cfg.get('batch_save_size', 100)
        
        os.makedirs(cfg.cache_dir, exist_ok=True)
        
        part_idx = 0
        current_features, current_labels = [], []
        
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                torch.cuda.empty_cache()
                
                images, target = images.cuda(), target.cuda()
                if torch.isnan(images).any():
                    raise RuntimeError("image has nan value")
                
                _, image_features = model.extract_feat_img(images)
                if torch.isnan(image_features).any():
                    raise RuntimeError("image features has nan value")
                
                image_features /= (image_features.norm(dim=1, keepdim=True) + 1e-8)
                if torch.isnan(image_features).any():
                    raise RuntimeError("image norm features has nan value")
                
                current_features.append(image_features.detach().cpu())
                current_labels.append(target.detach().cpu())
                
                del images, target, image_features
                
                if len(current_features) >= batch_save_size:
                    save_features = torch.cat(current_features)
                    save_labels = torch.cat(current_labels)
                    
                    torch.save(save_features, 
                              f"{cfg.cache_dir}/{split}local_f_part{part_idx}.pt")
                    torch.save(save_labels, 
                              f"{cfg.cache_dir}/{split}local_l_part{part_idx}.pt")
                    
                    part_idx += 1
                    current_features, current_labels = [], []
                    del save_features, save_labels
                    torch.cuda.empty_cache()
        
            if current_features:
                save_features = torch.cat(current_features)
                save_labels = torch.cat(current_labels)
                torch.save(save_features, 
                          f"{cfg.cache_dir}/{split}local_f_part{part_idx}.pt")
                torch.save(save_labels, 
                          f"{cfg.cache_dir}/{split}local_l_part{part_idx}.pt")
                del save_features, save_labels
        
        with open(f"{cfg.cache_dir}/{split}parts_info.txt", "w") as f:
            f.write(str(part_idx + 1))
        
    with open(f"{cfg.cache_dir}/{split}parts_info.txt", "r") as f:
        num_parts = int(f.read().strip())
    
    all_features, all_labels = [], []
    for part_idx in range(num_parts):
        print("loading num parts: ", part_idx)
        features = torch.load(f"{cfg.cache_dir}/{split}local_f_part{part_idx}.pt")
        labels = torch.load(f"{cfg.cache_dir}/{split}local_l_part{part_idx}.pt")
        all_features.append(features)
        all_labels.append(labels)
    
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    return features, labels

# def preload_local_features(cfg, split, model, loader):
#     if cfg.preload_local_features == False:
#         features, labels = [], []
#         with torch.no_grad():
#             for i, (images, target) in enumerate(tqdm(loader)):
#                 images, target = images.cuda(), target.cuda()
#                 if torch.isnan(images).any():
#                     raise RuntimeError("image has nan value")
                
#                 _, image_features = model.extract_feat_img(images)
#                 if torch.isnan(image_features).any():
#                     raise RuntimeError("image features has nan value")
                
#                 image_features /= (image_features.norm(dim=-1, keepdim=True) + 1e-8)
#                 if torch.isnan(image_features).any():
#                     raise RuntimeError("image norm features has nan value")
                
#                 features.append(image_features.cpu())
#                 labels.append(target.cpu())

#                 del images, target, image_features

#         features, labels = torch.cat(features), torch.cat(labels)

#         torch.save(features, cfg.cache_dir + "/" + split + "local_f.pt")
#         torch.save(labels, cfg.cache_dir + "/" + split + "local_l.pt")
   
#     else:
#         features = torch.load(cfg.cache_dir + "/" + split + "local_f.pt")
#         labels = torch.load(cfg.cache_dir + "/" + split + "local_l.pt")
    
#     return features, labels

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items

class MultiLabelDatum:
    """Data instance for multi-label classification.

    Args:
        impath (str): image path.
        labels (list): list of class labels (integers).
        domain (int): domain label.
        classnames (list): list of class names.
    """

    def __init__(self, impath='', labels=None, domain=-1, classnames=None):
        assert isinstance(impath, str)
        assert isinstance(labels, list) or labels is None
        assert isinstance(domain, int)
        assert isinstance(classnames, list) or classnames is None

        self._impath = impath
        self._labels = labels if labels is not None else []
        self._domain = domain
        self._classnames = classnames if classnames is not None else []

    @property
    def impath(self):
        return self._impath

    @property
    def labels(self):
        return self._labels

    @property
    def domain(self):
        return self._domain

    @property
    def classnames(self):
        return self._classnames

class MultiLabelDatasetBase:
    """A unified dataset class for multi-label tasks."""
    dataset_dir = ''  # Dataset storage directory
    domains = []      # Domain names

    def __init__(self, train_x=None, train_u=None, val=None, test=None, class_names = None, num_classes = None):
        self._train_x = train_x  # Labeled training data
        self._train_u = train_u  # Unlabeled training data
        self._val = val          # Validation data
        self._test = test        # Test data

        # Compute dataset-wide attributes
        self._classnames = class_names
        self._num_classes = num_classes

    # Properties (same as single-label)
    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def split_dataset_by_label(self, data_source):
        """Group instances by each label."""
        output = defaultdict(list)
        if data_source:
            for item in data_source:
                for label in item.labels:
                    output[label].append(item)
        return output

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        """Few-shot sampling per label."""
        if num_shots < 1:
            raise RuntimeError("number of shots should be at least 1")

        # output = []
        assert len(data_sources) == 1
        data_source = data_sources[0]
        tracker = self.split_dataset_by_label(data_source)
        dataset = []
        for label, items in tracker.items():
            if len(items) >= num_shots:
                sampled = random.sample(items, num_shots)
            else:
                sampled = random.choices(items, k=num_shots) if repeat else items
            dataset.extend(sampled)
        # output.append(dataset)

        return dataset

class MultilabelDatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False, k_tfm=1, num_classes = 7):
        self.data_source = data_source
        self.num_classes = num_classes

        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )
        # super.__init__(self, data_source, input_size, transform = None, is_train = False, read_image = )

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = {}

        one_hot = torch.zeros(self.num_classes)
        for i, sample in enumerate(item.labels):
            one_hot[sample] = 1
        
        output['labels'] = one_hot

        img0 = read_image(item.impath)

        if self.transform is not None:
            img = self._transform_image(self.transform, img0)
            
            if torch.isnan(img).any():
                raise RuntimeError("img has nan value")
            
            output['img'] = img
        else:
            raise RuntimeError("transform function is None!")
        
        return output['img'], output['labels']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    num_classes = 7,
):

    if dataset_wrapper is None:
        dataset_wrapper = MultilabelDatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train, num_classes = num_classes),
        batch_size = batch_size,
        num_workers = 0,
        shuffle = is_train,
        drop_last = False,
        pin_memory = (torch.cuda.is_available()),
    )

    assert len(data_loader) > 0

    return data_loader