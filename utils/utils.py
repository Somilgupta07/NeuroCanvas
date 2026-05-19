from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        # sorted for reproducibility
        self.files = sorted(os.listdir(root))
        self.files = [p for p in self.files if p.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(size, crop, final_size, augment=False):
    """
    augment=True  → heavy augmentation for small datasets
    augment=False → standard resize + crop (original behaviour)
    """
    transform_list = []

    if size > 0:
        transform_list.append(transforms.Resize(size))

    if augment:
        transform_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ]

    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


class RepeatedDataset(Dataset):
    """
    Wraps a dataset and repeats it N times so DataLoader
    sees more steps per epoch — critical for tiny datasets.

    e.g. 11 images x repeat=100 = 1100 samples per epoch
    """
    def __init__(self, dataset, repeat=100):
        self.dataset = dataset
        self.repeat  = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


def adapative_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std     = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var  = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std  = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
