import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

def get_data_folder():
    """
    return path ke folder dataset
    """
    return 'dataset/road signs categories classification'

# dataloader biasa
def get_road_sign_dataloaders(batch_size=128, num_workers=8):
    """
    road signs categories classification
    """
    print('Creating dataloader from Road Sign Categories Classification...')
    data_folder = get_data_folder()

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = datasets.ImageFolder(root=data_folder + '/train', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.ImageFolder(root=data_folder + '/test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    return train_loader, test_loader


# dataloader untuk contrastive
class RoadSignsDatasetContrastive(datasets.ImageFolder):
    """Custom dataset class for road signs classification with contrastive sampling."""
    def __init__(self, root, train=True, transform=None, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root,
                         transform=transform)
        self.train = train
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.percent = percent

        num_classes = len(os.listdir(root))
        if train:
            self.num_samples = len(self.samples)
        else:
            self.num_samples = len(self.imgs)

        self.cls_positive = [[] for _ in range(num_classes)]
        for i, (img_path, target) in enumerate(self.samples if train else self.imgs):
            self.cls_positive[target].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < self.percent < 1:
            n = int(len(self.cls_negative[0]) * self.percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        if self.train:
            img_path, target = self.samples[index]
        else:
            img_path, target = self.imgs[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_road_signs_contrastive_dataloaders(batch_size=128, num_workers=8, k=4096, mode='exact',
                                           is_sample=True, percent=1.0):
    """
    Get dataloaders for road signs classification dataset with contrastive sampling.
    """
    print("Creating contrastive dataloader from Road Sign Categories Classification...")
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = RoadSignsDatasetContrastive(root=os.path.join(data_folder, 'train'),
                                            train=True,
                                            transform=train_transform,
                                            k=k,
                                            mode=mode,
                                            is_sample=is_sample,
                                            percent=percent)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = RoadSignsDatasetContrastive(root=os.path.join(data_folder, 'test'),
                                           train=False,
                                           transform=test_transform,
                                           k=k,
                                           mode=mode,
                                           is_sample=is_sample,
                                           percent=percent)

    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader