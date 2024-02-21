import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import scipy.io as sio
import tarfile
import matplotlib.pyplot as plt

def extract_images(tgz_file, target_folder):
    with tarfile.open(tgz_file, 'r:gz') as tar:
        tar.extractall(path=target_folder)

def read_labels(mat_file):
    labels = sio.loadmat(mat_file)['labels'][0]
    return labels - 1  # Adjust labels to start from 0

def get_data_folder():
    data_folder = './dataset/102flowers'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

class Flowers102(Dataset):
    """Custom dataset class for 102 Flowers Dataset."""
    def __init__(self, root_folder, labels, transform=None, train=True):
        self.root_folder = root_folder
        self.labels = labels
        self.transform = transform
        self.train = train
        if self.train:
            self.total_samples = int(len(self.labels) * 0.8)
        else:
            self.total_samples = len(self.labels) - int(len(self.labels) * 0.8)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.train:
            img_path = os.path.join(self.root_folder, f'image_{idx+1:05d}.jpg')
            label = self.labels[idx]
        else:
            img_path = os.path.join(self.root_folder, f'image_{idx+self.total_samples+1:05d}.jpg')
            label = self.labels[idx+self.total_samples]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

def get_flowers102_dataloader(batch_size=128, num_workers=8):
    print('Creating dataloader from 102flowers...')

    data_folder = './dataset/102flowers'
    # Read labels
    mat_file = os.path.join(data_folder, 'imagelabels.mat')
    labels = read_labels(mat_file)

    # Extract images if not already extracted
    tgz_file = os.path.join(data_folder, '102flowers.tgz')
    if not os.path.exists(os.path.join(data_folder, 'jpg')):
        extract_images(tgz_file, data_folder)

    resolution = (416, 416)

    train_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = Flowers102(root_folder=os.path.join(data_folder, 'jpg'),
                           labels=labels,
                           transform=train_transform,
                           train=True)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = Flowers102(root_folder=os.path.join(data_folder, 'jpg'),
                          labels=labels,
                          transform=test_transform,
                          train=False)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))
    
    print(f"Train: {len(train_set)}")
    print(f"test: {len(test_set)}")

    sample_image, _ = train_set[0]
    sample_image_pil = transforms.ToPILImage()(sample_image)
    plt.figure()
    plt.imshow(sample_image_pil)
    plt.title('Sample Image')
    plt.axis('off')
    plt.text(10, 10, f'Resolution: {sample_image_pil.size}', color='white', fontsize=10, verticalalignment='top')
    plt.show()

    return train_loader, test_loader
