from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_folder():
    """
    return path ke folder dataset
    """
    return 'dataset/folder/road signs categories classification'

def get_road_sign_dataloaders(batch_size=128, num_workers=8):
    """
    road signs categories classification
    """
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
