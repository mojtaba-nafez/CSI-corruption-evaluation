import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed

DATA_PATH = './data/'
IMAGENET_PATH = './data/ImageNet'

CIFAR10_SUPERCLASS = list(range(10))  # one class
CIFAR100_CORUPTION_SUPERCLASS = list(range(20))  # one class
EMNIST_SUPERCLASS = list(range(26))  # one class

IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]



import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import shutil
import torchvision


EMNIST_CORRUPTION_TYPES = [
    'shot_noise',
    'impulse_noise',
    'glass_blur',
    'motion_blur',
    'shear',
    'scale',
    'rotate',
    'brightness',
    'contrast',
    'saturate',
    'inverse'
]

class EMNISTCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, corruption_type, root_dir='./', transform=None):
        """
        Args:
            root_dir (string): Directory with all the corrupted dataset .npy files.
            corruption_type (string): Type of corruption applied to the dataset.
                                      It is used to identify the files.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.transform = transform
        self.images = np.load(os.path.join(root_dir, f'{corruption_type}_images.npy'))
        self.labels = np.load(os.path.join(root_dir, f'{corruption_type}_labels.npy'))
        self.images = self.images.transpose((0, 2, 1))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image, mode='L')  # 'L' mode means grayscale

        if self.transform:
            image = self.transform(image)

        return image, label - 1

SVHN_CORRUPTION_TYPES = [
    'Contrast',
    'Gaussian Blur',
    'Gaussian Noise',
    'Glass Blur',
    'Impulse Noise',
    'Shot Noise',
    'Speckle Noise',
]

class SVHN_CORRUPTION(torch.utils.data.Dataset):
    def __init__(self, transform=None, svhn_corruption_label = './SVHN-C/labels.npy', svhn_corruption_data = './SVHN-C/Contrast.npy'):
        self.labels_10 = np.load(svhn_corruption_label)
        self.svhn_corruption_data = svhn_corruption_data
        self.data = np.load(svhn_corruption_data)
        self.data = np.transpose(self.data, (0, 2, 3, 1))
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        label = self.labels_10[index]

        if self.transform:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)    
            
        return x, label
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(corruption_type={self.svhn_corruption_data})"
    
class MNIST_CORRUPTION(Dataset):
    def __init__(self, root_dir, corruption_type, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.corruption_type = corruption_type
        self.train = train
        
        indicator = 'train' if train else 'test'
        folder = os.path.join(self.root_dir, self.corruption_type, f'saved_{indicator}_images')
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        
        if train:
            data = np.load(os.path.join(root_dir, corruption_type, 'train_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'train_labels.npy'))
        else:
            data = np.load(os.path.join(root_dir, corruption_type, 'test_images.npy'))
            labels = np.load(os.path.join(root_dir, corruption_type, 'test_labels.npy'))
            
        self.labels = labels
        self.image_paths = []

        for idx, img in enumerate(data):
            path = os.path.join(folder, f"{idx}.png")
            self.image_paths.append(path)
            
            if not os.path.exists(path):
                img_pil = torchvision.transforms.ToPILImage()(img)
                img_pil.save(path)
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB") 

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label



class FMNIST_CORRUPTION(Dataset):
    def __init__(self, split='test', transform=None):
        from datasets import load_dataset
        # Check if split is valid
        if split not in ['train', 'test']:
            raise ValueError("Split must be 'train' or 'test'.")

        self.split = split
        self.transform = transform or transforms.ToTensor()  # Default transform

        # Load the dataset
        self.data = load_dataset("mweiss/fashion_mnist_corrupted")[self.split]
        self.images = np.array([np.array(image) for image in self.data['image']])
        self.labels = np.array(self.data['label'])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label
        image, label = self.images[idx], self.labels[idx]

        # Convert to PIL Image for compatibility with torchvision transforms
        image = Image.fromarray(image, mode='L')  # 'L' mode means grayscale

        # Apply the transform to the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

def sparse2coarse(targets):
    coarse_labels = np.array(
        [4,1,14, 8, 0, 6, 7, 7, 18, 3, 3,
         14, 9, 18, 7, 11, 3, 9, 7, 11, 6, 11, 5,
         10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10,
         12, 14, 16, 9, 11, 5, 5, 19, 8, 8, 15,
         13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0,
         17, 4, 18, 17, 10, 3, 2, 12, 12, 16, 12,
         1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13,
         15, 13, 16, 19, 2, 4, 6, 19, 5, 5, 8,
         19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,])
    return coarse_labels[targets]

class CIFAR_CORRUCPION(Dataset):
    def __init__(self, transform=None, normal_idx = [0], cifar_corruption_label = 'CIFAR-10-C/labels.npy', cifar_corruption_data = './CIFAR-10-C/defocus_blur.npy'):
        self.labels_10 = np.load(cifar_corruption_label)
        self.labels_10 = self.labels_10[:10000]
        if cifar_corruption_label == 'CIFAR-100-C/labels.npy':
            self.labels_10 = sparse2coarse(self.labels_10)
        self.data = np.load(cifar_corruption_data)
        self.data = self.data[:10000]
        self.transform = transform
       
    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels_10[index]
        if self.transform:
            x = Image.fromarray((x * 255).astype(np.uint8))
            x = self.transform(x)    
        return x, y
    
    def __len__(self):
        return len(self.data)

class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    download = True
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(P.ood_samples,
                                                                                 P.resize_factor, P.resize_fix)
        else:
            train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset == 'svhn':
            image_size = (32, 32, 3)
            n_classes = 10
            train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=test_transform)
            test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
    elif dataset == 'svhn-10':
        image_size = (32, 32, 3)
        n_classes = 10
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=transform)
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
    elif dataset=='mnist-corruption':
        n_classes = 10
        transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
        ])
        test_set = MNIST_CORRUPTION(root_dir=P.mnist_corruption_folder, corruption_type=P.mnist_corruption_type, transform=transform, train=False)
        train_set = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset=='fmnist-corruption':
        
        n_classes = 10
        transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
        ])
        
        test_set = FMNIST_CORRUPTION(split='test', transform=transform)
        train_set = datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=transform)
        
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    elif dataset == 'svhn-10-corruption':
        image_size = (32, 32, 3)
        def gaussian_noise(image, mean=P.noise_mean, std = P.noise_std, noise_scale = P.noise_scale):
            image = image + (torch.randn(image.size()) * std + mean)*noise_scale
            return image

        n_classes = 10
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Lambda(gaussian_noise)
        ])

        train_set = datasets.SVHN(DATA_PATH, split='train', download=download, transform=train_transform)
        test_set =  SVHN_CORRUPTION(svhn_corruption_data=os.path.join(P.svhn_corruption_folder, f'{P.svhn_corruption_type}.npy'),
                                    svhn_corruption_label=os.path.join(P.svhn_corruption_folder, 'labels.npy'), 
                                    transform=train_transform)
        
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)

    elif dataset == 'emnist':
        
        image_size = (32, 32, 3)
        transpose_transform = transforms.Lambda(lambda img: img.transpose(1, 2))

        n_classes = 26
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transpose_transform,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transpose_transform,
        ])

        train_set = datasets.EMNIST(DATA_PATH, split='letters', train=True, download=download, transform=train_transform)    
        test_set = datasets.EMNIST(DATA_PATH,  split='letters', train=False, download=download, transform=test_transform)
        
        train_set.targets = train_set.targets - 1
        test_set.targets = test_set.targets - 1
        
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'emnist-corruption':
        image_size = (32, 32, 3)
        n_classes = 26
        
        transpose_transform = transforms.Lambda(lambda img: img.transpose(1, 2))


        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transpose_transform,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
       
        train_set = datasets.EMNIST(DATA_PATH, split='letters', train=True, download=download, transform=train_transform)
            
        test_set = EMNISTCorruptionDataset(root_dir=P.emnist_corruption_folder, corruption_type=P.emnist_corruption_type, transform=test_transform)
        
        train_set.targets = train_set.targets - 1
        
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
        
    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
    elif dataset=='cifar10-corruption':
        n_classes = 10
        transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=transform)
        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset=='cifar100-corruption':
        n_classes = 100
        transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
        ])
        test_set = CIFAR_CORRUCPION(transform=transform, cifar_corruption_label='CIFAR-100-C/labels.npy', cifar_corruption_data=P.cifar_corruption_data)
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=transform)
        
        train_set.targets = sparse2coarse(train_set.targets)

        print("train_set shapes: ", train_set[0][0].shape)
        print("test_set shapes: ", test_set[0][0].shape)
    
    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'lsun_pil' or dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet_pil' or dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10' or dataset=='cifar10-corruption' or dataset=='svhn' or dataset=='svhn-10-corruption' or dataset=='svhn-10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'emnist' or dataset=='emnist-corruption':
        return EMNIST_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == "cifar100-corruption":
        return CIFAR100_CORUPTION_SUPERCLASS
    elif dataset == "mnist-corruption":
        return CIFAR10_SUPERCLASS
    elif dataset == "fmnist-corruption":
        return CIFAR10_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    try:        
        for idx, tgt in enumerate(dataset.targets):
            if tgt in classes:
                indices.append(idx)
    except:
        # SVHN
        for idx, (_, tgt) in enumerate(dataset):
            if tgt in classes:
                indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
