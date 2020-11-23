import os
import torch
import torch.utils.data as data
import numpy as np
import cv2

from torchvision.datasets import ImageNet

from PIL import Image, ImageFilter
import h5py
from glob import glob


class ImageNet_blur(ImageNet):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = sample.filter(gauss_blur)
        blurred_img2 = sample.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        if self.transform is not None:
            sample = self.transform(sample)
            blurred_img = self.transform(blurred_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, blurred_img), target


class Imagenet_Segmentation(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_Blur(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        # self.h5py = h5py.File(path, 'r+')
        self.h5py = None
        tmp = h5py.File(path, 'r')
        self.data_length = len(tmp['/value/img'])
        tmp.close()
        del tmp

    def __getitem__(self, index):

        if self.h5py is None:
            self.h5py = h5py.File(self.path, 'r')

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        gauss_blur = ImageFilter.GaussianBlur(11)
        median_blur = ImageFilter.MedianFilter(11)

        blurred_img1 = img.filter(gauss_blur)
        blurred_img2 = img.filter(median_blur)
        blurred_img = Image.blend(blurred_img1, blurred_img2, 0.5)

        # blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
        # blurred_img2 = np.float32(cv2.medianBlur(img, 11))
        # blurred_img = (blurred_img1 + blurred_img2) / 2

        if self.transform is not None:
            img = self.transform(img)
            blurred_img = self.transform(blurred_img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return (img, blurred_img), target

    def __len__(self):
        # return len(self.h5py['/value/img'])
        return self.data_length


class Imagenet_Segmentation_eval_dir(data.Dataset):
    CLASSES = 2

    def __init__(self,
                 path,
                 eval_path,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.h5py = h5py.File(path, 'r+')

        # 500 each file
        self.results = glob(os.path.join(eval_path, '*.npy'))

    def __getitem__(self, index):

        img = np.array(self.h5py[self.h5py['/value/img'][index, 0]]).transpose((2, 1, 0))
        target = np.array(self.h5py[self.h5py[self.h5py['/value/gt'][index, 0]][0, 0]]).transpose((1, 0))
        res = np.load(self.results[index])

        img = Image.fromarray(img).convert('RGB')
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = np.array(self.target_transform(target)).astype('int32')
            target = torch.from_numpy(target).long()

        return img, target

    def __len__(self):
        return len(self.h5py['/value/img'])


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from tqdm import tqdm
    from imageio import imsave
    import scipy.io as sio

    # meta = sio.loadmat('/home/shirgur/ext/Data/Datasets/temp/ILSVRC2012_devkit_t12/data/meta.mat', squeeze_me=True)['synsets']

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_img_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    test_lbl_trans = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
    ])

    ds = Imagenet_Segmentation('/home/shirgur/ext/Data/Datasets/imagenet-seg/other/gtsegs_ijcv.mat',
                               transform=test_img_trans, target_transform=test_lbl_trans)

    for i, (img, tgt) in enumerate(tqdm(ds)):
        tgt = (tgt.numpy() * 255).astype(np.uint8)
        imsave('/home/shirgur/ext/Code/C2S/run/imagenet/gt/{}.png'.format(i), tgt)

    print('here')
