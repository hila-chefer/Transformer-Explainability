import torch
from torch.utils.data import Dataset
import h5py
import os


class ImagenetResults(Dataset):
    def __init__(self, path):
        super(ImagenetResults, self).__init__()

        self.path = os.path.join(path, 'results.hdf5')
        self.data = None

        print('Reading dataset length...')
        with h5py.File(self.path , 'r') as f:
        # tmp = h5py.File(self.path , 'r')
            self.data_length = len(f['/image'])

    def __len__(self):
        return self.data_length

    def __getitem__(self, item):
        if self.data is None:
            self.data = h5py.File(self.path, 'r')

        image = torch.tensor(self.data['image'][item])
        vis = torch.tensor(self.data['vis'][item])
        target = torch.tensor(self.data['target'][item]).long()

        return image, vis, target


if __name__ == '__main__':
    from utils import render
    import imageio
    import numpy as np

    ds = ImagenetResults('../visualizations/fullgrad')
    sample_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=5,
        shuffle=False)

    iterator = iter(sample_loader)
    image, vis, target = next(iterator)

    maps = (render.hm_to_rgb(vis[0].data.cpu().numpy(), scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)

    # imageio.imsave('../delete_hm.jpg', maps)

    print(len(ds))