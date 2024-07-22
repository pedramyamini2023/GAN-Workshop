from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import config
from torchvision.utils import save_image


class mydataset(Dataset):
    def __init__(self, map_path):
        self.map_path = map_path
        self.list_file = os.listdir(map_path)
    def __len__(self):
        return len(self.list_file)
    def __getitem__(self, index):
        path = self.list_file[index]
        path = os.path.join(self.map_path, path)
        img = Image.open(path).convert("RGB")
        x = np.array(img)[:, :256, :]
        y = np.array(img)[:, 256:, :]
        x = config.mytfsm(Image.fromarray(x))
        y = config.mytfsm(Image.fromarray(y))
        return x, y


def trainloader():
    dataset = mydataset(config.Train_DIR)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.num_workers)
    return loader


# if __name__ == "__main__":
#     dataset = mydataset(config.Train_DIR)
#     loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

#     x, y = next(iter(loader))
#     print(x.shape)
#     print(y.shape)
#     save_image(x*.5 + .5, "x.png")
#     save_image(y*.5 +.5, "y.png")
#     # [batch_size, channel, H, W]
