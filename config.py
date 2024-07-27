import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Train_DIR = "/kaggle/working/cityscapes/cityscapes/train"
Test_DIR = "/kaggle/working/cityscapes/cityscapes/val"
#-------------------------------------------------------------
BATCH_SIZE = 8
IMAGE_SIZE = 256
mean = [.5, .5, .5]
std = [.5, .5, .5]
num_workers = 2
LEARNING_RATE = 1e-4
LAMBDA = 100
NUM_EPOCH = 100

DISC_PATH = "/kaggle/working/drive/MyDrive/gan_checkpoints/disc.pth.tar"
GEN_PATH = "/kaggle/working/drive/MyDrive/gan_checkpoints/gen.pth.tar"

LOAD_MODEL = True

mytfsm = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize(mean, std)   
])


def save_example(gen, loader, epoch, folder):
    x, y = next(iter(loader))
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        fake = gen(x)
        fake = fake*.5 + .5
        output = torch.cat([x*.5+.5, fake, y*.5+.5], 0)
        save_image(output, folder + f"/fake_{epoch}.png")

    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
