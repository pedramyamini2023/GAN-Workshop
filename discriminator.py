import torch
import torch.nn as nn
import config
import torch.optim as optim


class Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, features = 64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            self._down(in_channel*2, features),
            self._down(features, features*2),
            self._down(features*2, features*4),
            self._down(features*4, 1, kernel=4, straid=1, padding=1),
        )


    def _down(self, in_channel, out_channel, kernel=4, straid=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, straid, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(.2)
        )


    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1)) 
    

def getDiscriminator():
    disc = Discriminator().to(config.DEVICE)
    opt = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(.5, .999))
    scr = torch.cuda.amp.GradScaler()

    try:
        if config.LOAD_MODEL:
            config.load_checkpoint(config.DISC_PATH, disc, opt, config.LEARNING_RATE)
    except:
        pass

    return disc, opt, scr


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    y = torch.randn((8, 3, 256, 256))
    disc = Discriminator(in_channel=3)
    out = disc(x, y)

    print(out.shape)