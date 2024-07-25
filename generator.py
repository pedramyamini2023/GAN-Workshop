import torch
import torch.nn as nn
import config
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, in_channel = 3, out_channel=3, features = 64):
        super(Generator, self).__init__()
        self.down1 = self._down(in_channel, features)
        self.down2 = self._down(features, features*2)
        self.down3 = self._down(features*2, features*4)
        self.down4 = self._down(features*4, features*8)
        self.down5 = self._down(features*8, features*8)
        self.down6 = self._down(features*8, features*8)
        self.down7 = self._down(features*8, features*8)
        self.down8 = self._down(features*8, features*8)
        #---------------------------------------------
        self.up1 = self._up(features*8, features*8)
        self.up2 = self._up(features*8*2, features*8)
        self.up3 = self._up(features*8*2, features*8)
        self.up4 = self._up(features*8*2, features*8)
        self.up5 = self._up(features*8*2, features*4)
        self.up6 = self._up(features*4*2, features*2)
        self.up7 = self._up(features*2*2, features)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channel, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        # self.up8 = self._up(features*2, out_channel)


    def _down(self, in_channel, out_channel, kernel=4, straid=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, straid, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(.2)
        )
    
    def _up(self, in_channel, out_channel, kernel=4, straid=2, padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel, straid, padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),

            # nn.ReLU() if atv=="relue" else nn.Tanh(),
        )


    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6) #2.2
        d8 = self.down8(d7)
        #----------------------------
        up1 = self.up1(d8) # 2.2 
        up2 = self.up2(torch.cat([up1, d7], 1)) 
        up3 = self.up3(torch.cat([up2, d6], 1)) 
        up4 = self.up4(torch.cat([up3, d5], 1)) 
        up5 = self.up5(torch.cat([up4, d4], 1))  
        up6 = self.up6(torch.cat([up5, d3], 1))   
        up7 = self.up7(torch.cat([up6, d2], 1))  
        up8 = self.up8(torch.cat([up7, d1], 1))   


        return up8


def getGenerator():
    gen = Generator().to(config.DEVICE)
    opt = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(.5, .999))
    scr = torch.cuda.amp.GradScaler()

    try:
        if config.LOAD_MODEL:
            config.load_checkpoint(config.GEN_PATH, gen, opt, config.LEARNING_RATE)
    except:
        pass

    return gen, opt, scr



if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    gen = Generator(in_channel=3)
    fake = gen(x)

    print(fake.shape)

    print(fake.min(), fake.max())