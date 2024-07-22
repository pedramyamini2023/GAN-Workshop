from dataset import trainloader
from generator import getGenerator
from discriminator import getDiscriminator
import torch.nn as nn
import config
from tqdm import tqdm
import torch



def train(gen, opt_gen, scr_gen, disc, opt_disc, scr_disc, loader, bce, l1):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        fake = gen(x)
        # train discriminator
        with torch.cuda.amp.autocast():
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))

            D_fake = disc(x, fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))

            loss_disc = (D_real_loss + D_fake_loss)/2
        
        disc.zero_grad()
        scr_disc.scale(loss_disc).backward()
        scr_disc.step(opt_disc)
        scr_disc.update()

        # train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, fake.detach())
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            G_L1_fake = l1(fake, y) * config.LAMBDA
            
            G_loss = G_fake_loss + G_L1_fake

        gen.zero_grad()
        scr_gen.scale(G_loss).backward()
        scr_gen.step(opt_gen)
        scr_gen.update()

        loop.set_postfix(
           disc =  loss_disc.item(),
           gen = G_loss.item()
        )

    return gen, opt_gen, scr_gen, disc, opt_disc, scr_disc








def main():
    # dataset
    loader = trainloader()
    # generator
    gen, opt_gen, scr_gen = getGenerator()
    # discriminator
    disc, opt_disc, scr_disc = getDiscriminator()

    # training

    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    for epoch in range(config.NUM_EPOCH):
        # train
        gen, opt_gen, scr_gen, disc, opt_disc, scr_disc = train(gen, opt_gen, scr_gen, disc, opt_disc, scr_disc, loader, BCE, L1)
        # save model
        config.save_checkpoint(gen, opt_gen, config.GEN_PATH)
        config.save_checkpoint(disc, opt_disc, config.DISC_PATH)
        # save some example
        config.save_example(gen, loader, epoch, "outcomes/")


if __name__ == "__main__":
    main()
