import torch.nn.functional as F
import torch
from Config import configure
def train(Nets, Optimizers, train_loader, train_dataset, DEVICE):

    netG, netD = Nets
    optG, optD = Optimizers

    batch_size = configure.batch_size

    train_length = len(train_dataset)
    iter_per_epoch = train_length // batch_size
    all_iter = configure.all_iter
    epochs = all_iter // iter_per_epoch

    for epoch in range(epochs):
        if epoch == epochs // 2:  # 10^5 iter가 지나면 러닝레이트를 줄여준다.
            optG = torch.optim.Adam(netG.parameters(), lr=1e-5)
            optD = torch.optim.Adam(netD.parameters(), lr=1e-5)

        print("{}/{} Epochs".format(epoch, epochs))
        for lr_image, hr_image in train_loader:
            lr_image = lr_image.to(DEVICE)
            hr_image = hr_image.to(DEVICE)

            # Discriminator 훈련
            pred_image = netG(lr_image)
            pred_fake, fake_content = netD(pred_image.detach())
            pred_real, real_contet = netD(hr_image)
            dis_loss = F.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake)) + F.binary_cross_entropy(pred_real,
                                                                                                               torch.ones_like(
                                                                                                                   pred_real))
            dis_loss *= 1e-3
            optD.zero_grad()
            dis_loss.backward()
            optD.step()

            # generator 훈련
            # content loss는 generator에 해당하는 부분이다.
            pred_fake, fake_content = netD(pred_image)
            pred_real, real_content = netD(hr_image)
            gen_loss = 1e-3 * F.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake)) + F.mse_loss(fake_content,
                                                                                                         real_content)
            optG.zero_grad()
            gen_loss.backward()
            optG.step()

        print(dis_loss.item(), gen_loss.item())


