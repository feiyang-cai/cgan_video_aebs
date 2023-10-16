import logging
import random
import torch

class AEBSRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(AEBSRegressionLoss, self).__init__()

    def forward(self, reg_output, label, **_):
        assert reg_output.shape == label.shape
        return torch.mean(torch.abs(reg_output - label)/(label+0.01))

def train(models, optims, dataloader, batch_idx, args):
    Gen, VDis, FDis = models
    optim_Gen, optim_VDis, optim_FDis = optims

    Gen.train()
    VDis.train()
    FDis.train()

    reg_loss = AEBSRegressionLoss()

    dataloader_iter = iter(dataloader)

    if batch_idx+1 == len(dataloader):
        dataloader_iter = iter(dataloader)
        batch_idx = 0


    # train G
    _, conditions = next(dataloader_iter)
    bs = conditions.size(0)
    seq_len = conditions.size(1)
    conditions = conditions.reshape(bs, seq_len, -1).cuda()
    batch_idx += 1

    ## sample noise
    h0 = torch.zeros(Gen.module.num_rnn_layers, bs, Gen.module.env_emb_dim).cuda()
    env_noises = torch.randn(bs, seq_len, Gen.module.env_noise_dim).cuda()

    ## generate fake video and images
    fake_video = Gen(h0, env_noises, conditions)
    fake_video_dis = VDis(fake_video)
    fake_images_dis, fake_images_reg = FDis(fake_video.reshape(-1, fake_video.shape[-3], fake_video.shape[-2], fake_video.shape[-1]))

    ## Loss measures generator's ability to fool the discriminator
    gen_reg_loss = reg_loss(fake_images_reg, conditions.reshape(-1))
    gen_reg_l1_loss = torch.nn.L1Loss()(fake_images_reg, conditions.reshape(-1))

    ### FC: using hinge loss
    video_gen_adv_loss = -fake_video_dis.mean()
    image_gen_adv_loss = -fake_images_dis.mean()

    g_loss = video_gen_adv_loss + image_gen_adv_loss + args.condition_lambda*gen_reg_loss
        
    optim_Gen.zero_grad()
    g_loss.backward()
    optim_Gen.step()
        
        
    # train D 

    for _ in range(args.num_D_steps):

        if batch_idx+1 == len(dataloader):
            dataloader_iter = iter(dataloader)
            batch_idx = 0

        real_video, conditions = next(dataloader_iter)
        bs = conditions.size(0)
        seq_len = conditions.size(1)
        real_video = real_video.cuda()
        conditions = conditions.reshape(bs, seq_len, -1).cuda()
        real_images_labels = conditions.reshape(-1)
        batch_idx += 1

        real_video_dis = VDis(real_video)
        real_images_dis, real_images_reg = FDis(real_video.reshape(-1, real_video.shape[-3], real_video.shape[-2], real_video.shape[-1]))

        fake_video_dis = VDis(fake_video.detach())
        fake_images_dis, _ = FDis(fake_video.detach().reshape(-1, fake_video.shape[-3], fake_video.shape[-2], fake_video.shape[-1]))

        video_dis_loss_real = torch.nn.ReLU()(1.0 - real_video_dis).mean()
        image_dis_loss_real = torch.nn.ReLU()(1.0 - real_images_dis).mean()
        video_dis_loss_fake = torch.nn.ReLU()(1.0 + fake_video_dis).mean()
        image_dis_loss_fake = torch.nn.ReLU()(1.0 + fake_images_dis).mean()

        dis_reg_loss = reg_loss(real_images_reg, real_images_labels)
        dis_reg_l1_loss = torch.nn.L1Loss()(real_images_reg, real_images_labels)

        dis_loss_real = video_dis_loss_real + image_dis_loss_real + args.condition_lambda*dis_reg_loss
        dis_loss_fake = video_dis_loss_fake + image_dis_loss_fake

        d_loss = dis_loss_real + dis_loss_fake

        optim_FDis.zero_grad()
        optim_VDis.zero_grad()
        d_loss.backward()
        optim_FDis.step()
        optim_VDis.step()

    return batch_idx, d_loss.item(), g_loss.item(), \
            video_dis_loss_real.item(), image_dis_loss_real.item(), \
                video_dis_loss_fake.item(), image_dis_loss_fake.item(), \
                    dis_reg_loss.item(), dis_reg_l1_loss.item(), \
                        video_gen_adv_loss.item(), image_gen_adv_loss.item(), \
                            gen_reg_loss.item(), gen_reg_l1_loss.item()