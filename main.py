import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.dataset import AEBSVideoDataset, DummyDataset
from src.model import Generator, VideoDiscriminator, FrameDiscriminator
from src.trainer import train
import torch.nn as nn
import os
import time
import logging
import json
import timeit

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dataset', type=str, default='aebs', help='Name of the dataset')
    parser.add_argument('--seq_len', type=int, default=64, help='Length of video sequence')
    parser.add_argument('--env_noise_dim', type=int, default=32, help='Dimension of environment noise')
    parser.add_argument('--env_emb_dim', type=int, default=32, help='Dimension of environment embedding')
    parser.add_argument('--cond_dim', type=int, default=1, help='Dimension of conditions')
    parser.add_argument('--cond_emb_dim', type=int, default=32, help='Dimension of conditional embedding')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of layers in RNN')
    parser.add_argument('--condition_on_rnn', type=bool, default=False, help='Condition on RNN')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning rate for generator')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='Learning rate for discriminator')
    parser.add_argument('--resume_num_iters', type=int, default=0, help='Resume number of iterations')
    parser.add_argument('--num_iters', type=int, default=20000, help='Number of iterations')
    parser.add_argument('--num_D_steps', type=int, default=2, help='Number of steps to train discriminator')
    parser.add_argument('--condition_lambda', type=float, default=10.0, help='Lambda for conditioning loss')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--log_frequency', type=int, default=100, help='Log frequency')
    parser.add_argument('--visualize_frequency', type=int, default=1000, help='Visualize frequency')
    parser.add_argument('--save_frequency', type=int, default=1000, help='Save frequency')
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"

    assert args.dataset in ['aebs', 'dummy'], "Dataset must be either 'aebs' or 'dummy'"
    assert args.seq_len == 64, "Sequence length must be 64"

    # create output folder if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    subfolder_name = time.strftime("%Y%m%d-%H%M%S")
    result_folder = os.path.join(args.output_dir, subfolder_name)
    os.makedirs(result_folder)
    models_foler = os.path.join(result_folder, 'models')
    os.makedirs(models_foler)
    generated_videos_folder = os.path.join(result_folder, 'videos')
    os.makedirs(generated_videos_folder)

    # set up logging
    logging.basicConfig(filename=os.path.join(result_folder, 'log.txt'), 
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        filemode='a')

    # log arguments
    with open(os.path.join(result_folder, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    logging.info('Arguments: %s', json.dumps(vars(args), indent=4))

    # perapare dataset
    if args.dataset == 'aebs':
        # Load AEBS dataset
        dataset = AEBSVideoDataset(seq_len=args.seq_len)
        logging.info("Loaded AEBS dataset")
    else:
        # Load dummy dataset
        dataset = DummyDataset(seq_len=args.seq_len)
        logging.info("Loaded dummy dataset")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # prepare model
    logging.info("Preparing model")
    Gen = Generator(args.cond_dim, args.cond_emb_dim, args.env_noise_dim, args.env_emb_dim, args.num_rnn_layers, args.condition_on_rnn)
    Gen = nn.DataParallel(Gen)
    Gen.cuda()

    VDis = VideoDiscriminator()
    VDis = nn.DataParallel(VDis)
    VDis.cuda()

    FDis = FrameDiscriminator(args.cond_dim)
    FDis = nn.DataParallel(FDis)
    FDis.cuda()

    if args.resume_num_iters > 0:
        raise NotImplementedError("Resume is not implemented yet")


    # prepare optimizer
    logging.info("Preparing optimizer")
    optim_Gen = torch.optim.Adam(Gen.parameters(), args.g_lr, (0.5, 0.999))
    optim_VDis = torch.optim.Adam(VDis.parameters(), args.d_lr, (0.5, 0.999))
    optim_FDis = torch.optim.Adam(FDis.parameters(), args.d_lr, (0.5, 0.999))

    # train
    batch_idx = 0
    start_time = timeit.default_timer()
    for niter in range(args.resume_num_iters, args.num_iters):
        models = (Gen, VDis, FDis)
        optims = (optim_Gen, optim_VDis, optim_FDis)
        batch_idx, d_loss, g_loss, video_dis_loss_real, image_dis_loss_real, \
            video_dis_loss_fake, image_dis_loss_fake, dis_reg_loss, dis_reg_l1_loss, \
                video_gen_adv_loss, image_gen_adv_loss, gen_reg_loss, gen_reg_l1_loss = \
                    train(models, optims, dataloader, batch_idx, args)
        
        if (niter+1) % args.log_frequency == 0:
            logging_str = "video-cGAN-%s: [Iter %d/%d] [D loss:%.4f] [G loss:%.4f] [Video D adv real:%.4f] [Video D adv fake:%.4f] [Image D adv real:%.4f] [Image D adv fake:%.4f] [D reg loss (aebs/l1):%.4f/%.4f] [Video G adv loss:%.4f] [Image G adv loss:%.4f] [G reg loss (aebs/l1):%.4f/%.4f] [Time:%.4f]" %\
                        (args.dataset, niter+1, args.num_iters, d_loss, g_loss, 
                        video_dis_loss_real, video_dis_loss_fake, image_dis_loss_real, image_dis_loss_fake,
                        dis_reg_loss, dis_reg_l1_loss,
                        video_gen_adv_loss, image_gen_adv_loss, 
                        gen_reg_loss, gen_reg_l1_loss, timeit.default_timer()-start_time)
            print(logging_str)
            logging.info(logging_str)
        
        if (niter+1) % args.visualize_frequency == 0:
            n_row = 8
            start_label = 0.05
            end_label = 0.95
            selected_labels = torch.linspace(start_label, end_label, args.seq_len).cuda()
            conditions = selected_labels.unsqueeze(0).repeat(n_row, 1).unsqueeze(-1)
            h0 = torch.zeros(Gen.module.num_rnn_layers, n_row, Gen.module.env_emb_dim).cuda()
            env_noises = torch.randn(n_row, args.seq_len, Gen.module.env_noise_dim).cuda()
            Gen.eval()
            with torch.no_grad():
                fake_video = Gen(h0, env_noises, conditions).detach()
            sampled_fake_video = fake_video[:, 0::8, :, :, :].reshape(-1, fake_video.shape[-3], fake_video.shape[-2], fake_video.shape[-1])
            video_file_path = os.path.join(generated_videos_folder, 'video_cGAN_%d.png' % (niter+1))
            save_image(sampled_fake_video.data, video_file_path, nrow=n_row, normalize=True)

        
        if (niter+1) % args.save_frequency == 0:
            model_file_path = os.path.join(models_foler, 'video_cGAN_checkpoint_niters_%d.pth' % (niter+1))
            torch.save({
                'netG_state_dict': Gen.state_dict(),
                'netVD_state_dict': VDis.state_dict(),
                'netFD_state_dict': FDis.state_dict(),
                'optimizerG_state_dict': optim_Gen.state_dict(),
                'optimizerVD_state_dict': optim_VDis.state_dict(),
                'optimizerFD_state_dict': optim_FDis.state_dict(),
                'rng_state': torch.get_rng_state(),
            }, model_file_path)


if __name__ == "__main__":
    main()