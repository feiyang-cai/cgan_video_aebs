import torch.nn as nn
import torch
from src.network_modules import *


class Generator(nn.Module):
    def __init__(self, cond_dim, cond_emb_dim, env_noise_dim, env_emb_dim, num_rnn_layers, condition_on_rnn, g_conv_dim=16) -> None:
        super(Generator, self).__init__()
        self.condition_on_rnn = condition_on_rnn
        self.num_rnn_layers = num_rnn_layers
        self.env_emb_dim = env_emb_dim
        self.env_noise_dim = env_noise_dim
        self.cond_emb_dim = cond_emb_dim

        self.condition_embedding = snlinear(cond_dim, cond_emb_dim)
        if condition_on_rnn:
            self.rnn = nn.GRU(env_noise_dim + cond_emb_dim, env_emb_dim, num_rnn_layers, batch_first=True)
            input_dim = env_emb_dim
        else:
            self.rnn = nn.GRU(env_noise_dim, env_emb_dim, num_rnn_layers, batch_first=True)
            input_dim = env_emb_dim + cond_emb_dim

        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(input_dim, g_conv_dim * 16 * 4 * 4)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        #self.self_attn = Self_Attn(g_conv_dim*16)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.block3 = GenBlock(g_conv_dim*16, g_conv_dim*16)
        self.bn = nn.BatchNorm2d(g_conv_dim*16, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim*16, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, h0, env_noises, conditions):
        # h0: (num_rnn_layers, batch_size, env_emb_dim)
        # env_noises: (batch_size, seq_len, env_noise_dim)
        # conditions: (batch_size, seq_len)
        batch_size, seq_len, _ = env_noises.shape

        # condition embedding
        cond_embs = self.condition_embedding(conditions) # (batch_size, seq_len, cond_emb_dim)
        if self.condition_on_rnn:
            # rnn
            env_embs, _ = self.rnn(torch.cat([env_noises, cond_embs], dim=-1), h0)
            # generate video
            x = env_embs.reshape(-1, self.env_emb_dim) # (batch_size x seq_len, env_emb_dim)
        else:
            # rnn
            env_embs, _ = self.rnn(env_noises, h0)        
            # generate video
            x = torch.cat([env_embs.reshape(-1, self.env_emb_dim), cond_embs.reshape(-1, self.cond_emb_dim)], dim=-1) # (batch_size x seq_len, env_emb_dim + cond_emb_dim)
        
        act0 = self.snlinear0(x)            # n x g_conv_dim*16*1*1
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 1 x 1
        act1 = self.block1(act0)    # n x g_conv_dim*16 x 2 x 2
        act2 = self.block2(act1)    # n x g_conv_dim*8 x 4 x 4
        #act2 = self.self_attn(act2)         # n x g_conv_dim*4 x 8 x 8
        act3 = self.block3(act2)    # n x g_conv_dim*4 x 8 x 8
        act5 = self.bn(act3)                # n x g_conv_dim  x 32 x 32
        act5 = self.relu(act5)              # n x g_conv_dim  x 32 x 32
        act6 = self.snconv2d1(act5)         # n x 3 x 32 x 32
        act6 = self.tanh(act6)              # n x 3 x 32 x 32

        videos = act6.reshape(batch_size, seq_len, act6.shape[-3], act6.shape[-2], act6.shape[-1]) # (batch_size, seq_len, 3, 32, 32)
        return videos

class FrameDiscriminator(nn.Module):
    """Discriminator."""

    def __init__(self, cond_dim, d_conv_dim=16):
        super(FrameDiscriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.block1 = DiscBlock(3, d_conv_dim*8)
        #self.self_attn = Self_Attn(d_conv_dim*8)
        self.block2 = DiscBlock(d_conv_dim*8, d_conv_dim*8)
        self.block3 = DiscBlock(d_conv_dim*8, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16*8*8, out_features=1)
        self.snlinear2 = snlinear(in_features=d_conv_dim*16*8*8, out_features=cond_dim)

    def forward(self, x):
        # x: (batch_size, 3, 32, 32)
        
        h1 = self.block1(x)    
        #h1 = self.self_attn(h1) 
        h2 = self.block2(h1)    
        h3 = self.block3(h2, False)    
        h4 = self.block4(h3, False)    
        out = self.relu(h4)              
        out = out.view(-1,self.d_conv_dim*16*8*8)

        dis = torch.squeeze(self.snlinear1(out))
        cond_pred = torch.squeeze(torch.sigmoid(self.snlinear2(out)))

        return dis, cond_pred

class VideoDiscriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim=16):
        super(VideoDiscriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.block1 = Disc3DBlock(3, d_conv_dim*8)
        #self.self_attn = Self_Attn(d_conv_dim*8*32)
        self.block2 = Disc3DBlock(d_conv_dim*8, d_conv_dim*8)
        self.block3 = Disc3DBlock(d_conv_dim*8, d_conv_dim*8)

        self.block4 = Disc3DBlock(d_conv_dim*8, d_conv_dim*8)
        self.block5 = Disc3DBlock(d_conv_dim*8, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16*8*4*4, out_features=1)

    def forward(self, x):
        # x: (batch_size, seq_len, 3, 32, 32)
        x = torch.permute(x, (0, 2, 1, 3, 4)) # (batch_size, 3, seq_len, 32, 32)
        
        h1 = self.block1(x)   
        #h1 = self.self_attn(h1) 
        #print(h1.shape)
        h2 = self.block2(h1)  
        h3 = self.block3(h2)
        h4 = self.block4(h3, False)    
        h5 = self.block5(h4, False)    

        out = self.relu(h5)              
        out = out.view(-1,self.d_conv_dim*16*8*4*4)

        dis = torch.squeeze(self.snlinear1(out))

        return dis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--env_noise_dim', type=int, default=32, help='Dimension of environment noise')
    parser.add_argument('--env_emb_dim', type=int, default=32, help='Dimension of environment embedding')
    parser.add_argument('--cond_dim', type=int, default=1, help='Dimension of conditions')
    parser.add_argument('--cond_emb_dim', type=int, default=32, help='Dimension of conditional embedding')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of layers in RNN')
    parser.add_argument('--condition_on_rnn', type=bool, default=False, help='Condition on RNN')
    args = parser.parse_args()

    batch_size = 4
    seq_len = 64

    Gen = Generator(args.cond_dim, args.cond_emb_dim, args.env_noise_dim, args.env_emb_dim, args.num_rnn_layers, args.condition_on_rnn)
    h0 = torch.randn(args.num_rnn_layers, batch_size, args.env_emb_dim)
    env_noises = torch.randn(batch_size, seq_len, args.env_noise_dim)
    conditions = torch.randn(batch_size, seq_len, args.cond_dim)
    videos = Gen(h0, env_noises, conditions)
    print(videos.shape)

    VDis = VideoDiscriminator()
    dis_video = VDis(videos)
    print(dis_video.shape)
    FDis = FrameDiscriminator(args.cond_dim)
    dis_frames, cond_pred = FDis(videos.reshape(-1, videos.shape[-3], videos.shape[-2], videos.shape[-1]))
    print(dis_frames.shape, cond_pred.shape)