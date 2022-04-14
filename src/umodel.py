'''
umodel.py

* Auxiliary functions:
    - RGB to YCbCr
    - Pixel unshuffle
    - Convolution layers
* Subnets:
    - PrepHidingNet
    - RevealNet
* Main net: StegoUNet
'''


import torch
import numpy as np
import torch.nn as nn
from torch import utils
import torch.nn.functional as F
from pystct import sdct_torch, isdct_torch




def rgb_to_ycbcr(img):
    # Taken from https://www.w3.org/Graphics/JPEG/jfif3.pdf
    # img is mini-batch N x 3 x H x W of an RGB image

    output = torch.zeros(img.shape).to(img.device)

    output[:,0,:,:] =  0.2990 * img[:,0,:,:] + 0.5870 * img[:,1,:,:] + 0.1114 * img[:,2,:,:]
    output[:,1,:,:] = -0.1687 * img[:,0,:,:] - 0.3313 * img[:,1,:,:] + 0.5000 * img[:,2,:,:] + 128
    output[:,2,:,:] =  0.5000 * img[:,0,:,:] - 0.4187 * img[:,1,:,:] - 0.0813 * img[:,2,:,:] + 128

    return output

def ycbcr_to_rgb(img):
    # Taken from https://www.w3.org/Graphics/JPEG/jfif3.pdf
    # img is mini-batch N x 3 x H x W of a YCbCr image

    output = torch.zeros(img.shape).to(img.device)

    output[:,0,:,:] =  img[:,0,:,:] + 1.40200 * (img[:,2,:,:]-128)
    output[:,1,:,:] =  img[:,0,:,:] - 0.34414 * (img[:,1,:,:]-128) - 0.71414*(img[:,2,:,:]-128)
    output[:,2,:,:] =  img[:,0,:,:] + 1.77200 * (img[:,1,:,:]-128)

    return output

def pixel_unshuffle(img, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = img.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                         1, downscale_factor, downscale_factor],
                         device=img.device, dtype=img.dtype)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(img, kernel, stride=downscale_factor, groups=c)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, img):
        return pixel_unshuffle(img, self.downscale_factor)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class Down(nn.Module):

    def __init__(self, in_channels, out_channels, downsample_factor=8):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.MaxPool2d(downsample_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, opp_channels=-1):
        # opp_channels -> The number of channels (depth) of the opposite replica of the unet
        #                   If -1, the same number as the current image is assumed
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels , out_channels, kernel_size=3, stride=4, output_padding=0),
            nn.LeakyReLU(0.8, inplace=True),
            nn.ConvTranspose2d(out_channels , out_channels, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.8, inplace=True),
        )
        if opp_channels == -1:
            opp_channels = out_channels
        self.conv = DoubleConv(opp_channels+out_channels, out_channels)

    def forward(self, mix, im_opposite, au_opposite = None):
        mix = self.up(mix)
        x = torch.cat((mix, im_opposite), dim=1)
        return self.conv(x)





class PrepHidingNet(nn.Module):
    def __init__(self, transform='cosine', embed='stretch'):
        super(PrepHidingNet, self).__init__()
        self._transform = transform
        self.embed = embed
    
        self.pixel_shuffle = nn.PixelShuffle(2)

        if self.embed == 'multichannel':
            # In multichannel we get the 3 color channels and output the 8 replicas
            self.im_encoder_layers = nn.ModuleList([
                Down(3, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64),
                Up(64, 8, opp_channels=3)
            ])
        else:
            self.im_encoder_layers = nn.ModuleList([
                Down(1, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64),
                Up(64, 1)
            ])
    
    
    def forward(self, im):

        if self.embed != 'multichannel':
            im = self.pixel_shuffle(im)

        if self.embed == 'stretch' or self.embed == 'luma':
        # Stretch the image to make it the same shape as the container (different for STDCT and STFT)
            if self._transform == 'cosine':
                im = nn.Upsample(scale_factor=(8, 2), mode='bilinear',align_corners=True)(im)
            elif self._transform == 'fourier':
                im = nn.Upsample(scale_factor=(2, 1), mode='bilinear',align_corners=True)(im)
            else: raise Exception(f'Transform not implemented')

        im_enc = [im]
        
        # Encoder part of the UNet
        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            im_enc.append(enc_layer(im_enc[-1]))

        mix_dec = [im_enc.pop(-1)]

        # Decoder part of the UNet
        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            mix_dec.append(dec_layer(mix_dec[-1], im_enc[-1 - dec_layer_idx], None))

        return mix_dec[-1]



class RevealNet(nn.Module):
    def __init__(self, mp_decoder=None, embed='stretch'):
        super(RevealNet, self).__init__()

        self.mp_decoder = mp_decoder
        self.pixel_unshuffle = PixelUnshuffle(2)
        self.embed = embed

        # If mp_decoder == unet or concatenating blocks, have RevealNet accept 2 channels as input instead of 1
        if self.mp_decoder == 'unet' or self.embed == 'blocks3':
            self.im_encoder_layers = nn.ModuleList([
                Down(2, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64),
                Up(64, 1, opp_channels=2)
            ])
        elif self.embed == 'multichannel':
            self.im_encoder_layers = nn.ModuleList([
                Down(8, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64),
                Up(64, 3, opp_channels=8)
            ])
        else:
            self.im_encoder_layers = nn.ModuleList([
                Down(1, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64),
                Up(64, 1)
            ])

        if self.embed == 'blocks2':
            self.decblocks = nn.Parameter(torch.rand(2))


    def forward(self, ct, ct_phase=None):

        # ct_phase is not None if and only if mp_decoder == unet
        # For other decoder types, ct is the only container
        assert not (self.mp_decoder == 'unet'  and ct_phase is None)
        assert not (self.mp_decoder != 'unet'  and ct_phase is not None)


        # Stretch the container to make it the same size as the image
        if self.embed == 'stretch':
            ct = F.interpolate(ct, size=(256 * 2, 256 * 2))
            if self.mp_decoder == 'unet':
                ct_phase = [F.interpolate(ct_phase, size=(256 * 2, 256 * 2))]
        
        if self.mp_decoder == 'unet':
            # Concatenate mag and phase containers to input to RevealNet
            im_enc = [torch.cat((im_enc, im_enc_phase), 1)]
        elif self.embed == 'blocks3':
            # Undo split and concatenate in another dimension
            (rep1, rep2) = torch.split(ct, 512, 2)
            im_enc = [torch.cat((rep1, rep2), 1)]
        elif self.embed == 'multichannel':
            # Split the eight replicas and concatenate. 1x1x1024x512 -> 1x8x256x256
            split1 = torch.split(ct, 256, 3)
            cat1 = torch.cat(split1, 1)
            split2 = torch.split(cat1, 256, 2)
            im_enc = [torch.cat(split2, 1)]
        else:
            # Else there is only one container (can be anything)
            im_enc = [ct]

        # Encoder part of the UNet
        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            im_enc.append(enc_layer(im_enc[-1]))

        im_dec = [im_enc.pop(-1)]

        # Decoder part of the UNet
        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            im_dec.append(
                dec_layer(im_dec[-1],
                im_enc[-1 - dec_layer_idx])
            )

        if self.embed == 'multichannel':
            # The revealed image is the output of the U-Net
            revealed = im_dec[-1]
        elif self.embed == 'luma':
            # Convert RGB to YUV, average lumas and back to RGB
            unshuffled = self.pixel_unshuffle(im_dec[-1])
            print('unshuffled:', unshuffled.shape)
            rgbs = torch.narrow(unshuffled, 1, 0, 3)
            luma = unshuffled[:,3,:,:]
            print('rgbs:', rgbs.shape)
            print('luma:', luma.shape)

            yuvs = rgb_to_ycbcr(rgbs)
            print('yuvs:', yuvs.shape)
            print('dev yuvs:', yuvs.device)
            print('dev rgbs:', rgbs.device)
            print('dev luma:', luma.device)

            yuvs[:,0,:,:] = 0.5*yuvs[:,0,:,:] + 0.5*luma
            print('yuvs:', yuvs.shape)
            assert(False)

            revealed = ycbcr_to_rgb(yuvs)
        else:
            # Pixel Unshuffle and delete 4th component
            revealed = torch.narrow(self.pixel_unshuffle(im_dec[-1]), 1, 0, 3)

        if self.embed == 'blocks':
            # Undo concatenation and recover a single image
            (rev1, rev2) = torch.split(revealed, 256, 2)
            # Average them
            revealed = torch.mean(torch.cat((rev1, rev2), 0), 0).unsqueeze(0)
        elif self.embed == 'blocks2':
            # Undo concatenation and recover a single image
            (rev1, rev2) = torch.split(revealed, 256, 2)
            # Scale and add
            revealed = rev1*self.decblocks[0] + rev2*self.decblocks[1]

        return revealed




class StegoUNet(nn.Module):
    def __init__(self, transform='cosine', ft_container='mag', mp_encoder='single', mp_decoder='double', mp_join='mean', permutation=False, embed='stretch'):

        super().__init__()

        self.transform = transform
        self.ft_container = ft_container
        self.mp_encoder = mp_encoder
        self.mp_decoder = mp_decoder
        self.mp_join = mp_join
        self.permutation = permutation
        self.embed = embed
        
        if transform != 'fourier' or ft_container != 'magphase':
            self.mp_decoder = None # For compatiblity with RevealNet

        # Sub-networks
        self.PHN = PrepHidingNet(self.transform, self.embed)
        self.RN = RevealNet(self.mp_decoder, self.embed)
        if transform == 'fourier' and ft_container == 'magphase':
            # The previous one is for the magnitude. Create a second one for the phase
            if mp_encoder == 'double':
                self.PHN_phase = PrepHidingNet(self.transform)
            if mp_decoder == 'double':
                self.RN_phase = RevealNet(self.mp_decoder)
                if mp_join == '2D':
                    self.mag_phase_join = nn.Conv2d(6,3,1)
                elif mp_join == '3D':
                    self.mag_phase_join = nn.Conv3d(2,1,1)

        if self.embed == 'blocks2' or self.embed == 'blocks3':
            self.encblocks = nn.Parameter(torch.rand(2))

    def forward(self, secret, cover, cover_phase=None):
        # cover_phase is not None if and only if using mag+phase
        # If using the phase only, 'cover' is actually the phase!
        assert not ((self.transform == 'fourier' and self.ft_container == 'magphase') and cover_phase is None)
        assert not ((self.transform == 'fourier' and self.ft_container != 'magphase') and cover_phase is not None)

        if self.embed != 'multichannel':
            if self.embed == 'luma':
                pass
                # Create a new channel with the luma values (R,G,B) -> (R,G,B,Y')
                lumas = rgb_to_ycbcr(secret)
                # Only keep the luma channel
                lumas = lumas[:,0,:,:].unsqueeze(1).cuda()
                secret = torch.cat((secret,lumas),1)
            else:
                # Create a new channel with 0 (R,G,B) -> (R,G,B,0)
                zero = torch.zeros(1, 1, 256, 256).type(torch.float32).cuda()
                secret = torch.cat((secret,zero),1)
        
        # Encode the image using PHN
        hidden_signal = self.PHN(secret)
        if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
            hidden_signal_phase = self.PHN_phase(secret)
        
        if self.embed == 'blocks' or self.embed == 'blocks2' or self.embed == 'blocks3':
            if self.transform != 'fourier':
                raise Exception('\'blocks\' embedding is only implemented for STFT')
            # Replicate the hidden image as many times as required (only two for STFT)
            if self.embed == 'blocks':
                # Simply duplicate and concat vertically
                hidden_signal = torch.cat((hidden_signal, hidden_signal), 2)
            else:
                # Else also scale with a learnable weight
                hidden_signal = torch.cat((hidden_signal*self.encblocks[0], hidden_signal*self.encblocks[1]), 2)
        elif self.embed == 'multichannel':
            # Split the 8 channels and replicate. 1x8x256x256 -> 1x1x1024x512
            split1 = torch.split(hidden_signal, 2, dim=1)
            cat1 = torch.cat(split1, dim=2)
            split2 = torch.split(cat1, 1, dim=1)
            hidden_signal = torch.cat(split2, dim=3)
        
        # Permute the encoded image if required
        if self.permutation:
            # Generate permutation index, which will be reused for the inverse
            perm_idx = torch.randperm(hidden_signal.nelement())
            # Permute the hidden signal
            hidden_signal = hidden_signal.view(-1)[perm_idx].view(hidden_signal.size())
            # Also permute the phase if necessary
            if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
                hidden_signal_phase = hidden_signal_phase.view(-1)[perm_idx].view(hidden_signal_phase.size())

        # Residual connection
        # Also keep a copy of the unpermuted containers to compute the loss
        container = cover + hidden_signal
        orig_container = container
        if self.transform == 'fourier' and self.ft_container == 'magphase':
            if self.mp_encoder == 'double':
                container_phase = cover_phase + hidden_signal_phase
            elif self.mp_encoder == 'single':
                container_phase = cover_phase + hidden_signal
            orig_container_phase = container_phase


        # Unpermute the encoded image if it was permuted
        if self.permutation:
            # Compute the inverse permutation
            inv_perm_idx = torch.argsort(perm_idx)
            # Permute the hidden signal with the inverse
            container = container.view(-1)[inv_perm_idx].view(container.size())
            # Also permute the phase if necessary
            if self.transform == 'fourier' and self.ft_container == 'magphase' and self.mp_encoder == 'double':
                container_phase = container_phase.view(-1)[inv_perm_idx].view(container_phase.size())

        # Reveal image
        if self.transform == 'fourier' and self.ft_container == 'magphase':
            if self.mp_decoder == 'unet':
                revealed = self.RN(container, container_phase)
            else:
                revealed = self.RN(container)
                revealed_phase = self.RN_phase(container_phase)
                if self.mp_join == 'mean':
                    revealed = revealed.add(revealed_phase)*0.5
                elif self.mp_join == '2D':
                    join = torch.cat((revealed,revealed_phase),1)
                    revealed = self.mag_phase_join(join)
                elif self.mp_join == '3D':
                    revealed = revealed.unsqueeze(0)
                    revealed_phase = revealed_phase.unsqueeze(0)
                    join = torch.cat((revealed,revealed_phase),1)
                    revealed = self.mag_phase_join(join).squeeze(1)
            return (orig_container, orig_container_phase), revealed
        else:
            # If only using one container, reveal and return
            revealed = self.RN(container)
            return orig_container, revealed
