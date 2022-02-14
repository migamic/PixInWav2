'''
train.py

* Training and validation functions
* wandb logging
'''

import time
import gc
import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch_stft import STFT
from pystct import sdct_torch, isdct_torch
from losses import ssim, SNR, PSNR, StegoLoss
from visualization import viz2paper


def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('OUT_PATH'),'models/checkpoint.pt')):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         print(f'SAVING TO: {filename}')
         torch.save(state, filename)  # save checkpoint
     else:
         print ("=> Loss did not improve")


def train(model, tr_loader, vd_loader, beta, lam, lr, epochs=5, prev_epoch = None, prev_i = None, summary=None, slide=50, experiment=0, transform='cosine', ft_container='mag'):

    # Initialize wandb logs
    wandb.init(project='PixInWavRGB')
    if summary is not None:
        wandb.run.name = summary
        wandb.run.save()
    wandb.watch(model)

    # Prepare to device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Parallelize on GPU
    if torch.cuda.device_count() > 1:
          print(f"Let's use {torch.cuda.device_count()} GPUs!")
          model = nn.DataParallel(model)

    model.to(device)

    # Set to training mode
    model.train()

    # Number of parameters in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {num_params}')

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ini = time.time()
    best_loss = np.inf

    # Initialize L1 loss constructor
    l1wavLoss = nn.L1Loss()

    # Initialize STFT transform constructor
    if transform == 'fourier':
        stft = STFT(
            filter_length=2 ** 11 - 1,
            hop_length=132,
            win_length=2 ** 11 - 1,
            window='hann'
        ).to(device)
        stft.num_samples = 67522

    # Start training ...
    for epoch in range(epochs):

        if prev_epoch != None and epoch < prev_epoch - 1: continue # Checkpoint pass

        # Initialize training metrics storage
        train_loss, train_loss_cover, train_loss_secret, train_loss_spectrum, snr, psnr, ssim_secret, train_l1_loss = [], [], [], [], [], [], [], []
        vd_loss, vd_loss_cover, vd_loss_secret, vd_snr, vd_psnr, vd_ssim, vd_l1 = [], [], [], [], [], [], []
        
        for i, data in enumerate(tr_loader):

            if prev_i != None and i < prev_i - 1: continue # Checkpoint pass

            # Load data from the loader
            secrets, covers = data[0].to(device), data[1].to(device)
            if transform == 'fourier': phase = data[2].to(device)
            secrets = secrets.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)
            covers = covers.unsqueeze(1) if transform == 'cosine' else covers

            optimizer.zero_grad()

            # Forward through the model
            if transform == 'fourier':
                if ft_container == 'mag':
                    containers, revealed = model(secrets, covers)
                elif ft_container == 'phase':
                    containers, revealed = model(secrets, phase)
                elif ft_container == 'magphase':
                    # If using mag+phase, get both mag and phase containers
                    (containers, containers_phase), revealed = model(secrets, covers, phase)
            else:
                # STDCT transform
                containers, revealed = model(secrets, covers)


            # Compute the loss
            if transform == 'cosine':
                original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130, window=torch.hamming_window)
                container_wav = isdct_torch(containers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130, window=torch.hamming_window)
                container_2x = sdct_torch(container_wav, frame_length=4096, frame_step=130, window=torch.hamming_window).unsqueeze(0).unsqueeze(0)
                loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
            elif transform == 'fourier': 
                if ft_container == 'mag':
                    original_wav = stft.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = stft.inverse(containers.squeeze(1), phase.squeeze(1))
                    container_2x = stft.transform(container_wav)[0].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
                elif ft_container == 'phase':
                    original_wav = stft.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = stft.inverse(covers.squeeze(1), containers.squeeze(1))
                    container_2x = stft.transform(container_wav)[1].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, phase, containers, container_2x, revealed, beta)
                elif ft_container == 'magphase':
                    # Using magnitude+phase. Compute both MSEs
                    original_wav = stft.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = stft.inverse(covers.squeeze(1), containers.squeeze(1))
                    container_phase_wav = stft.inverse(covers.squeeze(1), containers_phase.squeeze(1))
                    container_2x_phase = stft.transform(container_phase_wav)[1].unsqueeze(0)
                    container_2x_mag = stft.transform(container_wav)[0].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, phase, containers_phase, container_2x_phase, revealed, beta, covers, containers, container_2x_mag)

            # Compute L1 waveform loss. Add it only if specified
            l1_loss = l1wavLoss(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))
            objective_loss = loss 
            objective_loss += lam * l1_loss
            with torch.autograd.set_detect_anomaly(True):
                objective_loss.backward()
            optimizer.step()

            # Compute audio and image metrics
            if (transform != 'fourier') or (ft_container != 'magphase'):
                containers_phase = None # Otherwise it's the phase container
            snr_audio = SNR(
                covers, 
                containers, 
                None if transform == 'cosine' else phase,
                containers_phase,
                transform=transform,
                transform_constructor= None if transform == 'cosine' else stft,
                ft_container=ft_container,
            )
            psnr_image = PSNR(secrets, revealed)
            ssim_image = ssim(secrets, revealed)

            # Append and average the new losses
            train_loss.append(loss.detach().item())
            train_loss_cover.append(loss_cover.detach().item())
            train_loss_secret.append(loss_secret.detach().item())
            train_loss_spectrum.append(loss_spectrum.detach().item())
            snr.append(snr_audio)
            psnr.append(psnr_image.detach().item())
            ssim_secret.append(ssim_image.detach().item())
            train_l1_loss.append(l1_loss.detach().item())

            avg_train_loss = np.mean(train_loss[-slide:])
            avg_train_loss_cover = np.mean(train_loss_cover[-slide:])
            avg_train_loss_secret = np.mean(train_loss_secret[-slide:])
            avg_train_loss_spectrum = np.mean(train_loss_spectrum[-slide:])
            avg_snr = np.mean(snr[-slide:])
            avg_ssim = np.mean(ssim_secret[-slide:])
            avg_psnr = np.mean(psnr[-slide:])
            avg_l1_loss = np.mean(train_l1_loss[-slide:])
            avg_l1_loss = np.mean(train_l1_loss[-slide:])

            print(
                f'(#{i})[{np.round(time.time()-ini,2)}s]\
                Train Loss {round(loss.detach().item(),4)},\
                MSE audio {round(loss_cover.detach().item(),4)},\
                MSE image {round(loss_secret.detach().item(),4)},\
                MSE spectrum {round(loss_spectrum.detach().item(),4)},\
                SNR {round(snr_audio,4)},\
                PSNR {round(psnr_image.detach().item(),4)},\
                SSIM {round(ssim_image.detach().item(),4)},\
                L1 {round(l1_loss.detach().item(),4)}' 
                )

            # Log train average loss to wandb
            wandb.log({
                'tr_i_loss': avg_train_loss,
                'tr_i_cover_loss': avg_train_loss_cover,
                'tr_i_secret_loss': avg_train_loss_secret,
                'tr_i_spectrum_loss': avg_train_loss_spectrum,
                'SNR': avg_snr,
                'PSNR': avg_psnr,
                'SSIM': avg_ssim,
                'L1': avg_l1_loss,
            })

            # Every 50 iterations, do a validation step
            if (i % 50 == 0) and (i != 0):
                avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_l1 = validate(model, vd_loader, beta, transform=transform, transform_constructor=stft if transform=='fourier' else None, ft_container=ft_container, l1_criterion=l1wavLoss, tr_i=i, epoch=epoch)
                
                vd_loss.append(avg_valid_loss) 
                vd_loss_cover.append(avg_valid_loss_cover) 
                vd_loss_secret.append(avg_valid_loss_secret) 
                vd_snr.append(avg_valid_snr) 
                vd_psnr.append(avg_valid_psnr)
                vd_ssim.append(avg_valid_ssim) 
                vd_l1.append(avg_valid_l1)

                is_best = bool(avg_valid_loss < best_loss)
                # Save checkpoint if is a new best
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'beta': beta,
                    'lr': lr,
                    'i': i + 1,
                    'tr_loss': train_loss,
                    'tr_cover_loss': train_loss_cover,
                    'tr_loss_secret': train_loss_secret,
                    'tr_snr': snr,
                    'tr_psnr': psnr,
                    'tr_ssim': ssim_secret,
                    'tr_l1': train_l1_loss,
                    'vd_loss': vd_loss,
                    'vd_cover_loss': vd_loss_cover,
                    'vd_loss_secret': vd_loss_secret,
                    'vd_snr': vd_snr,
                    'vd_psnr': vd_psnr,
                    'vd_ssim': vd_ssim,
                    'vd_l1': vd_l1,
                }, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'), f'models/checkpoint_run_{experiment}.pt'))
        
        print(
            f'Epoch [{epoch + 1}/{epochs}], \
            Average_loss: {round(avg_train_loss,4)}, \
            Average_loss_cover: {round(avg_train_loss_cover,4)}, \
            Average_loss_secret: {round(avg_train_loss_secret,4)}, \
            Average_loss_spectrum: {round(avg_train_loss_spectrum,4)}, \
            Average SNR: {round(avg_snr,4)}, \
            Average PSNR: {round(avg_psnr,4)},\
            Average SSIM: {round(avg_ssim,4)}, \
            Average L1: {round(avg_l1_loss,4)}'
        )

        # Log train average loss to wandb
        wandb.log({
            'tr_loss': avg_train_loss,
            'tr_cover_loss': avg_train_loss_cover,
            'tr_secret_loss': avg_train_loss_secret,
        })
        
        is_best = bool(avg_train_loss < best_loss)
        best_loss = min(avg_train_loss, best_loss)

        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'beta': beta,
            'lr': lr,
            'i': i + 1,
        }, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'), f'models/checkpoint_run_{experiment}.pt'))

    print(f"Training took {time.time() - ini} seconds")
    torch.save(model.state_dict(), os.path.join(os.environ.get('OUT_PATH'), f'models/final_run_{experiment}.pt'))
    return model, avg_train_loss











def validate(model, vd_loader, beta, transform='cosine', transform_constructor=None, ft_container='mag', l1_criterion=None, epoch=None, tr_i=None):

    # Set device
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Parallelize on GPU
    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)

    model.to(device)

    # Set to evaluation mode
    model.eval()
    loss = 0

    valid_loss, valid_loss_cover, valid_loss_secret, valid_loss_spectrum, valid_snr, valid_psnr, valid_ssim, valid_l1 = [], [], [], [], [], [], [], []
    vd_datalen = len(vd_loader)

    # Start validating ...
    iniv = time.time()
    with torch.no_grad():
        print('Validating current model...')
        for i, data in enumerate(vd_loader):

            # Load data from the loader
            secrets, covers = data[0].to(device), data[1].to(device)
            secrets = secrets.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)
            if transform == 'fourier': phase = data[2].to(device)
            covers = covers.unsqueeze(1) if transform == 'cosine' else covers

            # Forward through the model
            if transform == 'fourier':
                if ft_container == 'mag':
                    containers, revealed = model(secrets, covers)
                elif ft_container == 'phase':
                    containers, revealed = model(secrets, phase)
                elif ft_container == 'magphase':
                    # If using mag+phase, get both mag and phase containers
                    (containers, containers_phase), revealed = model(secrets, covers, phase)
            else:
                # STDCT transform
                containers, revealed = model(secrets, covers)

            # Visualize results
            if i == 0:
                if transform == 'cosine':
                    fig = viz2paper(secrets.cpu(), revealed.cpu(), covers.cpu(), containers.cpu(), None, None, transform, ft_container)
                elif transform == 'fourier':
                    if ft_container == 'mag':
                        fig = viz2paper(secrets.cpu(), revealed.cpu(), covers.cpu(), containers.cpu(), None, None, transform, ft_container)
                    elif ft_container == 'phase':
                        fig = viz2paper(secrets.cpu(), revealed.cpu(), phase.cpu(), containers.cpu(), None, None, transform, ft_container)
                    elif ft_container == 'magphase':
                        fig = viz2paper(secrets.cpu(), revealed.cpu(), covers.cpu(), containers.cpu(), phase.cpu(), containers_phase.cpu(), transform, ft_container)
                else:
                    raise Exception('Transform not implemented')


                wandb.log({f"Revelation at epoch {epoch}, vd iteration {tr_i}": fig})

            # Compute the loss
            if transform == 'cosine':
                original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130, window=torch.hamming_window)
                container_wav = isdct_torch(containers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130, window=torch.hamming_window)
                container_2x = sdct_torch(container_wav, frame_length=4096, frame_step=130, window=torch.hamming_window).unsqueeze(0).unsqueeze(0)
                loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
            elif transform == 'fourier': 
                if ft_container == 'mag':
                    original_wav = transform_constructor.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = transform_constructor.inverse(containers.squeeze(1), phase.squeeze(1))
                    container_2x = transform_constructor.transform(container_wav)[0].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
                elif ft_container == 'phase':
                    original_wav = transform_constructor.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = transform_constructor.inverse(covers.squeeze(1), containers.squeeze(1))
                    container_2x = transform_constructor.transform(container_wav)[1].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, phase, containers, container_2x, revealed, beta)
                elif ft_container == 'magphase':
                    # Using magnitude+phase. Compute both MSEs
                    original_wav = transform_constructor.inverse(covers.squeeze(1), phase.squeeze(1))
                    container_wav = transform_constructor.inverse(covers.squeeze(1), containers.squeeze(1))
                    container_phase_wav = transform_constructor.inverse(covers.squeeze(1), containers_phase.squeeze(1))
                    container_2x_phase = transform_constructor.transform(container_phase_wav)[1].unsqueeze(0)
                    container_2x_mag = transform_constructor.transform(container_wav)[0].unsqueeze(0)
                    loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, phase, containers_phase, container_2x_phase, revealed, beta, covers, containers, container_2x_mag)


            # Compute audio and image metrics
            if (transform != 'fourier') or (ft_container != 'magphase'):
                containers_phase = None # Otherwise it's the phase container
            vd_snr_audio = SNR(
                covers, 
                containers, 
                None if transform == 'cosine' else phase,
                containers_phase,
                transform=transform,
                transform_constructor= None if transform == 'cosine' else transform_constructor,
                ft_container=ft_container,
            )
            vd_psnr_image = PSNR(secrets, revealed)
            ssim_image = ssim(secrets, revealed)

            if l1_criterion is not None:
                if transform == 'cosine':
                    original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130, window=torch.hamming_window)
                elif transform == 'fourier':
                    original_wav = transform_constructor.inverse(covers.squeeze(1), phase.squeeze(1))
                l1_loss = l1_criterion(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))

            valid_loss.append(loss.detach().item())
            valid_loss_cover.append(loss_cover.detach().item())
            valid_loss_secret.append(loss_secret.detach().item())
            valid_loss_spectrum.append(loss_spectrum.detach().item())
            valid_snr.append(vd_snr_audio)
            valid_psnr.append(vd_psnr_image.detach().item())
            valid_ssim.append(ssim_image.detach().item())
            valid_l1.append(l1_loss.detach().item())

            print(
                f'(#{i})[{np.round(time.time()-iniv,2)}s]\
                Valid Loss {loss.detach().item()},\
                cover_error {loss_cover.detach().item()},\
                secret_error {loss_secret.detach().item()},\
                spectrum_error {loss_spectrum.detach().item()},\
                SNR {vd_snr_audio},\
                PSNR {vd_psnr_image.detach().item()},\
                SSIM {ssim_image.detach().item()},\
                L1 {l1_loss.detach().item()}'
            )

            if i >= 500: break
            # if i >= vd_datalen: break

        avg_valid_loss = np.mean(valid_loss)
        avg_valid_loss_cover = np.mean(valid_loss_cover)
        avg_valid_loss_secret = np.mean(valid_loss_secret)
        avg_valid_loss_spectrum = np.mean(valid_loss_spectrum)
        avg_valid_snr = np.mean(valid_snr)
        avg_valid_psnr = np.mean(valid_psnr)
        avg_valid_ssim = np.mean(valid_ssim)
        avg_valid_l1 = np.mean(valid_l1)

        wandb.log({
            'vd_loss': avg_valid_loss,
            'vd_cover_loss': avg_valid_loss_cover,
            'vd_secret_loss': avg_valid_loss_secret,
            'vd_spectrum_loss': avg_valid_loss_spectrum,
            'vd_SNR': avg_valid_snr,
            'vd_PSNR': avg_valid_psnr,
            'vd_SSIM': avg_valid_ssim,
            'vd_L1': avg_valid_l1
        })
        print(f"Validation took {time.time() - iniv} seconds")

    del valid_loss
    del valid_loss_cover
    del valid_loss_secret
    del valid_loss_spectrum
    del valid_snr
    del valid_psnr
    del valid_ssim
    del valid_l1
    gc.collect()

    return avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_l1
