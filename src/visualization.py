import numpy as np
import matplotlib.pyplot as plt

def viz2paper(s, r, cv, ct, cv_phase, ct_phase, transform, ft_container, log=True):

    # 'ct_phase' is the STFT phase when using mag+phase
    # Otherwise 'ct' is the only container
    assert not ((transform == 'fourier' and ft_container == 'magphase') and ct_phase is None)
    assert not ((transform == 'fourier' and ft_container != 'magphase') and ct_phase is not None)

    # Define transform and container names to use in the plot
    if transform == 'cosine':
        tr_name = 'STDCT'
        ct_name = 'spectrogram'
    elif transform == 'fourier':
        tr_name = 'STFT'
        if ft_container == 'mag':
            ct_name = 'magnitude'
        elif ft_container == 'phase':
            ct_name = 'phase'
        elif ft_container == 'magphase':
            ct_name = 'magnitude'
            ct_name2 = 'phase'
    else:
        raise Exception('Transform not implemented')

    s = s.permute(0,2,3,1).detach().numpy().squeeze(0)
    r = r.permute(0,2,3,1).detach().numpy().squeeze(0)
    cv = cv.detach().numpy().squeeze(0).squeeze(0)
    ct = ct.detach().numpy().squeeze(0).squeeze(0)
    if ct_phase is not None:
        cv_phase = cv_phase.detach().numpy().squeeze(0).squeeze(0)
        ct_phase = ct_phase.detach().numpy().squeeze(0).squeeze(0)
    
    s = (s * 255.0).astype(np.uint8)
    r = np.clip(r * 255.0, 0, 255).astype(np.uint8)

    if transform == 'fourier' and ft_container == 'magphase':
        # Different plot when using two containers
        fig, ax = plt.subplots(2, 3, figsize=(18, 10))
        ax[0,0].imshow(s)
        ax[1,0].imshow(r)
        ax[0,0].set_title('Secret image')
        ax[1,0].set_title('Revealed image')
        ax[0,0].axis('off')
        ax[1,0].axis('off')

        if log:
            img1 = ax[0,1].imshow(np.log(np.abs(cv)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,1].set_title(f'Cover {tr_name} log {ct_name}')
            img2 = ax[1,1].imshow(np.log(np.abs(ct)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,1].set_title(f'Container {tr_name} log {ct_name}')
            img3 = ax[0,2].imshow(np.log(np.abs(cv_phase)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,2].set_title(f'Cover {tr_name} log {ct_name2}')
            img4 = ax[1,2].imshow(np.log(np.abs(ct_phase)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,2].set_title(f'Container {tr_name} log {ct_name2}')
        else:
            img1 = ax[0,1].imshow(np.abs(cv)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,1].set_title(f'Cover {tr_name} {ct_name}')
            img2 = ax[1,1].imshow(np.abs(ct)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,1].set_title(f'Container {tr_name} {ct_name}')
            img3 = ax[0,2].imshow(np.abs(cv_phase)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,2].set_title(f'Cover {tr_name} {ct_name2}')
            img4 = ax[1,2].imshow(np.abs(ct_phase)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,2].set_title(f'Container {tr_name} {ct_name2}')

        ax[0,1].set_xlabel('Time [n]')
        ax[0,1].set_ylabel('Frequency')
        ax[1,1].set_xlabel('Time [n]')
        ax[1,1].set_ylabel('Frequency')
        ax[1,2].set_xlabel('Time [n]')
        ax[1,2].set_ylabel('Frequency')

        plt.colorbar(img1, ax=ax[0,1])
        plt.colorbar(img2, ax=ax[1,1])
        plt.colorbar(img3, ax=ax[0,2])
        plt.colorbar(img4, ax=ax[1,2])
        plt.close('all')
    else:
        # Else using only one container
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0,0].imshow(s)
        ax[1,0].imshow(r)
        ax[0,0].set_title('Secret image')
        ax[1,0].set_title('Revealed image')
        ax[0,0].axis('off')
        ax[1,0].axis('off')

        if log:
            img1 = ax[0,1].imshow(np.log(np.abs(cv)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,1].set_title(f'Cover {tr_name} log {ct_name}')
            img2 = ax[1,1].imshow(np.log(np.abs(ct)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,1].set_title(f'Container {tr_name} log {ct_name}')
        else:
            img1 = ax[0,1].imshow(np.abs(cv) [:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[0,1].set_title(f'Cover {tr_name} {ct_name}')
            img2 = ax[1,1].imshow(np.abs(ct)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
            ax[1,1].set_title(f'Container {tr_name} {ct_name}')

        ax[0,1].set_xlabel('Time [n]')
        ax[0,1].set_ylabel('Frequency')
        ax[1,1].set_xlabel('Time [n]')
        ax[1,1].set_ylabel('Frequency')

        plt.colorbar(img1, ax=ax[0,1])
        plt.colorbar(img2, ax=ax[1,1])
        plt.close('all')

    return fig

