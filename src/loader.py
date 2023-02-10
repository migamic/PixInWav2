'''
loader.py

* Image and audio preprocessing
* Data set class
* Data loader
'''


import os
import re
import torch
import random
import pathlib
import torchaudio
import numpy as np
import glob as glob
from PIL import Image
from torch.utils.data import DataLoader

from pystct import sdct_torch, isdct_torch
from torch_stft import STFT
import matplotlib.pyplot as plt

MY_FOLDER = os.environ.get('USER_PATH')
DATA_FOLDER = os.environ.get('DATA_PATH')
AUDIO_FOLDER = f"{DATA_FOLDER}/FSDnoisy/FSDnoisy18k.audio_"
IMAGE_FOLDER = f'{DATA_FOLDER}/imagenet'

class ImageProcessor():
    """
    Function to preprocess the images from the custom 
    dataset. It includes a series of transformations:

    - At __init__ we convert the image to the desired [colorspace].
    - Crop function crops the image to the desired [proportion].
    - Scale scales the images to desired size [n]x[n].
    - Normalize performs the normalization of the channels.
    """
    def __init__(self, image_path, colorspace='RGB'):
        self.image = Image.open(image_path).convert(colorspace)

    def crop(self, proportion = 2 ** 6):
        nx, ny = self.image.size
        n = min(nx, ny)
        left = top = n / proportion
        right = bottom = (proportion - 1) * n / proportion
        self.image = self.image.crop((left, top, right, bottom))

    def scale(self, n = 256):
        self.image = self.image.resize((n, n), Image.ANTIALIAS)

    def normalize(self):
        self.image = np.array(self.image).astype('float') / 255.0

    def forward(self):
        self.crop()
        self.scale()
        self.normalize()
        return self.image

class AudioProcessor():
    """
    Function to preprocess the audios from the custom 
    dataset. We set the [_limit] in terms of samples,
    the [_frame_length] and [_frame_step] of the [transform]
    transform. 

    If transform is [cosine] it returns just the STDCT matrix.
    Else, if transform is [fourier] returns the STFT magnitude
    and phase.
    """
    def __init__(self, transform, stft_small=True, random_init=True):
        # Corresponds to 1.5 seconds approximately
        self._limit = 67522 # 2 ** 16 + 2 ** 11 - 2 ** 6 + 2
        if transform == 'cosine':
            self._frame_length = 2 ** 12
            self._frame_step = 2 ** 6 - 2
        else:
            if stft_small:
                self._frame_length = 2 ** 11 - 1
                self._frame_step = 2 ** 7 + 4
            else:
                self._frame_length = 2 ** 12 - 1
                self._frame_step = 2 ** 6 + 2

        self.random_init = random_init

        self._transform = transform
        if self._transform == 'fourier':
            self.stft = STFT(
                filter_length=self._frame_length, 
                hop_length=self._frame_step, 
                win_length=self._frame_length,
                window='hann'
            )   

    def forward(self, audio_path):
        self.sound, self.sr = torchaudio.load(audio_path)
        
        # Get the samples dimension
        sound = self.sound[0]

        # Create a temporary array
        tmp = torch.zeros([self._limit, ])

        # Check if the audio is shorter than the limit
        if sound.numel() < self._limit:
            # Zero-pad at the end, or randomly at both start and end
            if self.random_init:
                i = random.randint(0, self._limit - len(sound))
                tmp[i:i+sound.numel()] = sound[:]
            else:
                tmp[:sound.numel()] = sound[:]
        else:
            # Use only part of the audio. Either start at beginning or random
            if self.random_init:
                i = random.randint(0, len(sound) - self._limit)
            else:
                i = 0
            tmp[:] = sound[i:i + self._limit]

        if self._transform == 'cosine':
            return sdct_torch(
                tmp.type(torch.float32),
                frame_length = self._frame_length,
                frame_step = self._frame_step
            )
        elif self._transform == 'fourier':
            magnitude, phase = self.stft.transform(tmp.unsqueeze(0).type(torch.float32))
            return magnitude, phase
        else: raise Exception(f'Transform not implemented')


class StegoDataset(torch.utils.data.Dataset):
    """
    Custom datasets pairing images with spectrograms.
    - [image_root] defines the path to read the images from.
    - [audio_root] defines the path to read the audio clips from.
    - [folder] can be either [train] or [test].
    - [mappings] is the dictionary containing a descriptive name for 
    the images from ImageNet. It is used to index the different
    subdirectories.
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    - [image_extension] defines the extension of the image files. 
    By default it is set to JPEG.
    - [audio_extension] defines the extension of the audio files. 
    By default it is set to WAV.
    """

    def __init__(
        self,
        image_root: str,
        audio_root: str,
        folder: str,
        mappings: dict,
        rgb: bool = True,
        transform: str = 'cosine',
        stft_small: bool = True,
        image_extension: str = "JPEG",
        audio_extension: str = "wav"
    ):

        # self._image_data_path = pathlib.Path(image_root) / folder
        self._image_data_path = pathlib.Path(image_root) / 'train'
        self._audio_data_path = pathlib.Path(f'{audio_root}{folder}')
        self._MAX_LIMIT = 1000 if folder == 'train' else 900
        self._TOTAL = 1000
        self._MAX_AUDIO_LIMIT = 1758 if folder == 'train' else 946
        self._colorspace = 'RGB' if rgb else 'L'
        self._transform = transform
        self._stft_small = stft_small

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_extension = image_extension
        self.audio_extension = audio_extension
        self._index = 0
        self._indices = []
        self._audios = []

        #IMAGE PATH RETRIEVING
        test_i, test_j = 0, 0
        #keys are n90923u23
        if (folder == 'train'):
            for key in mappings.keys():
                for j, img in enumerate(glob.glob(f'{self._image_data_path}/{key}/*.{self.image_extension}')):
    
                    if j >= 10: break
                    self._indices.append((key, re.search(r'(?<=_)\d+', img).group()))
                    self._index += 1

                    if self._index == self._MAX_LIMIT: break
                if self._index == self._MAX_LIMIT: break

        elif (folder == "test"):
            for key in mappings.keys():
                for img in glob.glob(f'{self._image_data_path}/{key}/*.{self.image_extension}'):
    
                    if test_i > self._TOTAL:
                        if test_j >= 10: 
                            test_j = 0
                            break
                        self._indices.append((key, re.search(r'(?<=_)\d+', img).group()))
                        self._index += 1
                        test_j += 1
    
                    test_i += 1

                    if self._index == self._MAX_LIMIT: break
                if self._index == self._MAX_LIMIT: break

        #AUDIO PATH RETRIEVING (here the paths for test and train are different)
        self._index_aud = 0

        for audio_path in glob.glob(f'{self._audio_data_path}/*.{self.audio_extension}'):

            self._audios.append(audio_path)
            self._index_aud += 1

            if (self._index_aud == self._MAX_AUDIO_LIMIT): break


        self._AUDIO_PROCESSOR = AudioProcessor(transform=self._transform, stft_small=self._stft_small)

        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        key = self._indices[index][0]
        indexer = self._indices[index][1]
        rand_indexer = random.randint(0, self._MAX_AUDIO_LIMIT - 1)

        img_path = glob.glob(f'{self._image_data_path}/{key}/{key}_{indexer}.{self.image_extension}')[0]
        audio_path = self._audios[rand_indexer]

        img = np.asarray(ImageProcessor(image_path=img_path, colorspace=self._colorspace).forward()).astype('float64')
        
        if self._transform == 'cosine':
            sound_stct = self._AUDIO_PROCESSOR.forward(audio_path)
            return (img, sound_stct)
        elif self._transform == 'fourier':
            magnitude_stft, phase_stft = self._AUDIO_PROCESSOR.forward(audio_path)
            return (img, magnitude_stft, phase_stft)
        else: raise Exception(f'Transform not implemented')

def loader(set='train', rgb=True, transform='cosine', stft_small=True, batch_size=1, shuffle=False):
    """
    Prepares the custom dataloader.
    - [set] defines the set type. Can be either [train] or [test].
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    """
    print('Preparing dataset...')
    mappings = {}
    with open(f'{IMAGE_FOLDER}/mappings.txt') as f:
        for line in f:
            words = line.split()
            mappings[words[0]] = words[1]

    dataset = StegoDataset(
        image_root=f'{IMAGE_FOLDER}/ILSVRC/Data/CLS-LOC',
        audio_root=AUDIO_FOLDER,
        folder=set,
        mappings=mappings,
        rgb=rgb,
        transform=transform,
        stft_small=stft_small
    )

    print('Dataset prepared.')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=shuffle
    )

    print('Data loaded ++')
    return dataloader
