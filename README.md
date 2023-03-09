# PixInWav 2

Reimplementation of [PixInWav](https://github.com/margaritageleta/PixInWav). Refer to the original repo for details on the paper and how to run the code.

It has been rewritten from scratch, but expect much overlap with the original code.


### New features
* Ability to encode in the STFT magnitude and phase simultaneously. Plenty of different options to choose from, controlled through flags.
* New architectures and possibilities for embeddings, controlled through flags.
* Possibility of permuting the preprocessed image before adding it into the spectrogram and unpermutting before revealing.
* STFT can be made larger to obtain more container capacity and better results at the cost of training time and memory.
* Possiblity of using batch sizes > 1.
* Number of training epochs is now a flag.


### Dropped features
* Old architectures that were shown to be worse than the default 'resindep'.
* Possibility to add noise on the transmitted signal.
* Support for grayscale images. Everything is RGB now.


### Other changes / bug fixes
* The loss function now uses L1 loss instead of soft DTW.
* Tidier output during training. It is now easier to follow the progress from the log files.
* Validation plots are different when using STDCT or STFT (with magnitude, phase or both).
* Flags for controlling STFT magnitude and phase have changed, to allow for the possibility of having magnitude and phase together.
* Validation steps are less frequent and shorter (can be controlled through flags). Training is ~10x faster by default.
* Fix validation data loading, that was using the same images from the training set.
* Fix audio preprocessing: remove padding bias and noise.
* Set the model in training mode during training (used to be in 'eval'). Remove batch norm since it made the performance worse.


## Repository outline
All the code files are in the `src` directory. These have been renamed from the original code and redundant code files have been removed.
* `main.py`: script to run the overall training process.
* `umodel.py`: the complete PixInWav model.
* `train.py`: responsible for the training and validation of the given model.
* `loader.py`: loader script to create the data set from the images and audios.
* `losses.py`: losses and metrics for training and validation.
* `pystct.py`: computes the STDCT.
* `visualization.py`: functions for plotting the validation results during training.

In the `scripts` directory there is `srun_train.sh`, an example of a shell script to run the code with all the new flags.


## Usage
Refer to the [original repo](https://github.com/margaritageleta/PixInWav) for usage instruction. This section only details the flags accepted by `main.py`, since many have changed.

### Generic flags
* `experiment` sets the experiment number, which will be used when saving/loading checkpoints.
* `summary` gives a name to the current experiment that will be shown in Weights and Biases.
* `output` specifies a text file where the stardard output should be printed.
* `from_checkpoint` allows loading an existing model from a checkpoint instead of starting the training process anew.

### Training hyperparameters
* `lr` sets the learning rate.
* Use `val_itvl` and `val_size` to control after how many training iterations to perform validation and how many validation iterations should there be.
* Use `num_epochs` and `batch_size` to easily tweak the number of epochs and the batch size.

### Loss hyperparameters
* `beta` and `lambda` control the tradeoff between audio and image quality. If using both STFT magnitude and phase, `theta` weights the importance of each of the two containers.

### Audio transform
* `transform` is used to choose between STDCT or STFT. Note that STDCT is mostly obsolete and some of the most recent features do not support it.
* `stft_small` allows using the default 'small' STFT container or a larger one, twice the size in each dimension.
* `ft_container` selects which STFT container to use: magnitude, phase or both.

### Magnitude+phase
* `mp_encoder` and `mp_decoder` allow choosing between multiple architectures that use the STFT magnitude and phase at the same time.
* `mp_join` specifies the joining operation to use when revealing separately the magnitude and the phase.

### Other flags
* `permutation` specifies whether or not to permute the signals after preprocessing and before revealing.
* `embed` is used to choose between multiple embedding methods. Non-default options have only been tested using the STFT magnitude as a container.

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as *academic research* must contact `jaume.ros.alonso@estudiantat.upc.edu` AND `geleta@berkeley.edu` for a separate license. 
