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
