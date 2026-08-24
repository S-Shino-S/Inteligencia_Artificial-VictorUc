"""A tiny, from-scratch GAN in pure Python to illustrate how GANs work.

Everything is 1D and written without NumPy so each step is readable:
- gan.mlp    : a hand-written MLP with manual backprop and an Adam optimizer
- gan.data   : the real distribution to imitate and the noise prior
- gan.gan    : the generator/discriminator and the minimax training loop
- gan.gan2d/data2d : the same game with 2D data (program 05)
- gan.metrics/format : statistics, ASCII histograms and scatter plots
"""
