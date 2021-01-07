# FastApproximateInference.jl

Experiments for [Preconditioned training of normalizing flows for variational inference in inverse problems](https://openreview.net/forum?id=P9m1sMaNQ8T)—accepted at the [3rd Symposium on Advances in Approximate Bayesian Inference](http://approximateinference.org/).

To start running the examples, clone the repository:

```bash
$ git clone https://github.com/alisiahkoohi/FastApproximateInference.jl
```

Here, we heavily rely on [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl), a recently-developed, memory-efficient framework for training invertible networks in Julia.

## Installation

This repository is based on [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl). Before running examples, install `DrWatson,jl` by:

```julia
pkg> add DrWatson
```

The only other manual installation is to make sure you have `matplotlib` and `seaborn` installed in your Python environment since we depend on `PyPlot.jl` and `Seaborn` for creating figures.

The necessary dependencies will be installed upon running your first experiment. If you happen to have a CUDA-enabled GPU, the code will run on it.

### 2D toy Rosenbrock distribution example

In this examples, we train a normalizing flow on a joint distribution of samples from 2D Rosenbrock distribution and data, where data is obtained by adding Gaussian noise to individual samples.

Run the script below for training the normalizing flow:

```bash
$ julia scripts/2d-rosenbrock/train_supervised_hint.jl
```

To perform joint or conditional (posterior) samples via the pretrained normalizing flow (obtained by running the script above), run:

```bash
$ julia scripts/2d-rosenbrock/test_supervised_hint.jl
```

Next, we generate data by applying a close-to-identity, square matrix to the a sample from Rosenbrock distribution and adding Gaussian noise.

We apply the preconditioning approach to train a new normalizing flow capable of directly sampling the posterior of the new problem. Run the commands below, in order, to experiment with the preconditioning approach, as well as to train the normalizing flow with no preconditioning.

```bash
$ julia scripts/2d-rosenbrock/train_warm-start_unsupervised_hint.jl
```

```bash
$ julia scripts/2d-rosenbrock/train_unsupervised_hint_rosenbrock.jl
```

To create figures associated with the scripts above, run commands below:

```bash
$ julia scripts/2d-rosenbrock/test_warm-start_unsupervised_hint.jl
```

```bash
$ julia scripts/2d-rosenbrock/test_unsupervised_hint_rosenbrock.jl
```

Upon running the script above, figures will be created and saved at `plots/` directory.

### Seismic image compressed sensing example

In this examples, we train a normalizing flow on a joint distribution of seismic images and data, where data is obtained by applying a rank-deficient random matrix to individual images and adding Gaussian noise.

Run the script below for training the normalizing flow on 256x256 seismic images:

```bash
$ julia scripts/seismic-denoising/train_hint_denoising_256x256.jl
```

To perform joint or conditional (posterior) samples via the pretrained normalizing flow (obtained by running the script above), run:

```bash
$ julia scripts/seismic-denoising/test_hint_denoising_256x256.jl
```

To perform preconditioned training of a new normalizing flow to samples from the high-fidelity posterior, run:


```bash
$ julia scripts/seismic-denoising/train_warm-start_unsupervised_hint_256x256.jl
```

Finally, associated figures can be created by running the following script:


```bash
$ julia scripts/seismic-denoising/test_warm-start_unsupervised_hint_256x256.jl
```


## Citation

If you find this software useful in your research, please cite:


```bibtex
@conference {siahkoohi2020ABIpto,
  title = {Preconditioned training of normalizing flows for variational inference in inverse problems},
  booktitle = {{3rd Symposium on Advances in Approximate Bayesian Inference}},
  year = {2021},
  month = {1},
  abstract = {In the context of inverse problems with computationally expensive forward operators, specially for domains with limited access to high-fidelity training unknown and observed data pairs, we propose a preconditioned scheme for training a conditional normalizing flow (NF) capable of directly sampling the posterior distribution. Our training objective consists of the Kullback-Leibler divergence between the predicted and the desired posterior density. To minimize the costs associated with the forward operator, we initialize the NF via the weights of another pretrained low-fidelity NF, which is trained beforehand on available low-fidelity model and data pairs. Our numerical experiments, including a 2D toy and a seismic image compressed sensing example, demonstrate the improved performance and speed-up of the proposed method compared to training a NF from scratch.},
  author = {Ali Siahkoohi and Gabrio Rizzuti and Mathias Louboutin and Philipp Witte and Felix J. Herrmann},
  url={https://openreview.net/pdf?id=P9m1sMaNQ8T},
  keywords = {papers}
}
```


## Questions

Please contact alisk@gatech.edu for further questions.


## Author

Ali Siahkoohi
