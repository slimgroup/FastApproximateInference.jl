# Testing a pretrained normalizing flow for seismic denosing
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia test_hint_denoising.jl --batchsize 256 --max_epoch 5
"""

using DrWatson
@quickactivate :FastApproximateInference

using InvertibleNetworks
using JLD
using JLD2
using Random
using Statistics
using ArgParse
using BSON
using Printf
using PyPlot
using Seaborn
using Flux: gpu, cpu
using Flux.Data: DataLoader

set_style("whitegrid")
rc("font", family="serif", size=13)
font_prop = matplotlib.font_manager.FontProperties(
    family="serif",
    style="normal",
    size=13
)
sfmt=matplotlib.ticker.ScalarFormatter(useMathText=true)
sfmt.set_powerlimits((0, 0))

# Random seed
# Random.seed!(19)

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 50
    "--epoch"
        help = "Epoch to load net params"
        arg_type = Int
        default = -1
    "--lr"
        help = "starting learning rate"
        arg_type = Float32
        default = 1f-3
    "--lr_step"
        help = "lr scheduler applied at evey lr_step epoch"
        arg_type = Int
        default = 1
    "--batchsize"
        help = "batch size"
        arg_type = Int
        default = 4
    "--n_hidden"
        help = "# hidden channels"
        arg_type = Int
        default = 64
    "--depth"
        help = "depth of the network"
        arg_type = Int
        default = 12
    "--sigma"
        help = "noise std"
        arg_type = Float32
        default = 2f-2
    "--sim_name"
        help = "simulation name"
        arg_type = String
        default = "seismic-denoising-256x256-0.3CS"
end
parsed_args = parse_args(s)

max_epoch = parsed_args["max_epoch"]
lr = parsed_args["lr"]
lr_step = parsed_args["lr_step"]
batchsize = parsed_args["batchsize"]
n_hidden = parsed_args["n_hidden"]
depth = parsed_args["depth"]
sigma = parsed_args["sigma"]
sim_name = parsed_args["sim_name"]

if parsed_args["epoch"] == -1
    parsed_args["epoch"] = parsed_args["max_epoch"]
end
epoch = parsed_args["epoch"]

# Define raw data directory
mkpath(datadir("training-data"))
data_path = datadir("training-data", "seismic_samples_256_by_256_num_10k.jld")
label_path = datadir("training-data", "seismic_samples_256_by_256_num_10k_labels.jld")

# Download the dataset into the data directory if it does not exist
if isfile(data_path) == false
    run(`wget https://www.dropbox.com/s/vapl62yhh8fxgwy/'
        'seismic_samples_256_by_256_num_10k.jld -q -O $data_path`)
end
if isfile(label_path) == false
    run(`wget https://www.dropbox.com/s/blxkh6tdszudlcq/'
        'seismic_samples_256_by_256_num_10k_labels.jld -q -O $label_path`)
end

# Load seismic images and create training and testing data
JLD.jldopen(data_path, "r") do file
    X_orig = read(file, "X")

    labels = JLD.jldopen(label_path, "r")["labels"][:]
    idx = findall(x -> x == 1, labels)

    X_orig = X_orig[:, :, :, idx]
    global nx, ny, nc, nsamples = size(X_orig)

    # Whiten data
    X_orig = whiten_input(X_orig)

    # Load observed data
    Y_orig = wload(
        datadir("testing-data", "noisy_seismic_samples_256_by_256_num_10k.jld2")
    )["Y_orig"]

    # Load split in training/testing
    train_idx = wload(
        datadir("testing-data", "noisy_seismic_samples_256_by_256_num_10k.jld2")
    )["train_idx"]

    # Split in training - testing
    global test_size = 5
    global test_idx = shuffle(setdiff(1:nsamples, train_idx))[1:5]

    # Dimensions after wavelet squeeze to increase no. of channels
    global nx = Int(nx/2)
    global ny = Int(ny/2)
    global n_in = Int(nc*4)

    # Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
    global X_test = zeros(Float32, nx, ny, n_in, test_size)
    global Y_test = zeros(Float32, nx, ny, n_in, test_size)
    for (k, j) in enumerate(test_idx)
        X_test[:, :, :, k:k] = wavelet_squeeze(X_orig[:, :, :, j:j])
        Y_test[:, :, :, k:k] = wavelet_squeeze(Y_orig[:, :, :, j:j])
    end
end

# Create network
CH = NetworkConditionalHINT(
    nx, ny, n_in, batchsize, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
)

# Loading the experiment—only network weights and training loss
Params, fval = load_experiment(parsed_args)
put_params!(CH, convert(Array{Any,1}, Params))

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)

# Precited model and data
X_, Y_ = CH.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
X_fixed = X_test[:, :, :, 1:1]
Y_fixed = Y_test[:, :, :, 1:1]
Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
n_samples = 1000
X_post = zeros(Float32, nx, ny, n_in, n_samples)
CH = CH |> gpu

test_loader = DataLoader(
    (randn(Float32, nx, ny, n_in, n_samples), repeat(Zy_fixed, 1, 1, 1, n_samples)),
    batchsize=8, shuffle=false
)

for (itr, (X, Y)) in enumerate(test_loader)
    counter = (itr - 1)*size(X)[4] + 1

    X = X |> gpu
    Y = Y |> gpu

    X_post[:, :, :, counter:(counter+size(X)[4]-1)] = (CH.inverse(X, Y)[1] |> cpu)
    @printf("Sampling: [%d/%d]\r", counter, n_samples)
end

X_post = wavelet_unsqueeze(X_post)

# Some stats
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)

# Unsqueeze all tensors
X_test = wavelet_unsqueeze(X_test)
Y_test = wavelet_unsqueeze(Y_test)

X_ = wavelet_unsqueeze(X_)
Y_ = wavelet_unsqueeze(Y_)
Zx = wavelet_unsqueeze(Zx)
Zy = wavelet_unsqueeze(Zy)

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)
Zy_fixed = wavelet_unsqueeze(Zy_fixed)

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sim_name
save_path = plotsdir(sim_name, savename(save_dict; digits=6))

# Training loss
fig = figure("training logs", figsize=(7, 2.5))
plot(fval, color="#d48955")
title("Negative log-likelihood")
ylabel("Training objective")
xlabel("Iterations")
safesave(joinpath(save_path, "log.png"), fig)
close(fig)


# Plot the true model
fig = figure("x", figsize=(5, 5))
imshow(
    X_fixed[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"True model: $\mathbf{x} \sim \widehat{\pi}_x (\mathbf{x})$")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "true_model" * string(test_idx[1]) * ".png"), fig)
close(fig)

# Plot the observed data
fig = figure("y", figsize=(5, 5))
imshow(
    Y_fixed[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title(L"Observed data: $\mathbf{y} \sim \widehat{\pi}_y (\mathbf{y})$")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "observed_data" * string(test_idx[1]) * ".png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("x_cm", figsize=(5, 5))
imshow(
    X_post_mean[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Conditional mean")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "conditional_mean" * string(test_idx[1]) * ".png"), fig)
close(fig)

# Plot the pointwise standard deviation
fig = figure("x_std", figsize=(5, 5))
imshow(
    X_post_std[:, :, 1, 1], vmin=.30,
    vmax=.42, aspect=1, cmap="OrRd", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Pointwise STD")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "pointwise_std" * string(test_idx[1]) * ".png"), fig)
close(fig)

fig = figure("err", figsize=(5, 5))
imshow(
    2f0.*(X_fixed[:, :, 1, 1] - X_post_mean[:, :, 1, 1])./(abs.(X_fixed[:, :, 1, 1])  + abs.(X_post_mean[:, :, 1, 1])), vmin=-2, vmax=2, aspect=1, cmap="twilight_shifted", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Estimation error — accelerated")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "conditional_mean_err.png"), fig)
close(fig)
