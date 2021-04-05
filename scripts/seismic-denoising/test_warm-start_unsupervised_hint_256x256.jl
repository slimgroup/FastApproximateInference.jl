# Testing a normalizing flow for seismic denosing
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia test_warm-start_unsupervised_hint.jl
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


s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 15
    "--epoch"
        help = "Epoch to load net params"
        arg_type = Int
        default = 10
    "--lr"
        help = "starting learning rate"
        arg_type = Float32
        default = 1f-4
    "--lr_step"
        help = "lr scheduler applied at evey lr_step epoch"
        arg_type = Int
        default = 5
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
        default = 2f-1
    "--sim_name"
        help = "simulation name"
        arg_type = String
        default = "warm-start-unsupervised-hint-seismic-denoising-256x256-0.3CS"
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

# Loading the experiment—only network weights and training loss
Params, fval, exp_path = load_experiment(parsed_args; return_path=true)

Y_fixed = wload(exp_path)["Array(Y_obs)"]
X_fixed = wload(exp_path)["X_true"]
Zy_fixed = wload(exp_path)["Array(Zy_obs)"]
Zx = wload(exp_path)["Array(Z_train)"]
A = wload(exp_path)["A"]
pretrained_args = wload(exp_path)["pretrained_args"]

nx, ny, n_in = size(X_fixed)[1:3]

CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)
CH_rev = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)

put_params!(CH_rev, convert(Array{Any,1}, Params))
CH_rev = CH_rev |> gpu

Params0, fval0 = load_experiment(pretrained_args)
put_params!(CH, convert(Array{Any,1}, Params0))

# Testing data
test_size = size(Zx, 4)

# Now select single fixed sample from all Ys
Y_fixed = Y_fixed |> gpu
Zy_fixed = Zy_fixed |> gpu

CH_rev = reverse(CH_rev)

# Draw new Zx, while keeping Zy fixed
test_size = 1000
X_post0 = zeros(Float32, nx, ny, n_in, test_size)
X_post = zeros(Float32, nx, ny, n_in, test_size)

CH = CH |> gpu

test_loader = DataLoader(randn(Float32, nx, ny, n_in, test_size) |> gpu, batchsize=8, shuffle=false)

for (itr, zx) in enumerate(test_loader)
    counter = (itr - 1)*size(zx)[4] + 1

    zx = zx |> gpu

    X_post0[:, :, :, counter:(counter+size(zx)[4]-1)] = (CH.inverse(zx, repeat(Zy_fixed, 1, 1, 1, size(zx, 4)))[1] |> cpu)
    X_post[:, :, :, counter:(counter+size(zx)[4]-1)] = (CH_rev.forward(zx, repeat(Zy_fixed, 1, 1, 1, size(zx, 4)))[1] |> cpu)
    @printf("Sampling: [%d/%d]\r", counter, test_size)
end

X_post = wavelet_unsqueeze(X_post)
X_post0 = wavelet_unsqueeze(X_post0)

X_fixed = wavelet_unsqueeze(X_fixed)
Y_fixed = wavelet_unsqueeze(Y_fixed)

# Some stats
X_post_mean = mean(X_post; dims=4)
X_post_std = std(X_post; dims=4)
X_post_mean0 = mean(X_post0; dims=4)
X_post_std0 = std(X_post0; dims=4)

save_dict = @strdict max_epoch epoch lr lr_step batchsize n_hidden depth sigma
save_path = plotsdir(sim_name, savename(save_dict; digits=6))


# Plot the conditional mean estimate
fig = figure("x_cm", figsize=(5, 5))
imshow(
    X_fixed[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title("True model")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "true_model.png"), fig)
close(fig)

fig = figure("x_cm", figsize=(5, 5))
imshow(
    Y_fixed[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Observed data")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "observed_data.png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("x_cm", figsize=(5, 5))
imshow(
    X_post_mean[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Conditional mean — Equation (6)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "conditional_mean.png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("x_cm", figsize=(5, 5))
imshow(
    X_post_mean0[:, :, 1, 1], vmin=-4.0, vmax=4.0, aspect=1, cmap="Greys", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Conditional mean — Equation (3)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "pretrained-conditional_mean.png"), fig)
close(fig)


# Plot the pointwise standard deviation
fig = figure("x_std", figsize=(5, 5))
imshow(
    X_post_std0[:, :, 1, 1], vmin=.30,
    vmax=.42, aspect=1, cmap="OrRd", resample=true,
    interpolation="lanczos", filterrad=1
)
title("STD — Equation (3)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "pretrained-pointwise_std.png"), fig)
close(fig)

# Plot the pointwise standard deviation
fig = figure("x_std", figsize=(5, 5))
imshow(
    X_post_std[:, :, 1, 1], vmin=.13,
    vmax=.19, aspect=1, cmap="OrRd", resample=true,
    interpolation="lanczos", filterrad=1
)
title("STD — Equation (6)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "pointwise_std.png"), fig)
close(fig)


# Plot the conditional mean estimate
fig = figure("err", figsize=(5, 5))
imshow(
    2f0.*(X_fixed[:, :, 1, 1] - X_post_mean[:, :, 1, 1])./(abs.(X_fixed[:, :, 1, 1])  + abs.(X_post_mean[:, :, 1, 1])), vmin=-2.0, vmax=2.0, aspect=1, cmap="twilight_shifted", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Relative error — Equation (6)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "conditional_mean_err.png"), fig)
close(fig)

# Plot the conditional mean estimate
fig = figure("err", figsize=(5, 5))
imshow(
    2f0.*(X_fixed[:, :, 1, 1] - X_post_mean0[:, :, 1, 1])./(abs.(X_fixed[:, :, 1, 1])  + abs.(X_post_mean0[:, :, 1, 1])), vmin=-2.0, vmax=2.0, aspect=1, cmap="twilight_shifted", resample=true,
    interpolation="lanczos", filterrad=1
)
title("Relative error — Equation (3)")
colorbar(fraction=0.047, pad=0.01, format=sfmt)
grid(false)
ax = plt.gca()
ax.axes.xaxis.set_visible(false)
ax.axes.yaxis.set_visible(false)
safesave(joinpath(save_path, "pretrained-conditional_mean_err.png"), fig)
close(fig)
