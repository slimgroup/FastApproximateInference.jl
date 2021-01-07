# Testing a pretrained normalizing flow for 2D Rosenbrock example
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia test_supervised_hint.jl
"""

using DrWatson
@quickactivate :FastApproximateInference
import Pkg; Pkg.instantiate()

using InvertibleNetworks
using LinearAlgebra
using Random
using Distributions
using Statistics
using ArgParse
using BSON
using PyPlot
using Seaborn
using Printf
using Flux
using Flux.Optimise: update!

set_style("whitegrid")
rc("font", family="serif", size=8)
font_prop = matplotlib.font_manager.FontProperties(
    family="serif",
    style="normal",
    size=8
)

# Random seed
# Random.seed!(19)

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 25
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
        default = 64
    "--n_hidden"
        help = "# hidden channels"
        arg_type = Int
        default = 64
    "--depth"
        help = "depth of the network"
        arg_type = Int
        default = 8
    "--sigma"
        help = "noise std"
        arg_type = Float32
        default = 4f-1
    "--sim_name"
        help = "simulation name"
        arg_type = String
        default = "supervised-hint-rosenbrock"
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

# Define network
nx = 1
ny = 1
n_in = 2

CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)

# Noise distribution
π_t = Normal(0f0, sigma)

# Loading the experiment—only network weights and training loss
Params, fval = load_experiment(parsed_args)
put_params!(CH, Params)

# Testing data
test_size = 1000
RB_dist = RosenbrockDistribution(0f0, 5f-1, ones(Float32, 2, 1))
X_test = rand(RB_dist, test_size)
Y_test = X_test + sigma*randn(Float32, nx, ny, n_in, test_size)

# Predicted latent varables
Zx_, Zy_ = CH.forward(X_test, Y_test)[1:2]

Zx = randn(Float32, nx, ny, n_in, test_size)
Zy = randn(Float32, nx, ny, n_in, test_size)

# Precited model and data
X_, Y_ = CH.inverse(Zx, Zy)

# Now select single fixed sample from all Ys
X_fixed = zeros(Float32, 1, 1, 2, 1)
X_fixed[1, 1, 1, 1] = 2f0
X_fixed[1, 1, 2, 1] = 4f0
Y_fixed = X_fixed + sigma*randn(Float32, nx, ny, n_in, 1)

Zy_fixed = CH.forward_Y(Y_fixed)

# Draw new Zx, while keeping Zy fixed
X_post = CH.inverse(Zx, repeat(Zy_fixed, 1, 1, 1, test_size))[1]

# Likelihood histogtams
loglike_CH = exact_likelihood(CH, X_test, Y_test)

loglike_true = -nl_pdf(X_test, RB_dist) + sum(
    logpdf.(π_t, Y_test - X_test),
    dims=[1, 2, 3]
)[1, 1, 1, :]

# Create Flux parameter
x̂ = randn(Float32, size(X_fixed))
θ = Flux.Params([x̂])

# Optimizer
η = 0.05
opt = pSGLD(η)
max_itr = 20*test_size
X_sgld = zeros(Float32, 1, 1, 2, max_itr)


function objective(x_in, y_in, RB_dist)
    return (1f0/(2f0*sigma^2f0))*sum((x_in - y_in).^2f0) + sum(nl_pdf(x_in, RB_dist))
end

for j = 1:max_itr

    # Evaluate objective and gradients
    obj, back = Flux.pullback(θ) do
      objective(x̂, Y_fixed, RB_dist)
    end
    grads = back(1.0f0)
    update!(opt, x̂, grads[x̂])

    if j%100 == 0
      @printf("SGLD iterations: [%d/%d] | Objective: %4.2f\r", j, max_itr, obj)
    end
    X_sgld[:, :, :, j:j] = x̂
end

# Warm-up phase
X_sgld = X_sgld[:, :, :, fld(max_itr, 2):end]

# Thinning
X_sgld = X_sgld[:, :, :, 1:5:end]

save_dict = @strdict max_epoch lr lr_step batchsize n_hidden depth
save_path = plotsdir(sim_name, savename(save_dict))

# Training loss
fig = figure("training logs", figsize=(7, 2.5))
plot(fval, label="supervised HINT", color="#d48955")
title(L"$- \log p_{(y, x)} (y, x ) + const.$")
ylabel("Training loss")
xlabel("Iterations")
legend(loc="upper right", ncol=1, fontsize=9)
wsave(joinpath(save_path, "log.png"), fig)
close(fig)

# Plot one sample from X and Y and their latent versions
fig = figure(figsize=(18, 8))
subplot(2,4,1)
scatter(X_test[1, 1, 1, :], X_test[1, 1, 2, :], s=.5, color="#5e838f")
plot(X_fixed[1, 1, 1, :], X_fixed[1, 1, 2, :], "k*")
xlim([-4, 4])
ylim([-2, 10])
title(L"Model space: $x \sim \hat{p}_x$")

subplot(2,4,2)
scatter(Y_test[1, 1, 1, :], Y_test[1, 1, 2, :], s=.5, color="#5e838f")
plot(Y_fixed[1, 1, 1, :], Y_fixed[1, 1, 2, :], "k*")
xlim([-4, 4])
ylim([-2, 10])
title(L"Noisy data $y=x+n$ ")

subplot(2,4,3)
scatter(X_[1, 1, 1, :], X_[1, 1, 2, :], s=.5, color="#db76bf")
scatter(X_post[1, 1, 1, :], X_post[1, 1, 2, :], s=.5, color="#f2dc16")
scatter(X_sgld[1, 1, 1, :], X_sgld[1, 1, 2, :], s=.5, color="k", alpha=.2)
xlim([-4, 4])
ylim([-2, 10])
title(L"Model space: $x = f(z_x|z_y)^{-1}$")

subplot(2,4,4)
scatter(Y_[1, 1, 1, :], Y_[1, 1, 2, :], s=.5, color="#db76bf")
xlim([-4, 4])
ylim([-2, 10])
title(L"Data space: $y = f(z_x|z_y)^{-1}$")

subplot(2,4,5)
scatter(Zx_[1, 1, 1, :], Zx_[1, 1, 2, :], s=.5, color="#5e838f")
xlim([-4, 4])
ylim([-4, 4])
title(L"Latent space: $z_x = f(x|y)$")

subplot(2,4,6)
scatter(Zy_[1, 1, 1, :], Zy_[1, 1, 2, :], s=.5, color="#5e838f")
xlim([-4, 4])
ylim([-4, 4])
title(L"Latent space: $z_y = f(x|y)$")

subplot(2,4,7)
scatter(Zx[1, 1, 1, :], Zx[1, 1, 2, :], s=.5, color="#db76bf")
xlim([-4, 4])
ylim([-4, 4])
title(L"Latent space: $z_x \sim \hat{p}_{z_x}$")

subplot(2,4,8)
scatter(Zy[1, 1, 1, :], Zy[1, 1, 2, :], s=.5, color="#db76bf")
xlim([-4, 4])
ylim([-4, 4])
title(L"Latent space: $z_y \sim \hat{p}_{z_y}$")

wsave(joinpath(save_path, "true-x-and-their-zs.png"), fig)
close(fig)

fig = figure("histogram", figsize=(7, 2.5))
ax = distplot(
    loglike_true,
    kde=true,
    bins=50,
    hist_kws=Dict(
        "density"=> true,
        "label"=> "true log-likelihood",
        "alpha"=> 0.8,
        "histtype"=> "bar"
    ),
    color="#ff8800"
)
distplot(
    loglike_CH,
    kde=true,
    bins=50,
    hist_kws=Dict(
        "density"=> true,
        "label"=> "estimated log-likelihood",
        "alpha"=> 0.5,
        "histtype"=> "bar"
    ),
    color="#00b4ba")

for label in ax.get_xticklabels()
    label.set_fontproperties(font_prop)
end
for label in ax.get_yticklabels()
    label.set_fontproperties(font_prop)
end
ax.set_xlabel("Log-likelihood", fontproperties=font_prop)
ax.legend(prop=font_prop)
wsave(joinpath(save_path, "log-like-hist.png"), fig)
close(fig)
