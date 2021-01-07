# Testing a pretrained normalizing flow for 2D Rosenbrock example
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia train_unsupervised_hint_rosenbrock.jl
"""

using DrWatson
@quickactivate :FastApproximateInference
import Pkg; Pkg.instantiate()

using InvertibleNetworks
using LinearAlgebra
using Random
using Statistics
using ArgParse
using BSON
using PyPlot
using Seaborn
using Printf
using Flux
using Flux.Optimise: update!

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
Random.seed!(19)

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
        default = "unsupervised-hint-rosenbrock"
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

CH_rev_precond = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)
CH_rev = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)
CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth)

# Loading the experiment—only network weights and training loss
Params, fval, exp_path = load_experiment(parsed_args; return_path=true)

Y_fixed = wload(exp_path)["Y_obs"]
Zy_fixed = wload(exp_path)["Zy_obs"]
X_fixed = wload(exp_path)["X_true"]
Zx = wload(exp_path)["Z_train"]
A = wload(exp_path)["A.A"]
warmstarted_args = wload(exp_path)["warmstarted_args"]

put_params!(CH_rev, Params)
CH_rev = reverse(CH_rev)

Params_precond, fval_precond, exp_path_precond = load_experiment(
    warmstarted_args;
    return_path=true
)
pretrained_args = wload(exp_path_precond)["pretrained_args"]

put_params!(CH_rev_precond, Params_precond)
CH_rev_precond = reverse(CH_rev_precond)

Params0, fval0 = load_experiment(pretrained_args)
put_params!(CH, Params0)

# Forward operator
A = SquareCSop(A, n_in)

# Testing data
test_size = size(Zx, 4)
RB_dist = RosenbrockDistribution(0f0, 5f-1, ones(Float32, 2, 1))
X_test = rand(RB_dist, 20*test_size)
Y_test = A*X_test + sigma*randn(Float32, nx, ny, n_in, 20*test_size)
Y_train = X_test + sigma*randn(Float32, nx, ny, n_in, 20*test_size)

# Drawing posterior samples given y* (from trained network w/ warmstart)
X_post_precond = CH_rev_precond.forward(
    randn(Float32, nx, ny, n_in, 20*test_size),
    repeat(Zy_fixed, 1, 1, 1, 20*test_size)
)[1]

X_post = CH_rev.forward(
    randn(Float32, nx, ny, n_in, 20*test_size),
    repeat(Zy_fixed, 1, 1, 1, 20*test_size)
)[1]

X_post0 = CH.inverse(
    randn(Float32, nx, ny, n_in, 20*test_size),
    repeat(Zy_fixed, 1, 1, 1, 20*test_size)
)[1]


# Create Flux parameter
x̂ = randn(Float32, size(X_fixed))
θ = Flux.Params([x̂])

# Optimizer
η = 0.08
opt = pSGLD(η)
max_itr = 40*test_size
X_sgld = zeros(Float32, 1, 1, 2, max_itr)

function objective(x_in, y_in, RB_dist)
    return (1f0/(2f0*sigma^2f0))*sum((A*x_in - y_in).^2f0) + sum(nl_pdf(x_in, RB_dist))
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

Training loss
fig = figure("training logs", figsize=(5, 5))
plot(
    range(1, max_epoch, length=length(fval)), fval,
    label="Equation (2)", color="#addbc2"
)
plot(
    range(1, warmstarted_args["max_epoch"], length=length(fval_precond)), fval_precond,
    label="Equation (6)", color="#ff5900"
)
plot(
    range(0, max_epoch+1, length=length(fval)), 4.3f0*ones(Float32, length(fval)),
    color="k", linestyle="--", linewidth=.5
)
title("Training objective")
ylabel(L"$\mathbb{D}_{KL} + const.$")
xlabel("Epochs")
xlim([.6, max_epoch+1])
legend(loc="upper right", ncol=1)
wsave(joinpath(save_path, "training-obj.png"), fig)
close(fig)

# True samples from Rosenbrock distribuition
fig = figure("rosenbrock samples", figsize=(5, 5))
cmap = matplotlib.cm.Greys(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
ax = kdeplot(
    X_test[1, 1, 1, :], X_test[1, 1, 2, :], shade=false, cut=0, levels=15,
    cmap=cmap, bw_adjust=1.5, alpha=1.0
)
scatter(
    X_fixed[1, 1, 1, :], X_fixed[1, 1, 2, :], s=30.0, color="#000000", marker="*",
    label="true model"
)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.legend(loc="upper right", ncol=2)
ax.set_title("Rosenbrock prior density")
wsave(joinpath(save_path, "true-samples.png"), fig)
close(fig)


# Data samples
cmap = matplotlib.cm.Greys_r(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
fig = figure("data samples", figsize=(5, 5))
ax = kdeplot(
    Y_test[1, 1, 1, :], Y_test[1, 1, 2, :], shade=false, cut=0, levels=15,
    bw_adjust=1.7, alpha=1.0, cmap=cmap
)
label1 = matplotlib.patches.Patch(color="#4a4a4a", label=L"$\mathbf{y} = \mathbf{A} \mathbf{x} + \mathbf{\epsilon}$")
cmap = matplotlib.cm.Reds(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[20:end,1:end])
fig = figure("data samples", figsize=(5, 5))
ax = kdeplot(
    Y_train[1, 1, 1, :], Y_train[1, 1, 2, :], shade=false, cut=0, levels=15,
    bw_adjust=1.7, cmap=cmap, alpha=0.5
)
label2 = matplotlib.patches.Patch(color="#e05a5a", label=L"$\mathbf{y} = \mathbf{x} + \mathbf{\epsilon}$")
ax.legend(
    handles=[label1, label2], loc="upper left", ncol=2,
    markerscale=5
)
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$y_2$")
ax.set_xlabel(L"$y_1$")
ax.set_title("Data space")
wsave(joinpath(save_path, "test-data-samples.png"), fig)
close(fig)


# Predicted posterior samples
fig = figure("rosenbrock samples", figsize=(5, 5))
cmap = matplotlib.cm.BuGn(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[15:end,1:end])
ax = kdeplot(
    X_post[1, 1, 1, :], X_post[1, 1, 2, :], shade=true, cut=0, levels=10,
    cmap=cmap, alpha=0.4, bw_adjust=1.7
)
label1 = matplotlib.patches.Patch(color="#addbc2", label="Equation (2)")

cmap = matplotlib.cm.Reds(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
kdeplot(
    X_post_precond[1, 1, 1, :], X_post_precond[1, 1, 2, :], cmap=cmap, shade=false,
    alpha=1.0, cut=0, levels=10, linewidth=1, bw_adjust=1.7
)
label2 = matplotlib.patches.Patch(color="#ff5900", label="Equation (6)")

label3 = scatter(
    X_sgld[1, 1, 1, :], X_sgld[1, 1, 2, :], s=.5, color="#000000", alpha=.3,
    label="MCMC"
)
ax.set_xlim([0.75, 2.5])
ax.set_ylim([1.75, 5.0])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("Posterior density")
ax.legend(
    handles=[label1, label2, label3], loc="upper left", ncol=2,
    markerscale=5
)
wsave(joinpath(save_path, "unsupervised-posterior-samples.png"), fig)
close(fig)


fig = figure("rosenbrock samples", figsize=(5, 5))
cmap = matplotlib.cm.Greys(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
ax = kdeplot(
    X_test[1, 1, 1, :], X_test[1, 1, 2, :], shade=false, cut=0, levels=15,
    cmap=cmap, bw_adjust=1.5, alpha=0.9
)
label1 = matplotlib.patches.Patch(color="#4a4a4a", label="Prior")

cmap = matplotlib.cm.Reds(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
kdeplot(
    X_post_precond[1, 1, 1, :], X_post_precond[1, 1, 2, :], cmap=cmap, shade=false,
    alpha=0.8, cut=0, levels=10, linewidth=1, bw_adjust=1.7
)
label2 = matplotlib.patches.Patch(color="#ff5900", label="Equation (6)")

cmap = matplotlib.cm.Blues(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[15:end,1:end])
ax = kdeplot(
    X_post0[1, 1, 1, :], X_post0[1, 1, 2, :], shade=true, cut=0, levels=10,
    cmap=cmap, alpha=0.4, bw_adjust=1.7
)
label3 = matplotlib.patches.Patch(color="#65aadb", label="Equation (3)")
ax.set_xlim([-3, 3])
ax.set_ylim([-2.5, 7])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.legend(
    handles=[label1, label2, label3], loc="upper left", ncol=2,
    markerscale=5
)
ax.set_title("Prior vs. posterior density")
wsave(joinpath(save_path, "warmstarted-overlaid-posterior.png"), fig)
close(fig)



# Preeicted prior samples and posterior samples
fig = figure("rosenbrock samples", figsize=(5, 5))
cmap = matplotlib.cm.Blues(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[15:end,1:end])
ax = kdeplot(
    X_post0[1, 1, 1, :], X_post0[1, 1, 2, :], shade=true, cut=0, levels=10,
    cmap=cmap, alpha=0.4, bw_adjust=1.7
)
label1 = matplotlib.patches.Patch(color="#65aadb", label="Equation (3)")

cmap = matplotlib.cm.OrRd(range(0, 1, length=50))
cmap = matplotlib.colors.ListedColormap(cmap[13:end,1:end])
kdeplot(
    X_post_precond[1, 1, 1, :], X_post_precond[1, 1, 2, :], cmap=cmap, shade=false, alpha=1.0, cut=0,
    levels=10, linewidth=1, bw_adjust=1.7
)
label2 = matplotlib.patches.Patch(color="#ff5900", label="Equation (6)")
ax.set_xlim([0.75, 2.5])
ax.set_ylim([1.75, 5.0])
ax.set_ylabel(L"$x_2$")
ax.set_xlabel(L"$x_1$")
ax.set_title("Low- and high-fidelity Posterior")
ax.legend(
    handles=[label1, label2], loc="upper left", ncol=2,
    markerscale=5
)
wsave(joinpath(save_path, "prior-posterior-samples.png"), fig)
close(fig)
