# Training a normalizing flow for 2D Rosenbrock example
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia train_supervised_hint.jl
"""

using DrWatson
@quickactivate :FastApproximateInference

using InvertibleNetworks
using Random
using Statistics
using ArgParse
using BSON
using Printf
using Flux: gpu, cpu
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM

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

CH = NetworkConditionalHINT(n_in, n_hidden, depth) |> gpu

# Training data
ntrain = 5000
X_train = rand(RosenbrockDistribution(0f0, 5f-1, ones(Float32, 2, 1)), ntrain)
Y_train = X_train + sigma*randn(Float32, nx, ny, n_in, ntrain)
X_train = X_train |> gpu
Y_train = Y_train |> gpu

# Training
# Batch extractor
num_batches = cld(ntrain, batchsize)
train_loader = DataLoader((X_train, Y_train), batchsize=batchsize, shuffle=true)

# Optimizer
opt = Optimiser(ExpDecay(lr, .9f0, num_batches*lr_step, 0f0), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)

for j=1:max_epoch
    for (itr, (X, Y)) in enumerate(train_loader)

        fval[(j-1)*num_batches + itr] = loss_supervised(CH, X, Y)[1]

        @printf(
            "Epoch: [%d/%d] | Itr: [%d/%d] | Average negative log-likelihood: %4.2f\r",
            j, max_epoch, itr, num_batches, fval[(j-1)*num_batches + itr]
        )

        # Update params
        for p in get_params(CH)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end
end

# Saving parameters and logs
CH = CH |> cpu
Params = get_params(CH)
save_dict = @strdict max_epoch lr lr_step batchsize n_hidden depth sigma sim_name Params fval
@tagsave(datadir(sim_name, savename(save_dict, "bson")), save_dict; safe=true)
