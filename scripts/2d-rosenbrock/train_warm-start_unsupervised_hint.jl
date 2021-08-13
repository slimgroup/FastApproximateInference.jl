# Training a normalizing flow for 2D Rosenbrock example
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia train_warm-start_unsupervised_hint.jl
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
        default = 5
    "--lr"
        help = "starting learning rate"
        arg_type = Float32
        default = 1f-3
    "--lr_step"
        help = "lr scheduler applied at evey lr_step epoch"
        arg_type = Int
        default = 12
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
        default = "warm-start-unsupervised-hint-rosenbrock"
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

CH = NetworkConditionalHINT(n_in, n_hidden, depth)

# Loading the experiment—only network weights and training loss
pretrained_args = Dict(
    "max_epoch" => 25,
    "lr"        => 1f-3,
    "lr_step"   => 1,
    "batchsize" => 64,
    "n_hidden"  => 64,
    "depth"     => 8,
    "sigma"     => 4f-1,
    "sim_name"  => "supervised-hint-rosenbrock"
)
Params, fval = load_experiment(pretrained_args)
put_params!(CH, Params)

# Prior distribution
RB_dist = RosenbrockDistribution(0f0, 5f-1, ones(Float32, 2, 1))

# Forward operator
A = SquareCSop(n_in; s=3f0)

# Observed data
X_true = zeros(Float32, 1, 1, 2, 1)
X_true[1, 1, 1, 1] = 2f0
X_true[1, 1, 2, 1] = 4f0
Y_obs = A*X_true + sigma*randn(Float32, nx, ny, n_in, 1)

# Corresponding data latent variable
Zy_obs = CH.forward_Y(Y_obs)

# reversed network
CHrev = reverse(CH)

# Training
# Batch extractor
ntrain = 1000
Z_train = randn(Float32, nx, ny, n_in, ntrain)

num_batches = cld(ntrain, batchsize)
train_loader = DataLoader(Z_train, batchsize=batchsize, shuffle=true)

# Optimizer
opt = Optimiser(ExpDecay(lr, .9f0, num_batches*lr_step, 0f0), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)

for j=1:max_epoch
    for (itr, Zx) in enumerate(train_loader)

        fval[(j-1)*num_batches + itr] = loss_unsupervised(
            CHrev, Y_obs, Zx, Zy_obs, A, sigma, RB_dist
        )

        @printf(
            "Epoch: [%d/%d] | Itr: [%d/%d] | Objective: %4.2f\r",
            j, max_epoch, itr, num_batches, fval[(j-1)*num_batches + itr]
        )

        # Update params
        for p in get_params(CHrev)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(CHrev)
    end
end

# Saving parameters and logs
CHrev = CHrev |> cpu
Params = get_params(CHrev)
save_dict = @strdict max_epoch lr lr_step batchsize n_hidden depth sigma sim_name Y_obs Zy_obs X_true Z_train A.A pretrained_args Params fval
@tagsave(datadir(sim_name, savename(save_dict, "bson")), save_dict; safe=true)
