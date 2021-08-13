# Train an normalizing flow for seismic denosing
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
using JLD
using JLD2
using Random
using Statistics
using ArgParse
using Printf
using Flux: gpu, cpu
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using JOLI4Flux

# Random seed
Random.seed!(0)

s = ArgParseSettings()
@add_arg_table! s begin
    "--max_epoch"
        help = "Maximum # of epochs"
        arg_type = Int
        default = 15
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
        default = 1f-1
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

# Loading the experiment—only network weights and training loss
pretrained_args = Dict(
    "max_epoch" => 50,
    "epoch"     => 50,
    "lr"        => 1f-3,
    "lr_step"   => 1,
    "batchsize" => 4,
    "n_hidden"  => 64,
    "depth"     => 12,
    "sigma"     => 2f-2,
    "sim_name"  => "seismic-denoising-256x256-0.3CS"
)
Params, fval, exp_path = load_experiment(pretrained_args; return_path=true)
A = wload(exp_path)["A"]

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

# Load seismic images and create one observed data
JLD.jldopen(data_path, "r") do file
    X_orig = read(file, "X")

    labels = JLD.jldopen(label_path, "r")["labels"][:]
    idx = findall(x -> x == -1, labels)

    X_orig = X_orig[:, :, :, idx]
    global nx, ny, nc, nsamples = size(X_orig)

    global nx = Int(nx/2)
    global ny = Int(ny/2)
    global nc = Int(nc*4)

    # Whiten data
    X_orig = whiten_input(X_orig)

    idx_fixed = 733
    global X_true = X_orig[:, :, :, idx_fixed:idx_fixed]
end

# Define prior and warmstart network
CH = NetworkConditionalHINT(nc, n_hidden, depth)
CH_prior = NetworkConditionalHINT(nc, n_hidden, depth)

put_params!(CH, convert(Array{Any,1}, Params))
put_params!(CH_prior, convert(Array{Any,1}, copy(Params)))

# Observed data
Y_obs = A*X_true + sigma*randn(Float32, size(X_true))

X_true = wavelet_squeeze(X_true)
Y_obs = wavelet_squeeze(Y_obs)

CH = CH |> gpu
Y_obs = Y_obs |> gpu

# Corresponding data latent variable
Zy_obs = CH.forward_Y(Y_obs)

# reversed network
CH_prior = CH_prior |> gpu
CHrev = reverse(CH)

# Training
# Batch extractor
ntrain = 1000
Z_train = randn(Float32, nx, ny, nc, ntrain) |> gpu

num_batches = cld(ntrain, batchsize)
train_loader = DataLoader(Z_train, batchsize=batchsize, shuffle=true)

# Optimizer
opt = Optimiser(ExpDecay(lr, .9f0, num_batches*lr_step, 0f0), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)

for epoch=1:max_epoch
    for (itr, Zx) in enumerate(train_loader)

        fval[(epoch-1)*num_batches + itr] = loss_unsupervised(
            CHrev, Y_obs, Zx, Zy_obs, A, sigma, CH_prior
        )

        @printf(
            "Epoch: [%d/%d] | Itr: [%d/%d] | Average log-likelihood: %4.2f\r",
            epoch, max_epoch, itr, num_batches, fval[(epoch-1)*num_batches + itr]
        )

        # Update params
        for p in get_params(CHrev)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(CHrev)
    end

    # Saving parameters and logs
    Params = get_params(CHrev) |> cpu
    save_dict = @strdict epoch max_epoch lr lr_step batchsize n_hidden depth sigma sim_name Array(Y_obs) Array(Zy_obs) X_true Array(Z_train) A pretrained_args Params fval
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

end
