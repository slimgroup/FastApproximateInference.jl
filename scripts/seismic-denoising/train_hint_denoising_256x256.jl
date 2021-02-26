# Train an normalizing flow for seismic denosing
# Authors: Ali Siahkoohi, alisk@gatech.edu
#          Gabrio Rizzuti, rizzuti.gabrio@gatech.edu
#          Philipp Witte, pwitte3@gatech.edu
# Date: September 2020
"""
Example:
julia train_hint_denoising.jl --batchsize 256 --max_epoch 5
"""

using DrWatson
@quickactivate :FastApproximateInference
import Pkg; Pkg.instantiate()

using InvertibleNetworks
using JLD
using JLD2
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
        default = 50
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
        default = 1f-1
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

    # Forward operator
    global A = RombergOP(nx, ny, nc, 2f0)

    # Create observed data
    Y_orig = A*X_orig + sigma*randn(Float32, size(X_orig))

    # Split in training/testing
    global ntrain = Int(floor((nsamples*.9)))
    global train_idx = randperm(nsamples)[1:ntrain]

    # Save data for testing phase
    save_dict = @strdict train_idx Y_orig A
    @tagsave(
        datadir("testing-data", "noisy_seismic_samples_256_by_256_num_10k.jld2"),
        save_dict
    )

    # Dimensions after wavelet squeeze to increase no. of channels
    global nx = Int(nx/2)
    global ny = Int(ny/2)
    global n_in = Int(nc*4)
end

# Create network
CH = NetworkConditionalHINT(
    nx, ny, n_in, batchsize, n_hidden, depth, k1=3, k2=3, p1=1, p2=1
) |> gpu

# Training
# Batch extractor
num_batches = cld(ntrain, batchsize)

X_train = JLD.jldopen(data_path, "r")["X"]
labels = JLD.jldopen(label_path, "r")["labels"][:]
idx = findall(x -> x == 1, labels)
X_train = X_train[:, :, :, idx]

Y_train = JLD2.jldopen(
    datadir("testing-data", "noisy_seismic_samples_256_by_256_num_10k.jld2"),
    "r"
)["Y_orig"]

train_loader = DataLoader(train_idx, batchsize=batchsize, shuffle=true)

# Optimizer
opt = Optimiser(ExpDecay(lr, .9f0, num_batches*lr_step, 0f0), ADAM(lr))

# Training log keeper
fval = zeros(Float32, num_batches*max_epoch)

for epoch=1:max_epoch
    for (itr, idx) in enumerate(train_loader)

        # Apply wavelet squeeze (change dimensions to -> n1/2 x n2/2 x nc*4 x ntrain)
        X = wavelet_squeeze(whiten_input(X_train[:, :, :, idx]))
        Y = wavelet_squeeze(Y_train[:, :, :, idx])

        X = X |> gpu
        Y = Y |> gpu

        fval[(epoch-1)*num_batches + itr] = loss_supervised(CH, X, Y)[1]

        @printf(
            "Epoch: [%d/%d] | Itr: [%d/%d] | Average negative log-likelihood: %4.2f\r",
            epoch, max_epoch, itr, num_batches, fval[(epoch-1)*num_batches + itr]
        )

        # Update params
        for p in get_params(CH)
            update!(opt, p.data, p.grad)
        end
        clear_grad!(CH)
    end

    # Saving parameters and logs
    Params = get_params(CH) |> cpu
    save_dict = @strdict epoch max_epoch lr lr_step batchsize n_hidden depth sigma sim_name A Params fval
    @tagsave(
        datadir(sim_name, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

end
