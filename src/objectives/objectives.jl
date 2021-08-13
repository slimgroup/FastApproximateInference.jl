# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export loss_unsupervised, loss_supervised

function loss_unsupervised(
    Net::Union{InvertibleNetwork, ReverseNetwork},
    Y_obs,
    Zx,
    Zy,
    A::SquareCSop,
    sigma::Float32,
    RB_dist::RosenbrockDistribution
)
    X, Y, logdet = Net.forward(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)))
    CUDA.reclaim()

    Y_ = A*X
    f = -sum(logpdf(0f0, sigma, Y_ .- Y_obs)) + sum(nl_pdf(X, RB_dist)) - logdet*size(X, 4)
    ΔX = -A'*gradlogpdf(0f0, sigma, Y_ .- Y_obs) + nl_pdf_grad(X, RB_dist)
    ΔX = ΔX/size(X, 4)

    Net.backward(ΔX, 0f0.*ΔX, X, Y)
    CUDA.reclaim()
    GC.gc()

    return f/size(X, 4)
end


function loss_unsupervised(
    Net::ReverseNetwork,
    Y_obs,
    Zx,
    Zy,
    A::SquareCSop,
    sigma::Float32,
    Net_prior::NetworkConditionalHINT
)
    X, Y, logdet = Net.forward(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)); x_lane=true)
    CUDA.reclaim()

    Zx_p = Net_prior.forward(X, repeat(Y_obs, 1, 1, 1, size(X, 4)))[1]
    CUDA.reclaim()
    ΔZx = -gradlogpdf(0f0, 1f0, Zx_p)

    Y_ = A*X
    f = (-sum(logpdf(0f0, sigma, Y_ .- Y_obs))
         - sum(exact_likelihood(Net_prior, [Zx_p, Zy])) - logdet*size(X, 4))

    ΔX = (-A'*gradlogpdf(0f0, sigma, Y_ .- Y_obs)
          + Net_prior.backward(ΔZx, 0f0*Y, Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)))[1])
    CUDA.reclaim()
    ΔX = ΔX/size(X, 4)


    clear_grad!(Net_prior)
    Net.backward(ΔX, zero(ΔX), X, Y)
    CUDA.reclaim()

    GC.gc()

    return f/size(X, 4)
end


function loss_unsupervised(
    Net::ReverseNetwork,
    Y_obs,
    Zx,
    Zy,
    A::RombergOP,
    sigma::Float32,
    Net_prior::NetworkConditionalHINT
)
    X, Y, logdet = Net.forward(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)); x_lane=true)
    CUDA.reclaim()

    Zx_p = Net_prior.forward(X, repeat(Y_obs, 1, 1, 1, size(X, 4)))[1]
    CUDA.reclaim()
    ΔZx = -gradlogpdf(0f0, 1f0, Zx_p)

    Y_ = wavelet_squeeze(A*wavelet_unsqueeze(X |> cpu)) |> gpu
    f = (-sum(logpdf(0f0, sigma, Y_ .- Y_obs))
         - sum(exact_likelihood(Net_prior, [Zx_p, Zy])) - logdet*size(X, 4))

    ΔX = wavelet_squeeze(-A'*wavelet_unsqueeze(gradlogpdf(0f0, sigma, (Y_ .- Y_obs) |> cpu))) |> gpu
    ΔX += Net_prior.backward(ΔZx, 0f0*Y, Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)))[1]
    CUDA.reclaim()
    ΔX = ΔX/size(X, 4)


    clear_grad!(Net_prior)
    Net.backward(ΔX, zero(ΔX), X, Y)
    CUDA.reclaim()

    GC.gc()

    return f/size(X, 4)
end

function loss_unsupervised(
    Net::ReverseNetwork,
    Y_obs,
    Zx,
    Zy,
    A::Array{Float32,2},
    sigma::Float32,
    Net_prior::NetworkConditionalHINT
)
    X, Y, logdet = Net.forward(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)); x_lane=true)
    CUDA.reclaim()

    Zx_p = Net_prior.forward(X, repeat(Y_obs, 1, 1, 1, size(X, 4)))[1]
    CUDA.reclaim()
    ΔZx = -gradlogpdf(0f0, 1f0, Zx_p)

    Y_ = wavelet_squeeze(A.*wavelet_unsqueeze(X |> cpu)) |> gpu
    f = (-sum(logpdf(0f0, sigma, Y_ .- Y_obs))
         - sum(exact_likelihood(Net_prior, [Zx_p, Zy])) - logdet*size(X, 4))

    ΔX = wavelet_squeeze(-A'.*wavelet_unsqueeze(gradlogpdf(0f0, sigma, (Y_ .- Y_obs) |> cpu))) |> gpu
    ΔX += Net_prior.backward(ΔZx, 0f0*Y, Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)))[1]
    CUDA.reclaim()
    ΔX = ΔX/size(X, 4)


    clear_grad!(Net_prior)
    Net.backward(ΔX, zero(ΔX), X, Y)
    CUDA.reclaim()

    GC.gc()

    return f/size(X, 4)
end


function loss_supervised(
    Net::NetworkConditionalHINT,
    X::AbstractArray{Float32,4},
    Y::AbstractArray{Float32,4}
)

    Zx, Zy, logdet = Net.forward(X, Y)
    CUDA.reclaim()
    z_size = size(Zx)

    f = sum(logpdf(0f0, 1f0, Zx))
    f = f + sum(logpdf(0f0, 1f0, Zy))
    f = f + logdet*z_size[4]

    ΔZx = -gradlogpdf(0f0, 1f0, Zx)/z_size[4]
    ΔZy = -gradlogpdf(0f0, 1f0, Zy)/z_size[4]

    ΔX, ΔY = Net.backward(ΔZx, ΔZy, Zx, Zy)[1:2]
    CUDA.reclaim()
    GC.gc()

    return -f/z_size[4], ΔX, ΔY
end
