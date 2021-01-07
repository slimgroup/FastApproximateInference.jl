# Rosenbrock distribution — based on https://arxiv.org/abs/1903.09556
# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export RosenbrockDistribution, nl_pdf, nl_pdf_grad, rand

import Random: rand

struct RosenbrockDistribution
    μ::Float32
    a::Float32
    b::Array{Float32,2}
    n1::Int64
    n2::Int64
end


function RosenbrockDistribution(μ::Float32, a::Float32, b::Array{Float32,2})
    return RosenbrockDistribution(μ, a, b, size(b, 1), size(b, 2))
end


function nl_pdf_grad(X::AbstractArray{Float32, 4}, RB::RosenbrockDistribution)

    @assert size(X, 3) == 2 "Not implemented for larger dimension"

    nl_pdf_grad = cat(
        2f0*RB.a*(X[:, :, 1:1, :] .- RB.μ) - 4f0*RB.b[2, 1]*X[:, :, 1:1, :].*(X[:, :, 2:2, :] - X[:, :, 1:1, :].^2f0),
        2f0*RB.b[2, 1]*(X[:, :, 2:2, :] - X[:, :, 1:1, :].^2f0); dims=3)
    return nl_pdf_grad
end


function nl_pdf(X::AbstractArray{Float32, 4}, RB::RosenbrockDistribution)

    @assert size(X, 3) == (RB.n1 - 1) * RB.n2 + 1

    nlpdf = RB.a * (X[1, 1, 1, :] .- RB.μ).^2f0

    for j = 1:RB.n2
        for i = 2:RB.n1
            nlpdf += RB.b[i, j]*(
                X[1, 1, i + (j-1)*(RB.n1-1), :]
                - X[1, 1, i - 1 + (j-1)*(RB.n1-1), :].^2f0
            ).^2f0
        end
    end

    nc = sqrt(RB.a)/((1f0π)^(5f-1*((RB.n1 - 1) * RB.n2 + 1)))
    for j = 1:RB.n2
        for i = 2:RB.n1
            nc *= sqrt(RB.b[i, j])
        end
    end
    return -log(nc) .+ nlpdf
end


function rand(RB::RosenbrockDistribution, n_samples::Int64)

    dim = (RB.n1 - 1) * RB.n2 + 1
    X = zeros(Float32, 1, 1, dim, n_samples)

    X[1, 1, 1, :] = randn(Float32, n_samples)/sqrt(2f0*RB.a) .+ RB.μ
    for j = 1:RB.n2
        for i = 2:RB.n1
            X[1, 1, i+(j-1)*(RB.n1-1), :] = randn(Float32, n_samples)/sqrt(2f0*RB.b[i, j])
            X[1, 1, i+(j-1)*(RB.n1-1), :] += X[1, 1, i-1+(j-1)*(RB.n1 - 1), :].^2f0
        end
    end

    return X
end
