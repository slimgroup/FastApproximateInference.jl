# Test rosenbrock pdf gradients
# Authors: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020

using DrWatson
@quickactivate :FastApproximateInference
import Pkg; Pkg.instantiate()

using LinearAlgebra
using Test

# Input
n_samples = 1

# Rosenbrock distribution
RB_dist = RosenbrockDistribution(0f0, 5f-1, ones(Float32, 2, 1))

# samples
X0 = rand(RB_dist, n_samples)
X = rand(RB_dist, n_samples)
dX = X - X0

function rosenbrock_nlpdf(RB_dist, x)
    return nl_pdf(x, RB_dist)[1], nl_pdf_grad(x, RB_dist)
end

# Gradient test
f0, ΔX = rosenbrock_nlpdf(RB_dist, X0)
h = 0.1f0
maxiter = 5
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test\n")
for j=1:maxiter
    f = rosenbrock_nlpdf(RB_dist, X0 + h*dX)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(dX, ΔX))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
