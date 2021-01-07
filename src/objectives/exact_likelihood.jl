# Author: Ali Siahkoohi, alisk@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020

export exact_likelihood


function exact_likelihood(Net::NetworkConditionalHINT, X, Y)

    Zx, Zy, logdet = Net.forward(X, Y)

    loglike = sum(logpdf(0f0, 1f0, cat(Zx, Zy; dims=3)), dims=[1, 2, 3])[1, 1, 1, :]
    loglike = loglike .+ logdet

    return loglike

end


function exact_likelihood(Net::NetworkConditionalHINT, Z)

    Zx, Zy = Z[1], Z[2]

    logdet = Net.inverse(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)); logdet=true, x_lane=true)[3]

    loglike = sum(logpdf(0f0, 1f0, Zx), dims=[1, 2, 3])[1, 1, 1, :]
    loglike = loglike .- logdet

    return loglike

end


function exact_likelihood(Net::ReverseNetwork, Z)

    Zx, Zy = Z[1], Z[2]

    logdet = Net.forward(Zx, repeat(Zy, 1, 1, 1, size(Zx, 4)); logdet=true, x_lane=true)[3]

    loglike = sum(logpdf(0f0, 1f0, Zx), dims=[1, 2, 3])[1, 1, 1, :]
    loglike = loglike .- logdet

    return loglike

end
