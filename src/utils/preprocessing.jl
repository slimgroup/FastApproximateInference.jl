export whiten_input

function whiten_input(X::Array{Float32,4})
    for j = 1:size(X, 4)
        x = X[:, :, :, j:j]
        X[:, :, :, j:j] = (x .- mean(x))/std(x)
    end
    return X
end
