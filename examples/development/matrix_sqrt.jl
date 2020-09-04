using LinearAlgebra, ForwardDiff

α = 1.0
x11 = α*[1.0; 1.0]
x12 = α*[1.0; -1.0]
x13 = α*[-1.0; 1.0]
x14 = α*[-1.0; -1.0]
x1 = [x11,x12,x13,x14]
x1 = [zeros(2) for i = 1:4]

function sample_mean(X)
    N = length(X)
    nx = length(X[1])

    xμ = sum(X)./N
end

function sample_covariance(X; β=1.0,w=ones(length(X[1])))
    N = length(X)
    xμ = sample_mean(X)
    Σμ = (0.5/(β^2))*sum([(X[i] - xμ)*(X[i] - xμ)' for i = 1:N]) + Diagonal(w)
end

function resample(X; β=1.0,w=ones(length(X[1])))
    N = length(X)
    nx = length(X[1])
    xμ = sample_mean(X)
    Σμ = sample_covariance(X,β=β,w=w)
    cols = sqrt(Σμ)#Array(cholesky(Σμ))
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:nx]
    return Xs
end

sample_mean(x1)
sample_covariance(x1,w=zeros(2))
resample(x1,w=1.0e-8*ones(2))




A = Diagonal(2.0*ones(2))
vecA = vec(A)
@time sqrtA = sqrt(A)
In = Diagonal(ones(2))

inv(kron(sqrt(A)',In) + kron(In,sqrt(A)))

using FiniteDiff

function mat_sqrt(x)
    vec(sqrt(Array(reshape(x,2,2))))
end

sqrt(reshape(vecA,2,2))
mat_sqrt(vecA)

ForwardDiff.jacobian(mat_sqrt,vecA)
@time real.(FiniteDiff.finite_difference_jacobian(mat_sqrt,vecA))
