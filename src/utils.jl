function matrix_sqrt(A)
    e = eigen(A)

    for (i,ee) in enumerate(e.values)
        if ee < 0.0
            e.values[i] = 1.0e-3
        end
    end
    return e.vectors*Diagonal(sqrt.(e.values))*inv(e.vectors)
end

function fastsqrt(A)
    #FASTSQRT computes the square root of a matrix A with Denman-Beavers iteration
    n = size(A,1)
    Ep = 1e-8*Diagonal(ones(n))

    # if count(diag(A) .> 0.0) != size(A,1)
    #     S = diagm(sqrt.(diag(A)));
    #     return S
    # end

    In = Diagonal(ones(n))
    S = A;
    T = Diagonal(ones(n))

    T = 0.5*(T + inv(S+Ep));
    S = 0.5*(S+In);
    for k = 1:5
        Snew = 0.5*(S + inv(T+Ep));
        T = 0.5*(T + inv(S+Ep));
        S = Snew;
    end
    return S
end

# linear interpolation
function linear_interp(x0,xf,T)
    n = length(x0)
    X = [copy(Array(x0)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xf[i]-x0[i])/(T-1)*(t-1) + x0[i]
        end
    end

    return X
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function row_col_cartesian!(row,col,r,c)
    for i = 1:length(r)
        push!(row,convert(Int,r[i]))
        push!(col,convert(Int,c[i]))
    end
    return row, col
end

function TVLQR(A,B,Q,R)
    P = [zero(A[1]) for t = 1:T]
    K = [zero(B[1]') for t = 1:T-1]
    P[T] = Q[T]
    for t = T-1:-1:1
        K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
        P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
    end
    return K
end

function resample(X; β=1.0,w=1.0)
    N = length(X)
    nx = length(X[1])

    xμ = sum(X)./N
    Σμ = (0.5/(β^2))*sum([(X[i] - xμ)*(X[i] - xμ)' for i = 1:N]) + Diagonal(w)
    cols = fastsqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:nx]

    return Xs
end

function resample_vec(X,n,N,k; β=1.0,w=1.0)
    xμ = sum([X[(i-1)*n .+ (1:n)] for i = 1:N])./N
    Σμ = (0.5/(β^2))*sum([(X[(i-1)*n .+ (1:n)] - xμ)*(X[(i-1)*n .+ (1:n)] - xμ)' for i = 1:N]) + Diagonal(w)
    cols = fastsqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]
    return Xs[k]
end

function sample_dynamics_linear(X,U,A,B; β=1.0,w=1.0)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,A*X[i] + B*U[i])
    end
    Xs⁺ = resample(X⁺,β=β,w=w)
    return Xs⁺
end

function sample_dynamics(model,X,U,Δt; β=1.0,w=1.0)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,discrete_dynamics(model,X[i],U[i],Δt))
    end
    Xs⁺ = resample(X⁺,β=β,w=w)
    return Xs⁺
end
