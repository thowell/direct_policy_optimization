function matrix_sqrt(A)
    e = eigen(A)

    for (i,ee) in enumerate(e.values)
        if ee < 0.0
            e.values[i] = 1.0e-3
        end
    end
    return e.vectors*Diagonal(sqrt.(e.values))*inv(e.vectors)
end

function matrix_sqrt_axis(A)
    e = eigen(A)

    for (i,ee) in enumerate(e.values)
        if ee < 0.0
            e.values[i] = 1.0e-3
        end
    end
    return e.vectors*Diagonal(sqrt.(e.values))
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
    for k = 1:10
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

function nominal_jacobians(model,X,U;
        free_time=false,Δt=-1.0)
    A = []
    B = []
    w = zeros(model.nw)
    for t = 1:T-1
        x = X[t]
        u = U[t]
        # x⁺ = X[t+1]
        #
        # fx(z) = (free_time
        #          ? midpoint_implicit(model,x⁺,z,u[1:end-1],u[end],w)
        #          : midpoint_implicit(model,x⁺,z,u,Δt,w)
        #          )
        # fu(z) = (free_time
        #          ? midpoint_implicit(model,x⁺,x,z[1:end-1],z[end],w)
        #          : midpoint_implicit(model,x⁺,x,z,Δt,w)
        #          )
        # fx⁺(z) = (free_time
        #           ? midpoint_implicit(model,z,x,u[1:end-1],u[end],w)
        #           : midpoint_implicit(model,z,x,u,Δt,w)
        #           )
        #
        # A⁺ = ForwardDiff.jacobian(fx⁺,x⁺)
        # push!(A,-A⁺\ForwardDiff.jacobian(fx,x))
        # push!(B,-A⁺\ForwardDiff.jacobian(fu,u)[:,free_time ? (1:end-1) : (1:end)])

        fx(z) = (free_time
                 ? midpoint(model,z,u[1:end-1],u[end],w)
                 : midpoint(model,z,u,Δt,w)
                 )
        fu(z) = (free_time
                 ? midpoint(model,x,z[1:end-1],z[end],w)
                 : midpoint(model,x,z,Δt,w)
                 )

        push!(A,ForwardDiff.jacobian(fx,x))
        push!(B,ForwardDiff.jacobian(fu,u)[:,free_time ? (1:end-1) : (1:end)])
    end
    return A, B
end

function TVLQR_gains(model,X,U,Q_lqr,R_lqr;
        free_time=false,Δt=-1.0)

    A,B = nominal_jacobians(model,X,U,
            free_time=free_time,Δt=Δt)

    K = TVLQR(A,B,Q_lqr,R_lqr)
end

function sample_mean(X)
    N = length(X)
    nx = length(X[1])

    xμ = sum(X)./N
end

function sample_covariance(X; β=1.0,w=1.0e-8*ones(length(X[1])))
    N = length(X)
    xμ = sample_mean(X)
    Σμ = (0.5/(β^2))*sum([(X[i] - xμ)*(X[i] - xμ)' for i = 1:N]) + Diagonal(w)
end

function resample(X; β=1.0,w=ones(length(X[1])))
    N = length(X)
    nx = length(X[1])
    xμ = sample_mean(X)
    Σμ = sample_covariance(X,β=β,w=w)
    cols = sqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:nx]
    return Xs
end


function resample_fastsqrt(X; β=1.0,w=ones(length(X[1])))
    N = length(X)
    nx = length(X[1])
    xμ = sample_mean(X)
    Σμ = sample_covariance(X,β=β,w=w)
    cols = fastsqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:nx]
    return Xs
end

function resample_vec(X,n,N,k; β=1.0,w=1.0)
    xμ = sum([X[(i-1)*n .+ (1:n)] for i = 1:N])./N
    Σμ = (0.5/(β^2))*sum([(X[(i-1)*n .+ (1:n)] - xμ)*(X[(i-1)*n .+ (1:n)] - xμ)' for i = 1:N]) + Diagonal(w)
    cols = sqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]
    return Xs[k]
end

function sample_dynamics_linear(X,U,A,B; β=1.0,w=1.0,fast_sqrt=false)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,A*X[i] + B*U[i])
    end
    Xs⁺ = fast_sqrt ? resample_fastsqrt(X⁺,β=β,w=w) : resample(X⁺,β=β,w=w)
    return Xs⁺
end

function sample_dynamics(model,X,U,Δt,t; β=1.0,w=1.0,fast_sqrt=false)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,discrete_dynamics(model,X[i],U[i],Δt,t))
    end
    Xs⁺ = fast_sqrt ? resample_fastsqrt(X⁺,β=β,w=w) : resample(X⁺,β=β,w=w)
    return Xs⁺
end

function n_tri(n)
    convert(Int,(n^2 + n)/2)
end

function sigma_points(μ,L,W,β)
    n = length(μ)
    d = size(W,1)
    w0 = zeros(d)
    L_mat = vec_to_lt(L)

    x = cat([μ + s*β*L_mat[:,i] for s in [-1.0,1.0] for i = 1:n],[μ for i = 1:2d],dims=(1,1))
    w = cat([w0 for i = 1:2n],[s*β*W[:,i] for s in [-1.0,1.0] for i = 1:d],dims=(1,1))
    return x,w
end

function state_sigma_points(μ,L,β)
    n = length(μ)
    L_mat = vec_to_lt(L)

    x = [μ + s*β*L_mat[:,i] for s in [-1.0,1.0] for i = 1:n]
    return x
end



function dynamics_sample_mean(x_nom,u_nom,μ,L,K,W,β,Δt,t,sample_model,N)
    x, w = sigma_points(μ,L,W,β)
    s = [discrete_dynamics(sample_model,x[i],
            policy(sample_model,K,x[i],x_nom,u_nom),Δt,w[i],t) for i = 1:N]
    μ⁺ = sample_mean(s)
    μ⁺
end

function dynamics_sample_L(x_nom,u_nom,μ,L,K,W,β,Δt,t,sample_model,N)
    x,w = sigma_points(μ,L,W,β)
    s = [discrete_dynamics(sample_model,x[i],
            policy(sample_model,K,x[i],x_nom,u_nom),Δt,w[i],t) for i = 1:N]
    P⁺ = Array(sample_covariance(s))
    L⁺ = lt_to_vec(cholesky(P⁺).L)
    L⁺
    # u_nom
end

function sample_cost(x_nom,u_nom,μ,L,K,β,N,sample_model,Q,R)
    x = state_sigma_points(μ,L,β)
    u = [policy(sample_model,K,x[i],x_nom,u_nom) for i = 1:N]
    J = 0.0
    for i = 1:N
        J += (x[i] - x_nom)'*Q*(x[i] - x_nom)
        J += (u[i] - u_nom)'*R*(u[i] - u_nom)
    end
    return J
end

function sample_cost_terminal(x_nom,μ,L,β,N,sample_model,Q)
    x = state_sigma_points(μ,L,β)
    J = 0.0
    for i = 1:N
        J += (x[i] - x_nom)'*Q*(x[i] - x_nom)
    end
    return J
end

function lt_to_vec(L)
    L[tril!(trues(size(L)), 0)]
end

function vec_to_lt(v)
    n = length(v)
    s = round(Int,(sqrt(8n+1)-1)/2)
    x = zeros(eltype(v),s,s)
    shift = 1
    for j = 1:s
        for i = j:s
            x[i,j] = v[shift]
            shift += 1
        end
    end
    x
end

function sample_control_bounds!(c,μ,L,K,x_nom,u_nom,β,N,sample_model,ul,uu)
    x = state_sigma_points(μ,L,β)
    u = [policy(sample_model,K,x[i],x_nom,u_nom) for i = 1:N]
    for i = 1:N
        c[(i-1)*(2*sample_model.nu) .+ (1:sample_model.nu)] = uu - u[i]
        c[(i-1)*(2*sample_model.nu) + sample_model.nu .+ (1:sample_model.nu)] = u[i] - ul
    end
    return nothing
end

function sample_state_bounds!(c,μ,L,β,N,sample_model,xl,xu)
    x = state_sigma_points(μ,L,β)
    for i = 1:N
        c[(i-1)*(2*sample_model.nx) .+ (1:sample_model.nx)] = xu - x[i]
        c[(i-1)*(2*sample_model.nx) + sample_model.nx .+ (1:sample_model.nx)] = x[i] - xl
    end
    return nothing
end

function sample_stage!(c,μ,L,K,x_nom,u_nom,β,N,sample_model,m_stage,t)
    x = state_sigma_points(μ,L,β)
    u = [policy(sample_model,K,x[i],x_nom,u_nom) for i = 1:N]
    for i = 1:N
        c_stage!(view(c,(i-1)*m_stage .+ (1:m_stage)),x[i],u[i],t,sample_model)
    end
    return nothing
end
