using LinearAlgebra, ForwardDiff, Distributions, Plots
include("ipopt.jl")

function fastsqrt(A)
    #FASTSQRT computes the square root of a matrix A with Denman-Beavers iteration

    #S = sqrtm(A);
    Ep = 1e-8*Matrix(I,size(A))

    if count(diag(A) .> 0.0) != size(A,1)
        S = diagm(sqrt.(diag(A)));
        return S
    end

    In = Matrix(1.0*I,size(A));
    S = A;
    T = Matrix(1.0*I,size(A));

    T = .5*(T + inv(S+Ep));
    S = .5*(S+In);
    for k = 1:7
        Snew = .5*(S + inv(T+Ep));
        T = .5*(T + inv(S+Ep));
        S = Snew;
    end
    return S
end

# continuous-time dynamics
n = 2
m = 1
Ac = [0.0 1.0; 0.0 0.0]
Bc = [0.0; 1.0]

# discrete-time dynamics
Δt = 0.1
D = exp(Δt*[Ac Bc; zeros(1,n+m)])
A = D[1:n,1:n]
B = D[1:n,n .+ (1:m)]

function dynamics(x,u,Δt)
    Δt = 0.1
    D = exp(Δt*[Ac Bc; zeros(1,n+m)])
    A = D[1:n,1:n]
    B = D[1:n,n .+ (1:m)]

    return A*x + B*u
end

# TVLQR solution
T = 10
Q = Matrix(1.0*I,n,n)
R = Matrix(0.1*I,m,m)

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q
for t = T-1:-1:1
    K[t] = (R + B'*P[t+1]*B)\(B'*P[t+1]*A)
    P[t] = Q + K[t]'*R*K[t] + (A-B*K[t])'*P[t+1]*(A-B*K[t])
end

# number of samples
β = 1.0
x11 = β*[1.0; 1.0]
x12 = β*[1.0; -1.0]
x13 = β*[-1.0; 1.0]
x14 = β*[-1.0; -1.0]

x1 = [x11,x12,x13,x14]

N = length(x1)

n_nlp = m*n*(T-1) + N*(n*(T-1) + m*(T-1)) + N*(n*(T-1) + m*(T-1))
m_nlp = N*(n*(T-1) + m*(T-1)) + 2*N*(n*(T-1) + m*(T-1))

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_sx = [[(T-1)*(m*n) + N*(n*(T-1) + m*(T-1)) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_su = [[(T-1)*(m*n) + + N*(n*(T-1) + m*(T-1)) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_sxp = [[N*(n*(T-1) + m*(T-1)) + (i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_sxn = [[N*(n*(T-1) + m*(T-1)) + N*(n*(T-1)) + (i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]

idx_con_sup = [[N*(n*(T-1) + m*(T-1)) + N*(n*(T-1)) + N*(n*(T-1)) + (i-1)*(m*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]
idx_con_sun = [[N*(n*(T-1) + m*(T-1)) + N*(n*(T-1)) + N*(n*(T-1)) + N*(m*(T-1)) + (i-1)*(m*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

function resample(X; β=1.0,w=1.0)
    N = length(X)
    n = length(X[1])

    xμ = sum(X)./N
    Σμ = (0.5/(β^2))*sum([(X[i] - xμ)*(X[i] - xμ)' for i = 1:N]) + w*I
    # cols = cholesky(Σμ).U
    cols = fastsqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]

    return Xs
end

function sample_dynamics(X,U; β=1.0,w=1.0)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,A*X[i] + B*U[i])
    end
    # return X⁺
    Xs⁺ = resample(X⁺,β=β,w=w)
    return Xs⁺
end

function obj(z)
    J = 0
    for t = 1:T-1
        for i = 1:N
            # x = view(z,idx_x[i][t])
            # u = view(z,idx_u[i][t])
            # J += x'*Q*x + u'*R*u
            sx = view(z,idx_sx[i][t])
            su = view(z,idx_su[i][t])
            J += sum(sx) + sum(su)
        end
    end
    return J
end

function con!(c,z)
    β = 1.0
    w = 1.0e-1
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_dynamics(xs,u,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),m,n)
        sx = [view(z,idx_sx[i][t]) for i = 1:N]
        su = [view(z,idx_su[i][t]) for i = 1:N]

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*xs[i]
            c[idx_con_sxp[i][t]] = sqrt(Q)*xs⁺[i] - sx[i]
            c[idx_con_sxn[i][t]] = -sqrt(Q)*xs⁺[i] - sx[i]
            c[idx_con_sup[i][t]] = sqrt(R)*u[i] - su[i]
            c[idx_con_sun[i][t]] = -sqrt(R)*u[i] - su[i]
        end
    end
    return c
end

c0 = rand(m_nlp)
con!(c0,ones(n_nlp))
idx_ineq = vcat(vcat(idx_con_sxp...,idx_con_sxn...,idx_con_sup...,idx_con_sun...)...)

prob = Problem(n_nlp,m_nlp,obj,con!,true,idx_ineq=idx_ineq)

z0 = rand(n_nlp)
z_sol = solve(z0,prob)

K_sample = [reshape(z_sol[idx_k[t]],m,n) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("solution error: $(sum(K_error)/N)")

plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,width=2.0,label="β=$β",legend=:bottom,title="Gain matrix error")
