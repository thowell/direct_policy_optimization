using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("ipopt.jl")
include("control.jl")
include("integration.jl")

mutable struct Cartpole{T}
    mc::T # mass of the cart in kg (10)
    mp::T # mass of the pole (point mass at the end) in kg
    l::T  # length of the pole in m
    g::T  # gravity m/s^2
end

function dyn_c(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[2]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,1:2) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

Δt = 0.1
model = Cartpole(1.0,0.2,0.5,9.81)
n, m = 4,1

# Pendulum discrete-time dynamics (midpoint)
function dynamics(x,u,Δt)
    x + Δt*dyn_c(model,x + 0.5*Δt*dyn_c(model,x,u),u)
end

# horizon
T = 100
Δt = 0.01

# objective
Q = [t!=T ? Diagonal(1.0*@SVector [1.0,1.0,1.0,1.0]) : Diagonal(1.0*@SVector ones(n)) for t = 1:T]
R = [Diagonal(1.0e-1*@SVector ones(m)) for t = 1:T-1]
x1 = zeros(n)
xT = [0.0;π;0.0;0.0]

x_idx = [(t-1)*(n+m) .+ (1:n) for t = 1:T]
u_idx = [(t-1)*(n+m) + n .+ (1:m) for t = 1:T-1]

n_nlp = n*T + m*(T-1)
m_nlp = n*(T+1)

z0 = 1.0e-5*randn(n_nlp)

function obj(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        # s += (x-xT)'*Q[t]*(x-xT) + u'*R[t]*u
        s += x'*Q[t]*x + u'*R[t]*u

    end
    x = z[x_idx[T]]
    s += (x-xT)'*Q[T]*(x-xT)

    return s
end

obj(z0)

# Constraints
function con!(c,z)
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        x⁺ = z[x_idx[t+1]]
        c[(t-1)*n .+ (1:n)] = x⁺ - dynamics(x,u,Δt)
    end
    c[(T-1)*n .+ (1:n)] = z[x_idx[1]] - x1
    c[T*n .+ (1:n)] = z[x_idx[T]] - xT
    return c
end

c0 = zeros(m_nlp)
con!(c0,z0)

# NLP problem
prob = Problem(n_nlp,m_nlp,obj,con!,true)

# Solve
z_sol = solve(z0,prob)

# get nominal trajectories
z_nom = [z_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
u_nom = [z_sol[(t-1)*(n+m)+n .+ (1:m)] for t = 1:T-1]
plt = plot(hcat(z_nom...)',width=2.0)
plt = plot(hcat(u_nom...)',xlabel="t",ylabel="control",width=2.0)

## TVLQR controller
A = []
B = []
for t = 1:T-1
    x = z_nom[t]
    u = u_nom[t]
    fx(z) = dynamics(z,u,Δt)
    fu(z) = dynamics(x,z,Δt)

    push!(A,ForwardDiff.jacobian(fx,x))
    push!(B,ForwardDiff.jacobian(fu,u))
end

Q = [t < T ? Diagonal(1.0*@SVector [10.0,10.0,10.0,10.0]) : Diagonal([100.0;100.0;100.0;100.0]) for t = 1:T]
R = [Diagonal(1.0*@SVector ones(m)) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

## Unscented controller
β = 10.0
N_sample = 2*(n+m)
K_ukf = []
let K_ukf=K_ukf, H=Q[T]
    tmp = zeros(n+m,n+m)
    z_tmp = zeros(n+m)

    for t = T:-1:2
        println("t: $t")
        tmp[1:n,1:n] = H
        tmp[n .+ (1:m),n .+ (1:m)] = R[t-1]
        L = cholesky(Hermitian(inv(tmp))).U

        z_tmp[1:n] = z_nom[t]
        z_tmp[n .+ (1:m)] = u_nom[t-1]

        z_sample = [z_tmp + β*L[:,i] for i = 1:(n+m)]
        z_sample = [z_sample...,[z_tmp - β*L[:,i] for i = 1:(n+m)]...]

        z_sample_prev = [[dynamics(zs[1:n],zs[n .+ (1:m)],-Δt);zs[n .+ (1:m)]] for zs in z_sample]

        z_tmp[1:n] = z_nom[t-1]
        M = 1/(2*β^2)*sum([(zsp - z_tmp)*(zsp - z_tmp)' for zsp in z_sample_prev])

        P = inv(M)
        P[1:n,1:n] += Q[t-1]

        A = P[1:n,1:n]
        C = P[n .+ (1:m),1:n]
        B = P[n .+ (1:m), n .+ (1:m)]

        K = -(B + 1.0e-16*I)\C

        push!(K_ukf,-K)

        H = A + K'*B*K + K'*C + C'*K
        # H = Hermitian(0.5*(H + H'))
        # println(H)
    end
end

# simulate controllers
T_sim = 10*T
μ = zeros(n)
Σ = Diagonal(1.0e-12*rand(n))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)
z0_sim = copy(x1)

z_nom_sim, u_nom_sim = nominal_trajectories(z_nom,u_nom,T_sim,Δt)
plt = plot(hcat(z_nom_sim...)',color=:red,label=["ref." "" "" ""],width=2.0)
z_tvlqr, u_tvlqr = simulate_linear_controller(K,z_nom,u_nom,T_sim,Δt,z0_sim,w)
plt = plot!(hcat(z_tvlqr...)',color=:purple,label=["tvlqr" "" "" ""],width=2.0)
z_sample, u_sample = simulate_linear_controller(K_ukf,z_nom,u_nom,T_sim,Δt,z0_sim,w)
plt = plot!(hcat(z_sample...)',color=:orange,label=["unscented" "" "" ""],width=2.0)
