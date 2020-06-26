using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("cartpole.jl")
include("moi.jl")
include("control.jl")
include("integration.jl")

## Optimize nominal trajectory

# horizon
T = 100
Δt = 0.1

z_ref = [@SVector zeros(n) for t = 1:T-1]
push!(z_ref,@SVector [0.0, pi, 0.0, 0.0])
u_ref = [zeros(m) for t = 1:T-1]

# objective
Q = Diagonal(1.0*@SVector [1.0,1.0,1.0,1.0])
Qf = Diagonal(1.0*@SVector ones(n))
R = Diagonal(1.0e-1*@SVector ones(m))

# NLP dimensions
n_nlp = n*T + m*(T-1)
m_nlp = n*T

# NLP problem
prob = problem_goal(n_nlp,m_nlp,z_ref,u_ref,T,n,m,Q,Qf,R,model,rk3,Δt,false,false)

# NLP initialization
x0 = zeros(n_nlp)
for t = 1:T-1
    x0[(t-1)*(n+m) .+ (1:(n+m))] = [z_ref[t];u_ref[t]]
end
x0[(T-1)*(n+m) .+ (1:(n))] = z_ref[T]

# solve
x_sol = solve_ipopt(x0,prob)

# get nominal trajectories
z_nom = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
x_nom = [x_sol[(t-1)*(n+m) .+ (1:n)][1] for t = 1:T]
y_nom = [x_sol[(t-1)*(n+m) .+ (1:n)][2] for t = 1:T]
u_nom = [x_sol[(t-1)*(n+m)+n .+ (1:m)] for t = 1:T-1]
plt = plot(hcat(z_nom...)',xlabel="x",ylabel="y",width=2.0)
plt = plot(hcat(u_nom...)',xlabel="t",ylabel="control",width=2.0)

## TVLQR controller
Q = Diagonal(1.0*@SVector [10.0,10.0,1.0,1.0])
Qf = Diagonal(100.0*@SVector ones(n))
R = Diagonal(1.0*@SVector ones(m))
K_tvlqr, P, A, B = tvlqr(z_nom,u_nom,model,Q,R,Qf,Δt)

## Unscented controller

β = 2.0
N_sample = 2*(n+m)
K_ukf = []

let K_ukf=K_ukf, H=Qf
    tmp = zeros(n+m,n+m)
    z_tmp = zeros(n+m)

    for t = T:-1:2
        println("t: $t")
        tmp[1:n,1:n] = H
        tmp[n .+ (1:m),n .+ (1:m)] = R
        L = Array(cholesky(Hermitian(inv(tmp))))

        z_tmp[1:n] = z_nom[t]
        z_tmp[n .+ (1:m)] = u_nom[t-1]

        z_sample = [z_tmp + β*L[:,i] for i = 1:(n+m)]
        z_sample = [z_sample...,[z_tmp - β*L[:,i] for i = 1:(n+m)]...]

        z_sample_prev = [[midpoint(model,zs[1:n],zs[n .+ (1:m)],-Δt);zs[n .+ (1:m)]] for zs in z_sample]

        M = 0.5/(β^2)*sum([(zsp - z_tmp)*(zsp - z_tmp)' for zsp in z_sample_prev])

        P = inv(M)
        P[1:n,1:n] += Q

        A = P[1:n,1:n]
        C = P[n .+ (1:m),1:n]
        B = P[n .+ (1:m), n .+ (1:m)]

        K = B\C

        push!(K_ukf,K)

        H = A + K'*B*K - K'*C - C'*K
        H = 0.5*(H + H')
    end
end

K_ukf
##
# Simulate TVLQR controller
N_sim = 10
T_sim = 100
ρ = 1.0e-16
W = Distributions.MvNormal(zeros(n),Diagonal(ρ*ones(n)))
_z_nom, _u_nom = nominal_trajectories(z_nom,u_nom,T_sim,Δt)

plt = plot(title="TVLQR tracking (Σ = $ρ I)",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_tvlqr,z_nom,u_nom,T_sim,Δt,z_nom[1],W,rk3)
    plt = plot!(hcat(z_sim...)',color=:green,label="")
end
plt = plot!(hcat(_z_nom...)',color=:orange,legend=:bottomleft,label=["ref." "" "" ""],width=2.0,linetype=:steppost)
display(plt)

# Simulate unscented TVLQR controller
plt = plot(title="Unscented TVLQR tracking (Σ = $ρ I)",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_ukf,z_nom,u_nom,T_sim,Δt,z_nom[1],W,rk3)
    plt = plot!(hcat(z_sim...)',color=:purple,label="")
end
plt = plot!(hcat(_z_nom...)',color=:orange,legend=:bottomleft,label=["ref." "" "" ""],width=2.0,linetype=:steppost)
display(plt)
