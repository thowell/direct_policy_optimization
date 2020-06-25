using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("double_integrator_2D.jl")
include("moi.jl")
include("control.jl")

## Optimize nominal trajectory

# horizon
T = 50

# reference position trajectory
x_ref = range(0.0,stop=1.0,length=T)
y_ref = sin.(range(0.0,stop=2pi,length=T))
z_ref = [[x_ref[t];y_ref[t];0.0;0.0] for t = 1:T]

plot(x_ref,y_ref)
# reference control trajectory
u_ref = [zeros(m) for t = 1:T-1]

# objective
Q = Diagonal(@SVector[10.0,10.0,0.01,0.01])
Qf = Diagonal(@SVector[100.0,100.0,0.01,0.01])
R = 1.0e-2*sparse(I,m,m)

# NLP dimensions
n_nlp = n*T + m*(T-1)
m_nlp = n*T

# NLP problem
prob = Problem(n_nlp,m_nlp,z_ref,u_ref,T,n,m,Q,Qf,R,model,Δt,false)

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

plt = plot(x_ref,y_ref,color=:black,label="nominal")
plt = plot!(x_nom,y_nom,xlabel="x",ylabel="y",color=:orange,label="opt.",width=2.0)

## TVLQR controller
K_tvlqr, P, A, B = tvlqr(z_nom,u_nom,model,Δt)

## Unscented controller
β = 1.0
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
    end
end

K_ukf

##
# Simulate TVLQR controller
N_sim = 100
W = Distributions.MvNormal(zeros(n),Diagonal(1.0e-4*ones(n)))

plt = plot(title="TVLQR tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_tvlqr,z_nom,u_nom,1000,Δt,z_nom[1],W)
    x_sim = [z[1] for z in z_sim]
    y_sim = [z[2] for z in z_sim]

    plt = plot!(x_sim,y_sim,color=:green,label="")
end
plt = plot!(x_nom,y_nom,color=:orange,label="ref.",width=2.0)
display(plt)

# Simulate UKF controller
plt = plot(title="UKF tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_ukf,z_nom,u_nom,1000,Δt,z_nom[1],W)
    x_sim = [z[1] for z in z_sim]
    y_sim = [z[2] for z in z_sim]

    plt = plot!(x_sim,y_sim,color=:purple,label="")
end
plt = plot!(x_nom,y_nom,color=:orange,label="ref.",width=2.0)
display(plt)
