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

# Simulate controller
N_sim = 100
W = Distributions.MvNormal(zeros(n),Diagonal(1.0e-5*ones(n)))

plt = plot(title="TVLQR tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_tvlqr,z_nom,u_nom,1000,Δt,z_nom[1],W)
    x_sim = [z[1] for z in z_sim]
    y_sim = [z[2] for z in z_sim]

    plt = plot!(x_sim,y_sim,color=:green,label="")
end
plt = plot!(x_nom,y_nom,color=:orange,label="opt.",width=2.0)
display(plt)

## Sample-based controller
N = 2n

N_dist = [8]
W = [Distributions.MvNormal(zeros(n),Diagonal(10.0^(-i)*ones(n))) for i in N_dist for j in N_dist]
w = [[vec(rand(W[rand(1:length(N_dist))],1)) for t = 1:T] for i = 1:N]

n_nlp_ctrl = n*N*T + (m*n)*(T-1)
m_nlp_ctrl = n*N*T

# NLP problem
prob_ctrl = ProblemCtrl(n_nlp_ctrl,m_nlp_ctrl,z_nom,u_nom,T,n,m,Q,Qf,R,A,B,model,Δt,N,w,false)

# NLP initialization
x0_ctrl = zeros(n_nlp_ctrl)

for t = 1:T
    for i = 1:N
        x0_ctrl[(t-1)*(n*N + m*n)+(i-1)*n .+ (1:n)] = z_nom[t]
    end
    # if t < T
    #     x0_ctrl[(t-1)*(n*N + m*n)+n*N .+ (1:m*n)] = 0.0*vec(K_tvlqr[t]) + 0.0e-3*rand(m*n)
    # end
end

# solve
x_sol = solve_ipopt(x0_ctrl,prob_ctrl)

# sample-based K
K_sample = [reshape(x_sol[(t-1)*(n*N + m*n)+n*N .+ (1:m*n)],m,n) for t = 1:T-1]

println("K_tvlqr: $(vec(K_tvlqr[1]))")
println("K_sample: $(vec(K_sample[1]))")
println("norm(K_tvlqr-K_sample): $(norm(vec(K_tvlqr[1])-vec(K_sample[1])))")

# Simulate controller
N_sim = 100
W = Distributions.MvNormal(zeros(n),Diagonal(1.0e-8*ones(n)))

plt = plot(title="Sample-based controller tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_sample,z_nom,u_nom,1000,Δt,z_nom[1],W)
    x_sim = [z[1] for z in z_sim]
    y_sim = [z[2] for z in z_sim]

    plt = plot!(x_sim,y_sim,color=:green,label="")
end
plt = plot!(x_nom,y_nom,color=:orange,label="opt.",width=2.0)
display(plt)
