using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("cartpole.jl")
include("moi.jl")
include("control.jl")
include("integration.jl")

## Optimize nominal trajectory

# horizon
T = 50
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
R = Diagonal(1.0e-1*@SVector ones(m))
## TVLQR controller
K_tvlqr, P, A, B = tvlqr(z_nom,u_nom,model,Q,R,Qf,Δt)

# Simulate controller with mass uncertainty
T_sim = 2T
_z_nom, _u_nom = nominal_trajectories(z_nom,u_nom,T_sim,Δt)
W = Distributions.MvNormal(zeros(n),Diagonal(0.0e-8*ones(n)))

models = [model,Cartpole(1.5,0.2,0.5,9.81)]#,Cartpole(5.0,1.0,0.5,9.81),Cartpole(3.0,0.1,0.5,9.81)]#,DoubleIntegrator2D(10.0,1.0),DoubleIntegrator2D(2.0,1.0),DoubleIntegrator2D(3.0,5.0)]

N_sim = length(models)

plt = plot(title="TVLQR tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_tvlqr,z_nom,u_nom,T_sim,Δt,z_nom[1],W,models[k],rk3)
    plt = plot!(hcat(z_sim...)',color=(k==1 ? :green : :cyan),label=(k==1 ? "nominal" : ""))
end
plt = plot!(hcat(_z_nom...)',color=:orange,legend=:bottomleft,label="opt.",width=2.0)
display(plt)



## Sample-based controller
β = 1.0
sigma_models = [Cartpole(1.0+β,0.2,0.5,9.81),Cartpole(1.0-β,0.2,0.5,9.81)]#,Cartpole(1.0,0.2+β,0.5,9.81),Cartpole(1.0,0.2-β,0.5,9.81)]
N = length(sigma_models)
# A = []
# B = []
# for i = 1:N
#     K_tvlqr, P, _A, _B = tvlqr(z_nom,u_nom,sigma_models[i],Q,R,Qf,Δt)
#     push!(A,_A)
#     push!(B,_B)
# end
# A
# n_nlp_ctrl = n*N*T + (m*n)*(T-1)
# m_nlp_ctrl = n*N*T

# NLP problem
prob_ctrl = ProblemCtrl(n_nlp_ctrl,m_nlp_ctrl,z_nom,u_nom,T,n,m,Q,Qf,R,A,B,sigma_models,Δt,N,sigma_models,false)

# NLP initialization
x0_ctrl = zeros(n_nlp_ctrl)

for t = 1:T
    for i = 1:N
        x0_ctrl[(t-1)*(n*N + m*n)+(i-1)*n .+ (1:n)] = z_nom[t] .+ 1.0e-3
    end
    if t < T
        x0_ctrl[(t-1)*(n*N + m*n)+n*N .+ (1:m*n)] = 0.0*vec(K_tvlqr[t])
    end
end

# solve
x_sol = solve_ipopt(x0_ctrl,prob_ctrl)

MOI.eval_objective(prob_ctrl,x0_ctrl)
MOI.eval_objective_gradient(prob_ctrl,zeros(n_nlp_ctrl),x0_ctrl)
MOI.eval_constraint(prob_ctrl,zeros(m_nlp_ctrl),x0_ctrl)
MOI.eval_constraint_jacobian(prob_ctrl,zeros(m_nlp_ctrl*n_nlp_ctrl),x0_ctrl)


# sample-based K
K_sample = [reshape(x_sol[(t-1)*(n*N + m*n)+n*N .+ (1:m*n)],m,n) for t = 1:T-1]

# Simulate controller

plt = plot(title="Sample-based controller tracking",xlabel="x",ylabel="y")
for k = 1:N_sim
    z_sim, u_sim = simulate_linear_controller(K_sample,z_nom,u_nom,T_sim,Δt,z_nom[1],W,models[k],rk3)
    plt = plot!(hcat(z_sim...)',color=(k==1 ? :green : :cyan),label=(k==1 ? "nominal" : ""))
end
plt = plot!(hcat(_z_nom...)',color=:orange,legend=:bottomleft,label="opt.",width=2.0)
display(plt)

K_sample[end]

K_tvlqr[end]
