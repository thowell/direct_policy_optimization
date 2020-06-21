using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("ipopt_nominal_trajectory.jl")
include("ipopt_optimize_controller.jl")

# parameterized model
mutable struct DoubleIntegrator2D
    # z = (x,y,ẋ,ẏ)
    mx
    my
end

# double integrator dynamics (2D)
function dynamics(model::DoubleIntegrator2D,z,u)
    @SVector [z[3], z[4], u[1]/model.mx, u[2]/model.my]
end

n = 4 # number of states
m = 2 # number of controls

model = DoubleIntegrator2D(1.0,1.0) # nominal model
z0 = zeros(4) # initial condition
u0 = zeros(2) # initial controls

# discrete dynamics (midpoint)
Δt = 0.05
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end

# horizon
T = 40

# reference position trajectory
x_ref = sin.(range(0.0,stop=pi,length=T))
y_ref = range(0.0,stop=pi,length=T)
z_ref = [[x_ref[t];y_ref[t];0.0;0.0] for t = 1:T]

# reference control trajectory
u_ref = [u0 for t = 1:T-1]

# optimize nominal trajectories
z0 = [x_ref[1];y_ref[1];0.0;0.0]

# objective
Q = Diagonal(@SVector[10.0,10.0,1.0e-1,1.0e-1])
Qf = Diagonal(@SVector[100.0,100.0,1.0,1.0])
R = 1.0e-1*sparse(I,m,m)

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
x_nom = zeros(T)
y_nom = zeros(T)

z_nom = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
u_nom = [x_sol[(t-1)*(n+m)+n .+ (1:m)] for t = 1:T-1]
for t = 1:T
    x_nom[t] = x_sol[(t-1)*(n+m) .+ (1:n)][1]
    y_nom[t] = x_sol[(t-1)*(n+m) .+ (1:n)][2]
end

plt = plot(x_ref,y_ref,title="reference trajectory",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
plt = scatter!(x_nom,y_nom,color=:orange,label="nominal")

# tvlqr controller
A = []
B = []
for t = 1:T-1
    fz(z) = midpoint(model,z,u_nom[t],Δt)
    fu(u) = midpoint(model,z_nom[t],u,Δt)
    push!(A,ForwardDiff.jacobian(fz,z_nom[t]))
    push!(B,ForwardDiff.jacobian(fu,u_nom[t]))
end
P = []
push!(P,Qf)
K = []

for t = T-1:-1:1
    push!(K,(R + B[t]'*P[end]*B[t])\(B[t]'*P[end]*A[t]))
    push!(P,A[t]'*P[end]*A[t] - (A[t]'*P[end]*B[t])*K[end] + Q)
end

# optimize controller
N = 2n # number of samples

# initial condition distribution
r_sample = 0.1 # initial condition sample radius

mv = Distributions.MvNormal(z_nom[1],r_sample*Diagonal(ones(n))) # multi-variate Gaussian
Z0 = rand(mv,N) # initial condition samples
z0 = [Z0[:,i] for i = 1:N]

function plot_circle(x0,r;N=100,label="",plt=plot())
    θ = range(0,stop=2pi,length=N)
    x = r.*cos.(θ)
    y = r.*sin.(θ)
    plt = plot!(x,y,width=2.0,color=:magenta,label=label)
    return plt
end


# NLP dimensions
n_nlp_ctrl = n*N*T + (m*n)*(T-1)
m_nlp_ctrl = n*N*T

# NLP problem
prob_ctrl = ProblemCtrl(n_nlp_ctrl,m_nlp_ctrl,z_nom,u_nom,T,n,m,Q,Qf,R,A,B,model,Δt,N,z0,false)

# NLP initialization
x0_ctrl = zeros(n_nlp_ctrl)

for t = 1:T
    for i = 1:N
        x0_ctrl[(t-1)*(n*N + m*n)+(i-1)*n .+ (1:n)] = z_nom[t]
    end
end

# MOI.eval_objective(prob_ctrl,x0_ctrl)
# grad_f = ones(n_nlp_ctrl)
# MOI.eval_objective_gradient(prob_ctrl,grad_f,x0_ctrl)
# grad_f
# g = ones(m_nlp_ctrl)
# MOI.eval_constraint(prob_ctrl,g,x0_ctrl)
# g
# jac = ones(m_nlp_ctrl*n_nlp_ctrl)
# MOI.eval_constraint_jacobian(prob_ctrl,jac,x0_ctrl)

# solve
x_sol = solve_ipopt(x0_ctrl,prob_ctrl)

# optimized controller
K_ctrl = [reshape(x_sol[(t-1)*(n*N+m*n)+n*N .+ (1:m*n)],m,n) for t = 1:T-1]

# optimized trajectories
x_ctrl = [zeros(T) for i = 1:N]
y_ctrl = [zeros(T) for i = 1:N]
for t = 1:T
    for i = 1:N
        x_ctrl[i][t] = x_sol[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)][1]
        y_ctrl[i][t] = x_sol[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)][2]
    end
end

plt = plot(x_ref,y_ref,title="reference trajectory",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
plt = scatter!(x_nom,y_nom,color=:orange,label="nominal")
for i = 1:N
    plt = scatter!(x_ctrl[i],y_ctrl[i],label="z0_$i")
end
display(plt)

# test controllers
N_sim = 1000
Z0_sim = rand(mv,N_sim)
x0_sim = [Z0_sim[1,k] for k = 1:N_sim]
y0_sim = [Z0_sim[1,k] for k = 1:N_sim]

# tvlqr test
times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
plt = plot(title="TVLQR controller",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
for k = 1:N_sim
    z_tvlqr_rollout = [Z0_sim[:,k]]
    u_tvlqr = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        w = randn(n)*0.0
        z = z_tvlqr_rollout[end] + 1.0*w
        u = u_nom[k] - K[k]*(z - z_nom[k])
        push!(z_tvlqr_rollout,midpoint(model,z,u,dt_sim))
        push!(u_tvlqr,u)
    end

    z_tvlqr_rollout
    x_tvlqr_rollout = [z_tvlqr_rollout[t][1] for t = 1:T_sim]
    y_tvlqr_rollout = [z_tvlqr_rollout[t][2] for t = 1:T_sim]

    plt = plot!(x_tvlqr_rollout,y_tvlqr_rollout,color=:purple,label="",width=2.0)
end
display(plt)
plt = scatter!(x_nom,y_nom,color=:orange,label="nominal")
plt = scatter!([x_nom[1]],[y_nom[1]],color=:red,marker=:square,label="start",width=2.0)
plt = scatter!([x_nom[end]],[y_nom[end]],color=:green,marker=:hex,label="goal",width=2.0)
# plt = plot_circle(z_nom[1][1:2],r_sample,N=100,label="Σ=0.1",plt=plt)
savefig(plt,joinpath(pwd(),"results_6_21_2020/tvlqr_ctrl.png"))

# sampled controller
times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)

plt = plot(title="optimized linear controller",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
for k = 1:N_sim
    z_ctrl_rollout = [Z0_sim[:,k]]
    u_ctrl = []

    for tt = 1:T_sim-1
       t = t_sim[tt]
       k = searchsortedlast(times,t)
       w = randn(n)*0.0
       z = z_ctrl_rollout[end] + 1.0*w
       u = u_nom[k] - K_ctrl[k]*(z - z_nom[k])
       push!(z_ctrl_rollout,midpoint(model,z,u,dt_sim))
       push!(u_ctrl,u)
    end

    z_ctrl_rollout
    x_ctrl_rollout = [z_ctrl_rollout[t][1] for t = 1:T_sim]
    y_ctrl_rollout = [z_ctrl_rollout[t][2] for t = 1:T_sim]

    plt = plot!(x_ctrl_rollout,y_ctrl_rollout,color=:cyan,label="",width=2.0)
end

display(plt)
plt = scatter!()
plt = scatter!(x_nom,y_nom,color=:orange,label="nominal")
plt = scatter!([x_nom[1]],[y_nom[1]],color=:red,marker=:square,label="start",width=2.0)
plt = scatter!([x_nom[end]],[y_nom[end]],color=:green,marker=:hex,label="goal",width=2.0)
savefig(plt,joinpath(pwd(),"results_6_21_2020/opt_ctrl.png"))
