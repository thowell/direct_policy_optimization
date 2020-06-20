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

model = DoubleIntegrator2D(1.0,1.0)
z0 = zeros(4)
u0 = ones(2)
dynamics(model,z0,u0)

# discrete dynamics (midpoint)
Δt = 0.1
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end
midpoint(model,z0,u0,Δt)

T = 20
x_ref = sin.(range(0.0,stop=pi,length=T))
y_ref = range(0.0,stop=pi,length=T)
z_ref = [[x_ref[t];y_ref[t];0.0;0.0] for t = 1:T]
u0 = zeros(m)
u_ref = [u0 for t = 1:T-1]

# optimize nominal trajectories
z0 = [x_ref[1];y_ref[1];0.0;0.0]
Q = Diagonal(@SVector[100.0,100.0,1.0e-2,1.0e-2])
Qf = Diagonal(@SVector[100.0,100.0,1.0e-2,1.0e-2])
R = 1.0e-3*sparse(I,m,m)

n_nlp = n*T + m*(T-1)
m_nlp = n*T

prob = Problem(n_nlp,m_nlp,z_ref,u_ref,T,n,m,Q,Qf,R,model,Δt,false)

x0 = zeros(n_nlp)

for t = 1:T-1
    x0[(t-1)*(n+m) .+ (1:(n+m))] = [z_ref[t];u_ref[t]]
end
x0[(T-1)*(n+m) .+ (1:(n))] = z_ref[T]

x_sol = solve_ipopt(x0,prob)
x_nom = zeros(T)
y_nom = zeros(T)

z_nom = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
u_nom = [x_sol[(t-1)*(n+m)+n .+ (1:m)] for t = 1:T-1]
for t = 1:T
    x_nom[t] = x_sol[(t-1)*(n+m) .+ (1:n)][1]
    y_nom[t] = x_sol[(t-1)*(n+m) .+ (1:n)][2]
end

# tvlqr
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
    println(t)
    push!(K,(R + B[t]'*P[end]*B[t])\(B[t]'*P[end]*A[t]))
    push!(P,A[t]'*P[end]*A[t] - (A[t]'*P[end]*B[t])*K[end] + Q)
end

# optimize controller
N = 2*n
mv = Distributions.MvNormal(z_nom[1],0.1*Diagonal(ones(n)))
Z0 = rand(mv,N)
#
z0 = [Z0[:,i] for i = 1:N]
# z0 = [[0.1;0.;0.;0.], [0.;0.1;;0.;0.],[-0.1;0.;0.;0.],[0.;-0.1;0.;0.]]

n_nlp_ctrl = n*N*T + (m*n)*(T-1)
m_nlp_ctrl = n*N*T

prob_ctrl = ProblemCtrl(n_nlp_ctrl,m_nlp_ctrl,z_nom,u_nom,T,n,m,Q,Qf,R,A,B,model,Δt,N,z0,false)
x0_ctrl = zeros(n_nlp_ctrl)

for t = 1:T
    for i = 1:N
        x0_ctrl[(t-1)*(n*N + m*n)+(i-1)*n .+ (1:n)] = z_nom[t] + 1.0*randn(n)*0.001
    end
end

MOI.eval_objective(prob_ctrl,x0_ctrl)
grad_f = ones(n_nlp_ctrl)
MOI.eval_objective_gradient(prob_ctrl,grad_f,x0_ctrl)
grad_f
g = ones(m_nlp_ctrl)
MOI.eval_constraint(prob_ctrl,g,x0_ctrl)
g
jac = ones(m_nlp_ctrl*n_nlp_ctrl)
MOI.eval_constraint_jacobian(prob_ctrl,jac,x0_ctrl)

x_sol = solve_ipopt(x0_ctrl,prob_ctrl)

x_ctrl = [zeros(T) for i = 1:N]
y_ctrl = [zeros(T) for i = 1:N]
K_ctrl = [reshape(x_sol[(t-1)*(n*N+m*n)+n*N .+ (1:m*n)],m,n) for t = 1:T-1]
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
K












K_ctrl

# TVLQR
times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
z_tvlqr_rollout = [z_nom[1]]
u_tvlqr = []
for tt = 1:T_sim-1
    t = t_sim[tt]
    k = searchsortedlast(times,t)
    w = randn(n)*1.0e-2
    z = z_tvlqr_rollout[end] + 1.0*w
    u = u_nom[k] - K[k]*(z - z_nom[k])
    push!(z_tvlqr_rollout,midpoint(model,z,u,dt_sim))
    push!(u_tvlqr,u)
end

z_tvlqr_rollout
x_tvlqr_rollout = [z_tvlqr_rollout[t][1] for t = 1:T_sim]
y_tvlqr_rollout = [z_tvlqr_rollout[t][2] for t = 1:T_sim]

plot(x_ref,y_ref,title="simulation",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")
plot!(x_tvlqr_rollout,y_tvlqr_rollout,color=:purple,label="TVLQR",width=2.0)

# sampled controller
times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
z_ctrl_rollout = [z_nom[1]]
u_ctrl = []
for tt = 1:T_sim-1
   t = t_sim[tt]
   k = searchsortedlast(times,t)
   w = randn(n)*1.0e-2
   z = z_ctrl_rollout[end] + 1.0*w
   u = u_nom[k] - K_ctrl[k]*(z - z_nom[k])
   push!(z_ctrl_rollout,midpoint(model,z,u,dt_sim))
   push!(u_ctrl,u)
end

z_ctrl_rollout
x_ctrl_rollout = [z_ctrl_rollout[t][1] for t = 1:T_sim]
y_ctrl_rollout = [z_ctrl_rollout[t][2] for t = 1:T_sim]

plot!(x_ctrl_rollout,y_ctrl_rollout,color=:cyan,label="sample ctrl",width=2.0)
