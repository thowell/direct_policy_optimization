include("../src/sample_trajectory_optimization.jl")
include("../tests/ipopt.jl")
include("../dynamics/pendulum.jl")

# Pendulum discrete-time dynamics (midpoint)
Δt = 0.05
function discrete_dynamics(model,x,u,Δt)
    midpoint(model,x,u,Δt)
end

nx = model.nx
nu = model.nu

# Trajectory optimization for nominal trajectory
T = 20
x1_nom = [0.0; 0.0]
xT_nom = [π; 0.0]

x_nom_ref = linear_interp(x1_nom,xT_nom,T)

Q_nom = [t < T ? Diagonal([1.0; 0.1]) : Diagonal([10.0; 1.0]) for t = 1:T]
R_nom = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]

x_nom_idx = [(t-1)*(nx+nu) .+ (1:nx) for t = 1:T]
u_nom_idx = [(t-1)*(nx+nu) + nx .+ (1:nu) for t = 1:T-1]

n_nom_nlp = nx*T + nu*(T-1)
m_nom_nlp = nx*(T+1)

z0_nom = 1.0e-5*randn(n_nom_nlp)
for t = 1:T
    z0_nom[x_idx[t]] = x_nom_ref[t]
end

function obj_nom(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        s += x'*Q_nom[t]*x + u'*R_nom[t]*u
    end
    x = z[x_nom_idx[T]]
    s += (x-xT_nom)'*Q_nom[T]*(x-xT_nom)

    return s
end

obj_nom(z0_nom)

# Constraints
function con_nom!(c,z)
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        x⁺ = z[x_nom_idx[t+1]]
        c[(t-1)*nx .+ (1:nx)] = x⁺ - dynamics_discrete(x,u,Δt)
    end
    c[(T-1)*nx .+ (1:nx)] = z[x_nom_idx[1]] - x1_nom
    c[T*nx .+ (1:nx)] = z[x_nom_idx[T]] - xT_nom
    return c
end

c0_nom = zeros(m_nom_nlp)
con_nom!(c0_nom,z0_nom)

# NLP problem
prob_nom = ProblemIpopt(n_nom_nlp,m_nom_nlp,obj_nom,con_nom!,true)

# Solve
z_nom_sol = solve(z0_nom,prob_nom)

x_nom = [z_nom_sol[x_nom_idx[t]] for t = 1:T]
u_nom = [z_nom_sol[u_nom_idx[t]] for t = 1:T-1]
θ_nom_sol = vec([x_nom[t][1] for t = 1:T])
dθ_nom_sol = vec([x_nom[t][2] for t = 1:T])

plot(hcat(x_nom...)',xlabel="time step",ylabel="state",label=["θ" "dθ"],width=2.0,legend=:topleft)
plot(hcat(u_nom...)',xlabel="time step",ylabel="control",label="",width=2.0)
plot(θ_nom_sol,dθ_nom_sol,xlabel="θ",ylabel="dθ",width=2.0)

# TVLQR solution
A = []
B = []
for t = 1:T-1
    x = x_nom[t]
    u = u_nom[t]
    fx(z) = dynamics(z,u,Δt)
    fu(z) = dynamics(x,z,Δt)

    push!(A,ForwardDiff.jacobian(fx,x))
    push!(B,ForwardDiff.jacobian(fu,u))
end

Q = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0;100.0]) for t = 1:T]
R = [Diagonal(ones(nu)) for t = 1:T-1]
K = TVLQR(A,B,Q,R)

# Samples
α = 1.0
x11 = α*[1.0; 0.0]
x12 = α*[-1.0; 0.0]
x13 = α*[0.0; 1.0]
x14 = α*[0.0; -1.0]
x1 = [x11,x12,x13,x14]

N = length(x1)

# Indices
n_nlp = N*(nx*(T-1) + nu*(T-1)) + nu*nx*(T-1)
m_nlp = N*(nx*(T-1) + nu*(T-1))

idx_k = [(t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]
idx_x = [[(T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1)) + (t-1)*(nx+nu) .+ (1:nx) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1)) + (t-1)*(nx+nu) + nx .+ (1:nu) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(nx*(T-1)) + (t-1)*nx .+ (1:nx) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(nu*(T-1)) + N*(nx*(T-1)) + (t-1)*nu .+ (1:nu) for t = 1:T-1] for i = 1:N]

function obj(z)
    s = 0
    for t = 1:T-1
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            s += (x - x_nom[t+1])'*Q[t+1]*(x - x_nom[t+1]) + (u - u_nom[t])'*R[t]*(u - u_nom[t])
        end
    end
    return s
end

function con!(c,z)
    β = 1.0
    w = 1.0e-1
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_dynamics(model,xs,u,Δt,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),nu,nx)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*(xs[i] - x_nom[t]) - u_nom[t]
        end
    end
    return c
end

prob = ProblemIpopt(n_nlp,m_nlp,obj,con!,true)

z0 = rand(n_nlp)
z_sol_s = solve(z0,prob)

K_sample = [reshape(z_sol_s[idx_k[t]],nu,nx) for t = 1:T-1]
K_difference = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("solution difference: $(sum(K_difference)/N)")

using Plots
plot(K_difference,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,
    width=2.0,label="β=$β",title="Gain matrix difference",
    legend=:bottomright)

# Simulate controllers
using Distributions
model_sim = model
T_sim = 100*T

μ = zeros(nx)
Σ = Diagonal(1.0e-1*ones(nx))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(nx)
Σ0 = Diagonal(1.0e-2*ones(nx))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(x_nom[1]) + w0)

t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

plt = plot(t_nom,hcat(x_nom...)[1,:],legend=:bottom,color=:red,label="ref.",
    width=2.0,xlabel="time (s)",title="Pendulum",ylabel="state")
plt = plot!(t_nom,hcat(x_nom...)[2,:],color=:red,label="",width=2.0)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)
plt = plot!(t_sim,hcat(z_tvlqr...)[1,:],color=:purple,label="tvlqr",width=2.0)
plt = plot!(t_sim,hcat(z_tvlqr...)[2,:],color=:purple,label="",width=2.0)

z_sample, u_sample, J_sample,Jx_sample, Ju_sample = simulate_linear_controller(K_sample,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)
plt = plot!(t_sim,hcat(z_sample...)[1,:],linetype=:steppost,color=:orange,label="sample",width=2.0)
plt = plot!(t_sim,hcat(z_sample...)[2,:],linetype=:steppost,color=:orange,label="",width=2.0)

plot(t_nom[1:end-1],hcat(u_nom...)',color=:red,label="ref.",linetype=:steppost,
    title="Pendulum",yaxis="control")
plot!(t_sim[1:end-1],hcat(u_tvlqr...)',color=:purple,label="tvlqr",linetype=:steppost)
plot!(t_sim[1:end-1],hcat(u_sample...)',color=:orange,label="sample",linetype=:steppost)

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
