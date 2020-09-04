include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"tests/ipopt.jl"))
include(joinpath(pwd(),"dynamics/pendulum.jl"))
using Plots

# pendulum dimensions
nx = model.nx
nu = model.nu

# Trajectory optimization for nominal trajectory
T = 51
Δt = 0.05
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
    z0_nom[x_nom_idx[t]] = x_nom_ref[t]
end

function obj(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        s += x'*Q_nom[t]*x + u'*R_nom[t]*u
    end
    x = z[x_nom_idx[T]]
    s += (x-xT_nom)'*Q_nom[T]*(x-xT_nom)

    return s
end

obj(z0_nom)

# Constraints
function con!(c,z)
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        x⁺ = z[x_nom_idx[t+1]]
        c[(t-1)*nx .+ (1:nx)] = x⁺ - discrete_dynamics(model,x,u,Δt,t)
    end
    c[(T-1)*nx .+ (1:nx)] = z[x_nom_idx[1]] - x1_nom
    c[T*nx .+ (1:nx)] = z[x_nom_idx[T]] - xT_nom
    return c
end

c0_nom = zeros(m_nom_nlp)
con!(c0_nom,z0_nom)

# NLP problem
prob_nom = ProblemIpopt(n_nom_nlp,m_nom_nlp)

# Solve
z_nom_sol = solve(z0_nom,prob_nom)

x_nom = [z_nom_sol[x_nom_idx[t]] for t = 1:T]
u_nom = [z_nom_sol[u_nom_idx[t]] for t = 1:T-1]
θ_nom_sol = vec([x_nom[t][1] for t = 1:T])
dθ_nom_sol = vec([x_nom[t][2] for t = 1:T])

plot(hcat(x_nom...)',xlabel="time step",ylabel="state",label=["θ" "dθ"],
    width=2.0,legend=:topleft)
plot(hcat(u_nom...)',xlabel="time step",ylabel="control",label="",width=2.0)
plot(θ_nom_sol,dθ_nom_sol,xlabel="θ",ylabel="dθ",width=2.0)

# TVLQR solution
Q = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0;100.0]) for t = 1:T]
R = [Diagonal(ones(nu)) for t = 1:T-1]
A, B = nominal_jacobians(model,x_nom,u_nom,[Δt for t = 1:T-1])
K = TVLQR_gains(model,x_nom,u_nom,[Δt for t = 1:T-1],Q,R)

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

idx_θ = [(t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]
idx_x = [[((T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1))
    + (t-1)*(nx+nu) .+ (1:nx)) for t = 1:T-1] for i = 1:N]
idx_u = [[((T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1))
    + (t-1)*(nx+nu) + nx .+ (1:nu)) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[((i-1)*(nx*(T-1))
    + (t-1)*nx .+ (1:nx)) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[((i-1)*(nu*(T-1)) + N*(nx*(T-1))
    + (t-1)*nu .+ (1:nu)) for t = 1:T-1] for i = 1:N]

function obj(z)
    s = 0
    for t = 1:T-1
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            s += x'*Q[t+1]*x + u'*R[t]*u
        end
    end
    return s
end

# linearized dynamics
function con!(c,z)
    β = 1.0
    w = 1.0e-1*ones(nx)
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_dynamics_linear(xs,u,A[t],B[t],β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        θ = reshape(view(z,idx_θ[t]),nu,nx)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + θ*xs[i]
        end
    end
    return c
end

prob_linear = ProblemIpopt(n_nlp,m_nlp)

z0 = ones(n_nlp)
z_sol_linear = solve(z0,prob_linear)

Θ_linear = [reshape(z_sol_linear[idx_θ[t]],nu,nx) for t = 1:T-1]
policy_error_linear = [norm(vec(Θ_linear[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (avg.) [linear dynamics]:
    $(sum(policy_error_linear)/T)")

plt = plot(policy_error_linear,xlabel="time step",ylabel="matrix-norm error",yaxis=:log,
    ylims=(1.0e-16,1.0),width=2.0,legend=:bottom,label="")
savefig(plt,joinpath(@__DIR__,"results/TVLQR_pendulum.png"))

# nonlinear dynamics
function con!(c,z)
    β = 1.0
    w = 1.0e-1*ones(nx)
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_dynamics(model,xs,u,Δt,t,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        θ = reshape(view(z,idx_θ[t]),nu,nx)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + θ*xs[i]
        end
    end
    return c
end

prob_nonlinear = ProblemIpopt(n_nlp,m_nlp)

z_sol_nonlinear = solve(z_sol_linear,prob_nonlinear)

Θ_nonlinear = [reshape(z_sol_nonlinear[idx_θ[t]],nu,nx) for t = 1:T-1]
policy_error_nonlinear = [norm(vec(Θ_nonlinear[t]-K[t]))/norm(vec(K[t]))
    for t = 1:T-1]
println("Policy solution difference (avg.) [nonlinear dynamics]:
    $(sum(policy_error_nonlinear)/T)")

# Simulate policy
using Distributions
model_sim = model
T_sim = 10*T

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-3*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-3*ones(nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(x_nom[1]) + w0)

t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)

z_linear, u_linear, J_linear, Jx_linear, Ju_linear = simulate_linear_controller(Θ_linear,
    x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)

z_nonlin, u_nonlin, J_nonlin, Jx_nonlin, Ju_nonlin = simulate_linear_controller(Θ_nonlinear,
    x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)

plt_x = plot(t_nom,hcat(x_nom...)[1:2,:]',legend=:topright,color=:red,
    label=["θ (nom.)" "dθ (nom.)"],width=2.0,xlabel="time (s)",
    title="Pendulum",ylabel="state")
plt_x = plot!(t_sim,hcat(z_tvlqr...)[1:2,:]',color=:purple,label="tvlqr",
    width=2.0)
plt_x = plot!(t_sim,hcat(z_linear...)[1:2,:]',linetype=:steppost,color=:cyan,
    label=["Θ (linear)" "dθ (linear)"],width=2.0)
plt_x = plot!(t_sim,hcat(z_nonlin...)[1:2,:]',linetype=:steppost,color=:orange,
    label=["Θ (nonlinear)" "dθ (nonlinear)"],width=2.0)

plt_u = plot(t_nom[1:T-1],hcat(u_nom...)[1:1,:]',legend=:topright,color=:red,
    label=["nominal"],width=2.0,xlabel="time (s)",
    title="Pendulum",ylabel="control",linetype=:steppost)
plt_u = plot!(t_sim[1:T_sim-1],hcat(u_tvlqr...)[1:1,:]',color=:purple,label="tvlqr",
    width=2.0)
plt_u = plot!(t_sim[1:T_sim-1],hcat(u_linear...)[1:1,:]',color=:cyan,label="linear",
    width=2.0)
plt_u = plot!(t_sim[1:T_sim-1],hcat(u_nonlin...)[1:1,:]',color=:orange,label="nonlinear",
    width=2.0)

# objective value
J_tvlqr
J_linear
J_nonlin

# state tracking
Jx_tvlqr
Jx_linear
Jx_nonlin

# control tracking
Ju_tvlqr
Ju_linear
Ju_nonlin
