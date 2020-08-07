include("../src/sample_trajectory_optimization.jl")
include("../tests/ipopt.jl")
include("../dynamics/pendulum.jl")

# Pendulum discrete-time dynamics (midpoint)
Δt = 0.05
function discrete_dynamics(model,x,u,Δt,t)
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
    z0_nom[x_nom_idx[t]] = x_nom_ref[t]
end

function obj_nom(z)
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

obj_nom(z0_nom)

# Constraints
function con_nom!(c,z)
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
A
B

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

function objective(z)
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

function con!(c,z)
    β = 1.0
    w = 1.0e-1*ones(nx)
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_dynamics_linear(xs,u,A[t],B[t],β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),nu,nx)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*xs[i]
        end
    end
    return c
end

prob = ProblemIpopt(n_nlp,m_nlp,objective,con!,false)

z0 = rand(n_nlp)
z_sol = solve(z0,prob)

K_sample = [reshape(z_sol[idx_k[t]],nu,nx) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error: $(sum(K_error)/N)")

plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,
    ylims=(1.0e-16,1.0),width=2.0,legend=:bottom,
    title="Gain matrix error")
