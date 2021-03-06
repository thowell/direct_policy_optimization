include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"tests/ipopt.jl"))

# double-integrator continuous-time dynamics
nx = 2
nu = 1
Ac = [0.0 1.0; 0.0 0.0]
Bc = [0.0; 1.0]

# double-integrator discrete-time dynamics
Δt = 0.1
D = exp(Δt*[Ac Bc; zeros(1,nx+nu)])
A = D[1:nx,1:nx]
B = D[1:nx,nx .+ (1:nu)]

function dynamics(x,u,Δt)
    Δt = 0.1
    D = exp(Δt*[Ac Bc; zeros(1,nx+nu)])
    A = D[1:nx,1:nx]
    B = D[1:nx,nx .+ (1:nu)]

    return A*x + B*u
end

# LQR solution
T = 51
Q = Diagonal(ones(nx))
R = Diagonal(ones(nu))
K = TVLQR([A for t=1:T-1],[B for t=1:T-1],[Q for t=1:T],[R for t=1:T-1])

# 4 samples
α = 1.0
x11 = α*[1.0; 1.0]
x12 = α*[1.0; -1.0]
x13 = α*[-1.0; 1.0]
x14 = α*[-1.0; -1.0]
x1 = [x11,x12,x13,x14]

N = length(x1)

# indices
n_nlp = N*(nx*(T-1) + nu*(T-1)) + nu*nx*(T-1)
m_nlp = N*(nx*(T-1) + nu*(T-1))

idx_θ = [(t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]
idx_x = [[((T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1))
    + (t-1)*(nx+nu) .+ (1:nx)) for t = 1:T-1] for i = 1:N]
idx_u = [[((T-1)*(nu*nx) + (i-1)*(nx*(T-1) + nu*(T-1))
    + (t-1)*(nx+nu) + nx .+ (1:nu)) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(nx*(T-1)) + (t-1)*nx .+ (1:nx) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[((i-1)*(nu*(T-1)) + N*(nx*(T-1))
    + (t-1)*nu .+ (1:nu)) for t = 1:T-1] for i = 1:N]

function obj(z)
    s = 0
    for t = 1:T-1
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            s += x'*Q*x + u'*R*u
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
        xs⁺ = sample_dynamics_linear(xs,u,A,B,β=β,w=w,fast_sqrt=true)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        θ = reshape(view(z,idx_θ[t]),nu,nx)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + θ*xs[i]
        end
    end
    return c
end

prob = ProblemIpopt(n_nlp,m_nlp)

z0 = ones(n_nlp)
z_sol = solve(z0,prob)

Θ = [reshape(z_sol[idx_θ[t]],nu,nx) for t = 1:T-1]
policy_error = [norm(vec(Θ[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (avg.): $(sum(policy_error)/T)")

using Plots
plt = plot(policy_error,xlabel="time step",ylims=(1.0e-16,1.0),yaxis=:log,
    width=2.0,legend=:bottom,ylabel="matrix-norm error",label="")
savefig(plt,joinpath(@__DIR__,"results/LQR_double_integrator.png"))

using PGFPlots
const PGF = PGFPlots

p = PGF.Plots.Linear(range(1,stop=T-1,length=T-1),policy_error,mark="",style="color=black, very thick")

a = Axis(p,
    xmin=1., ymin=1.0e-16, xmax=T-1, ymax=1.0,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="matrix-norm error",
	xlabel="time step",
	ymode="log",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"LQR_double_integrator.tikz"), a, include_preamble=false)
