using LinearAlgebra, ForwardDiff, Distributions, Plots
include("ipopt.jl")
include("integration.jl")
include("control.jl")

# continuous-time dynamics
n = 2
m = 1
Ac = [0.0 1.0; 0.0 0.0]
Bc = [0.0; 1.0]

# discrete-time dynamics
Δt = 0.1
D = exp(Δt*[Ac Bc; zeros(1,n+m)])
A = D[1:n,1:n]
B = D[1:n,n .+ (1:m)]

function dynamics(x,u,Δt)
    Δt = 0.1
    D = exp(Δt*[Ac Bc; zeros(1,n+m)])
    A = D[1:n,1:n]
    B = D[1:n,n .+ (1:m)]

    return A*x + B*u
end

# TVLQR solution
T = 20
Q = Matrix(1.0*I,n,n)
R = Matrix(0.1*I,m,m)

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q
for t = T-1:-1:1
    K[t] = (R + B'*P[t+1]*B)\(B'*P[t+1]*A)
    P[t] = Q + K[t]'*R*K[t] + (A-B*K[t])'*P[t+1]*(A-B*K[t])
end

# number of samples
x11 = [1.0; 0.0]
x12 = [-1.0; 0.0]
x13 = [0.0; 1.0]
x14 = [0.0; -1.0]

x1 = [x11,x12,x13,x14]

N = length(x1)

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1)) + N*(m*(T-1)) #+ N*(n*(T-1))

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]
# idx_con_k = [[N*(m*(T-1)) + N*(n*(T-1)) + (i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]

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
    for t = 1:T-1
        for i = 1:N
            x = (t==1 ? x1[i] : view(z,idx_x[i][t-1]))
            x⁺ = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            k = reshape(view(z,idx_k[t]),m,n)
            c[idx_con_dyn[i][t]] = A*x + B*u - x⁺
            c[idx_con_ctrl[i][t]] = u + k*x
        end
    end
    return c
end

c0 = rand(m_nlp)
con!(c0,ones(n_nlp))

# idx_ineq = vcat(vcat(idx_con_k...)...)
prob = Problem(n_nlp,m_nlp,obj,con!,true)#,idx_ineq=idx_ineq)

z0 = rand(n_nlp)
z_sol = solve(z0,prob)

K_sample = [reshape(z_sol[idx_k[t]],m,n) for t = 1:T-1]
println("solution error: $([norm(vec(K_sample[t]-K[t])) for t = 1:T-1])")
z_sol_K = copy(z_sol)
for t = 1:T-1
    z_sol_K[idx_k[t]] = vec(K[t])
    for i = 1:N
        z_sol_K[idx_u[i][t]] = -K[t]*(t==1 ? x1[i] : z_sol_K[idx_x[i][t-1]])
        z_sol_K[idx_x[i][t]] = A*(t==1 ? x1[i] : z_sol_K[idx_x[i][t-1]]) + B*z_sol_K[idx_u[i][t]]
    end
end
norm(vec(z_sol - z_sol_K))
obj(z_sol)
obj(z_sol_K)

# simulate controller
T_sim = 2*T
μ = zeros(2)
Σ = Diagonal(1.0e-4*rand(2))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)
z0_sim = rand(n)

plt = plot(xlims=(-2,2),ylims=(-2,2),aspect_ratio=:equal)
plt = scatter!([0.0],[0.0],color=:green,marker=:circle,label="",width=2.0)
plt = scatter!([z0_sim[1]+w[1,1]],[z0_sim[2]+w[2,1]],color=:red,marker=:square,label="",width=2.0)

z_tvlqr, u_tvlqr = simulate_linear_controller(K,[zeros(n) for t = 1:T],[zeros(m) for t = 1:T-1],T_sim,Δt,z0_sim,w)
x_tvlqr = [z_tvlqr[t][1] for t = 1:T_sim]
y_tvlqr = [z_tvlqr[t][2] for t = 1:T_sim]
plt = plot!(x_tvlqr,y_tvlqr,color=:purple,label="",width=2.0)

z_sample, u_sample = simulate_linear_controller(K_sample,[zeros(n) for t = 1:T],[zeros(m) for t = 1:T-1],T_sim,Δt,z0_sim,w)
x_sample = [z_sample[t][1] for t = 1:T_sim]
y_sample = [z_sample[t][2] for t = 1:T_sim]
plt = plot!(x_sample,y_sample,color=:orange,label="",width=2.0)

# plot(hcat(u_tvlqr...)')
# plot!(hcat(u_sample...)')
#
# K_sample
# plot(vec(hcat(K_sample...) - hcat(K...)))
