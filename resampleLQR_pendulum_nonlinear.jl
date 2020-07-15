using LinearAlgebra, ForwardDiff, Distributions, Plots
include("ipopt.jl")
include("integration.jl")
include("control.jl")

function fastsqrt(A)
    #FASTSQRT computes the square root of a matrix A with Denman-Beavers iteration

    #S = sqrtm(A);
    Ep = 1e-8*Matrix(I,size(A))

    if count(diag(A) .> 0.0) != size(A,1)
        S = diagm(sqrt.(diag(A)));
        return S
    end

    In = Matrix(1.0*I,size(A));
    S = A;
    T = Matrix(1.0*I,size(A));

    T = .5*(T + inv(S+Ep));
    S = .5*(S+In);
    for k = 1:4
        Snew = .5*(S + inv(T+Ep));
        T = .5*(T + inv(S+Ep));
        S = Snew;
    end
    return S
end

n = 2
m = 1

struct Pendulum
    m
    l
    b
    lc
    I
    g
end

function dyn_c(model,x,u)
    m = model.m
    l = model.l
    b = model.b
    lc = model.lc
    I = model.I
    g = model.g
    return [x[2];
        u[1]/(m*lc*lc) - g*sin(x[1])/lc - b*x[2]/(m*lc*lc)]
end

# Pendulum discrete-time dynamics (midpoint)
Δt = 0.05
function dynamics(model,x,u,Δt)
    x + Δt*dyn_c(model,x + 0.5*Δt*dyn_c(model,x,u),u)
end

model = Pendulum(1.0,0.5,0.1,0.5,0.25,9.81)

# Trajectory optimization
T = 20
x1 = [0.0; 0.0]
xT = [π; 0.0]

function linear_interp(x1,xT,T)
    n = length(x1)
    X = [copy(Array(x1)) for t = 1:T]
    for t = 1:T
        for i = 1:n
            X[t][i] = (xT[i]-x1[i])/(T-1)*(t-1) + x1[i]
        end
    end

    return X
end
x_ref = linear_interp(x1,xT,T)

Q = [t < T ? Diagonal([1.0; 0.1]) : Diagonal([10.0; 1.0]) for t = 1:T]
R = [Diagonal(0.1*ones(m)) for t = 1:T-1]

x_idx = [(t-1)*(n+m) .+ (1:n) for t = 1:T]
u_idx = [(t-1)*(n+m) + n .+ (1:m) for t = 1:T-1]

n_nlp = n*T + m*(T-1)
m_nlp = n*(T+1)

z0 = 1.0e-5*randn(n_nlp)
for t = 1:T
    z0[x_idx[t]] = x_ref[t]
end

function obj_traj(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        s += x'*Q[t]*x + u'*R[t]*u
    end
    x = z[x_idx[T]]
    s += (x-xT)'*Q[T]*(x-xT)

    return s
end

obj_traj(z0)

# Constraints
function con_traj!(c,z)
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        x⁺ = z[x_idx[t+1]]
        c[(t-1)*n .+ (1:n)] = x⁺ - dynamics(model,x,u,Δt)
    end
    c[(T-1)*n .+ (1:n)] = z[x_idx[1]] - x1
    c[T*n .+ (1:n)] = z[x_idx[T]] - xT
    return c
end

c0 = zeros(m_nlp)
con_traj!(c0,z0)

# NLP problem
prob = Problem(n_nlp,m_nlp,obj_traj,con_traj!,true)

# Solve
z_sol = solve(z0,prob)

x_nom = [z_sol[x_idx[t]] for t = 1:T]
u_nom = [z_sol[u_idx[t]] for t = 1:T-1]
θ_sol = vec([x_nom[t][1] for t = 1:T])
dθ_sol = vec([x_nom[t][2] for t = 1:T])

plot(hcat(x_nom...)',xlabel="time step",ylabel="state",label=["θ" "dθ"],width=2.0,legend=:topleft)
plot(hcat(u_nom...)',xlabel="time step",ylabel="control",label="",width=2.0)
plot(θ_sol,dθ_sol,xlabel="θ",ylabel="dθ",width=2.0)

# TVLQR solution
A = []
B = []
for t = 1:T-1
    x = x_nom[t]
    u = u_nom[t]
    fx(z) = dynamics(model,z,u,Δt)
    fu(z) = dynamics(model,x,z,Δt)

    push!(A,ForwardDiff.jacobian(fx,x))
    push!(B,ForwardDiff.jacobian(fu,u))
end

Q = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0;100.0]) for t = 1:T]
R = [Diagonal(ones(m)) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]

P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

β = 1.0
x11 = β*[1.0; 0.0] + x_nom[1]
x12 = β*[-1.0; 0.0] + x_nom[1]
x13 = β*[0.0; 1.0] + x_nom[1]
x14 = β*[0.0; -1.0] + x_nom[1]

x1 = [x11,x12,x13,x14]

N = length(x1)

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1) + m*(T-1))

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

function resample(X; β=1.0,w=1.0)
    N = length(X)
    n = length(X[1])

    xμ = sum(X)./N
    Σμ = (0.5/(β^2))*sum([(X[i] - xμ)*(X[i] - xμ)' for i = 1:N]) + w*I
    # cols = cholesky(Σμ).U
    cols = fastsqrt(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]

    return Xs
end

function sample_nonlinear_dynamics(model,X,U; β=1.0,w=1.0)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,dynamics(model,X[i],U[i],Δt))
    end
    # return X⁺
    Xs⁺ = resample(X⁺,β=β,w=w)
    return Xs⁺
end

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

function ∇obj!(g,z)
    g .= 0.0
    for t = 1:T-1
        for i = 1:N
            x = z[idx_x[i][t]]
            u = z[idx_u[i][t]]
            g[idx_x[i][t]] = 2.0*Q[t+1]*(x - x_nom[t+1])
            g[idx_u[i][t]] = 2.0*R[t]*(u - u_nom[t])
        end
    end
    return nothing
end

function con!(c,z)
    β = 1.0
    w = 1.0e-1
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_nonlinear_dynamics(model,xs,u,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),m,n)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*(xs[i] - x_nom[t]) - u_nom[t]
        end
    end
    return c
end

c0 = rand(m_nlp)
con!(c0,ones(n_nlp))

grad_f1 = ones(n_nlp)
grad_f2 = ones(n_nlp)
∇obj!(grad_f1,z0)
ForwardDiff.gradient!(grad_f2,obj,z0)
grad_f1
grad_f2
norm(grad_f1-grad_f2)
# ∇obj!(g,z) = ForwardDiff.gradient!(g,obj,z)
∇con!(∇c,z) = ForwardDiff.jacobian!(∇c,con!,zeros(eltype(z),m_nlp),z)
prob = problem_derivatives(n_nlp,m_nlp,obj,∇obj!,con!,∇con!,true)

# prob = Problem(n_nlp,m_nlp,obj,con!,true)

z0 = rand(n_nlp)
z_sol_s = solve(z0,prob)

K_sample = [reshape(z_sol_s[idx_k[t]],m,n) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("solution error: $(sum(K_error)/N)")

plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,width=2.0,label="β=$β",title="Gain matrix error")

# simulate controllers
model_sim = model
T_sim = 10*T

μ = zeros(n)
Σ = Diagonal(1.0e-1*ones(n))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(n)
Σ0 = Diagonal(1.0e-2*ones(n))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(x_nom[1]) + w0)

z_nom_sim, u_nom_sim = nominal_trajectories(x_nom,u_nom,T_sim,Δt)
t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

plt = plot(t_nom,hcat(x_nom...)[1,:],legend=:bottom,linetype=:steppost,color=:red,label="ref.",width=2.0,xlabel="time (s)")
plt = plot!(t_nom,hcat(x_nom...)[2,:],linetype=:steppost,color=:red,label="",width=2.0)

z_tvlqr, u_tvlqr, J_tvlqr = simulate_linear_controller(K,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)
plt = plot!(t_sim,hcat(z_tvlqr...)[1,:],linetype=:steppost,color=:purple,label="tvlqr",width=2.0)
plt = plot!(t_sim,hcat(z_tvlqr...)[2,:],linetype=:steppost,color=:purple,label="",width=2.0)

z_sample, u_sample, J_sample = simulate_linear_controller(K_sample,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,_norm=2)
plt = plot!(t_sim,hcat(z_sample...)[1,:],linetype=:steppost,color=:orange,label="sample",width=2.0)
plt = plot!(t_sim,hcat(z_sample...)[2,:],linetype=:steppost,color=:orange,label="",width=2.0)

plot(t_nom[1:end-1],vcat(K...)[:,1],xlabel="time (s)",title="Gains",label="tvlqr",width=2.0,color=:purple,linetype=:steppost)
plot!(t_nom[1:end-1],vcat(K...)[:,2],label="",width=2.0,color=:purple,linetype=:steppost)

plot!(t_nom[1:end-1],vcat(K_sample...)[:,1],legend=:bottom,label="sample",color=:orange,width=2.0,linetype=:steppost)
plot!(t_nom[1:end-1],vcat(K_sample...)[:,2],label="",color=:orange,width=2.0,linetype=:steppost)

plot(hcat(u_nom_sim...)',color=:red,label="ref.",linetype=:steppost)
plot!(hcat(u_tvlqr...)',color=:purple,label="tvlqr",linetype=:steppost)
plot!(hcat(u_sample...)',color=:orange,label="sample",linetype=:steppost)

# objective value
J_tvlqr
J_sample
