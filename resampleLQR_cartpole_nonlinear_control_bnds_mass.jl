using LinearAlgebra, ForwardDiff, Distributions, Plots, StaticArrays
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

mutable struct Cartpole{T}
    mc::T # mass of the cart in kg (10)
    mp::T # mass of the pole (point mass at the end) in kg
    l::T  # length of the pole in m
    g::T  # gravity m/s^2
end

function dyn_c(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[2]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,1:2) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

model = Cartpole(1.0,0.2,0.5,9.81)
n, m = 4,1

# Cartpole discrete-time dynamics (midpoint)
Δt = 0.05
function dynamics(model,x,u,Δt)
    x + Δt*dyn_c(model,x + 0.5*Δt*dyn_c(model,x,u),u)
end

# Trajectory optimization
T = 40
x1 = zeros(n)
xT = [0.0;π;0.0;0.0]

ul = -10.0
uu = 10.0

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

Q = [t!=T ? Diagonal(1.0*@SVector [1.0,1.0,1.0,1.0]) : Diagonal(1.0*@SVector ones(n)) for t = 1:T]
R = [Diagonal(1.0e-1*@SVector ones(m)) for t = 1:T-1]

x_idx = [(t-1)*(n+m) .+ (1:n) for t = 1:T]
u_idx = [(t-1)*(n+m) + n .+ (1:m) for t = 1:T-1]

n_nlp = n*T + m*(T-1)
m_nlp = n*(T+1)

zl = -Inf*ones(n_nlp)
zu = Inf*ones(n_nlp)

z0 = 1.0e-5*randn(n_nlp)
for t = 1:T
    z0[x_idx[t]] = x_ref[t]

    if t < T
        zl[u_idx[t]] .= ul
        zu[u_idx[t]] .= uu
    end
end

function obj_traj(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        s += (x-xT)'*Q[t]*(x-xT) + u'*R[t]*u
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
prob = Problem(n_nlp,m_nlp,obj_traj,con_traj!,true,
    primal_bounds=(zl,zu))

# Solve
z_sol = solve(z0,prob)

x_nom = [z_sol[x_idx[t]] for t = 1:T]
u_nom = [z_sol[u_idx[t]] for t = 1:T-1]

plot(hcat(x_nom...)',xlabel="time step",ylabel="state",label=["x" "θ" "dx" "dθ"],width=2.0,legend=:topleft)
plot(hcat(u_nom...)',xlabel="time step",ylabel="control",label="",width=2.0)

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

Q = [t < T ? Diagonal(1.0*@SVector [10.0,10.0,1.0,1.0]) : Diagonal([100.0;100.0;100.0;100.0]) for t = 1:T]
R = [Diagonal(1.0*@SVector ones(m)) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]

P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

α = 1.0e-32
x11 = α*[1.0; 0.0; 0.0; 0.0] + x_nom[1]
x12 = α*[-1.0; 0.0; 0.0; 0.0] + x_nom[1]
x13 = α*[0.0; 1.0;; 0.0; 0.0] + x_nom[1]
x14 = α*[0.0; -1.0; 0.0; 0.0] + x_nom[1]
x15 = α*[0.0; 0.0; 1.0;; 0.0] + x_nom[1]
x16 = α*[0.0; 0.0; -1.0; 0.0] + x_nom[1]
x17 = α*[0.0; 0.0; 0.0; 1.0] + x_nom[1]
x18 = α*[0.0; 0.0; 0.0; -1.0] + x_nom[1]

x1 = [x11,x12,x13,x14,x15,x16,x17,x18]
x1_vec = vcat(x1...)

N = length(x1)

model1 = Cartpole(1.0,0.17,0.5,9.81)
model2 = Cartpole(1.0,0.18,0.5,9.81)
model3 = Cartpole(1.0,0.19,0.5,9.81)
model4 = Cartpole(1.0,0.195,0.5,9.81)
model5 = Cartpole(1.0,0.205,0.5,9.81)
model6 = Cartpole(1.0,0.21,0.5,9.81)
model7 = Cartpole(1.0,0.215,0.5,9.81)
model8 = Cartpole(1.0,0.22,0.5,9.81)

# models_mass = [model for i = 1:N]#model1,model2,model3,model4,model5,model6,model7,model8]
models_mass = [model1,model2,model3,model4,model5,model6,model7,model8]

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1) + m*(T-1))

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_x_vec = [vcat([idx_x[i][t] for i = 1:N]...) for t = 1:T-1]
idx_u_vec = [vcat([idx_u[i][t] for i = 1:N]...) for t = 1:T-1]

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

function sample_nonlinear_dynamics(models,X,U; β=1.0,w=1.0)
    N = length(X)
    X⁺ = []
    for i = 1:N
        push!(X⁺,dynamics(models[i],X[i],U[i],Δt))
    end
    # return X⁺
    Xs⁺ = resample(X⁺,β=β,w=w)
    return Xs⁺
end

idx_x_vec = [vcat([idx_x[i][t] for i = 1:N]...) for t = 1:T-1]
idx_u_vec = [vcat([idx_u[i][t] for i = 1:N]...) for t = 1:T-1]

function resample_vec(X; β=1.0,w=1.0)
    xμ = sum([X[(i-1)*n .+ (1:n)] for i = 1:N])./N
    Σμ = (0.5/(β^2))*sum([(X[(i-1)*n .+ (1:n)] - xμ)*(X[(i-1)*n .+ (1:n)] - xμ)' for i = 1:N]) + w*I
    cols = fastsqrt(Σμ)
    Xs = vcat([xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]...)
    return Xs
end

function sample_nonlinear_dynamics_vec(models,X,U; β=1.0,w=1.0)
    X⁺ = vcat([dynamics(models[i],X[(i-1)*n .+ (1:n)],U[(i-1)*m .+ (1:m)],Δt) for i = 1:N]...)
    # return X⁺
    Xs⁺ = resample_vec(X⁺,β=β,w=w)
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

function ∇obj!(g,z)
    g .= 0.0
    for t = 1:T-1
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])

            g[idx_x[i][t]] = 2.0*Q[t+1]*(x - x_nom[t+1])
            g[idx_u[i][t]] = 2.0*R[t]*(u - u_nom[t])
        end
    end
    return g
end

function con!(c,z)
    β = 1.0
    w = 1.0e-3
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_nonlinear_dynamics(models_mass,xs,u,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),m,n)

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*(xs[i] - x_nom[t]) - u_nom[t]
        end
    end
    return c
end

function ∇con_vec!(∇c,z)
    β = 1.0
    w = 1.0e-3
    Im = Matrix(I,m,m)
    ∇tmp_x = zeros(n*N,n*N)
    ∇tmp_u = zeros(n*N,m*N)

    shift = 0
    for t = 1:T-1
        xs = (t==1 ? x1_vec : view(z,idx_x_vec[t-1]))
        u = view(z,idx_u_vec[t])
        k = reshape(view(z,idx_k[t]),m,n)

        tmp_x(y) = sample_nonlinear_dynamics_vec(models_mass,y,u,β=β,w=w)
        tmp_u(y) = sample_nonlinear_dynamics_vec(models_mass,xs,y,β=β,w=w)

        ForwardDiff.jacobian!(∇tmp_x,tmp_x,xs)
        ForwardDiff.jacobian!(∇tmp_u,tmp_u,u)

        for i = 1:N
            if t > 1
                r_idx = idx_con_dyn[i][t]
                c_idx = idx_x_vec[t-1]
                len = length(r_idx)*length(c_idx)
                ∇c[shift .+ (1:len)] = vec(∇tmp_x[(i-1)*n .+ (1:n),1:n*N])
                shift += len
            end

            r_idx = idx_con_dyn[i][t]
            c_idx = idx_u_vec[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(∇tmp_u[(i-1)*n .+ (1:n),1:m*N])
            shift += len

            r_idx = idx_con_dyn[i][t]
            c_idx = idx_x[i][t]
            len = length(r_idx)
            ∇c[shift .+ (1:len)] .= -1.0
            shift += len

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_u[i][t]
            len = length(r_idx)
            ∇c[shift .+ (1:len)] .= 1.0
            shift += len

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_k[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(kron((xs[(i - 1)*n .+ (1:n)] - x_nom[t])',Im))
            shift += len

            if t > 1
                r_idx = idx_con_ctrl[i][t]
                c_idx = idx_k[t]
                len = length(r_idx)*length(c_idx)
                ∇c[shift .+ (1:len)] = vec(k)
                shift += len
            end
        end
    end
    return ∇c
end

function _sparsity_jacobian()

    row = []
    col = []

    for t = 1:T-1
        for i = 1:N
            if t > 1
                r_idx = idx_con_dyn[i][t]
                c_idx = idx_x_vec[t-1]
                row_col!(row,col,r_idx,c_idx)
                # ∇c[idx_con_dyn[i][t],idx_x_vec[t-1]] = view(∇tmp_x,(i-1)*n .+ (1:n),1:n*N)
            end
            r_idx = idx_con_dyn[i][t]
            c_idx = idx_u_vec[t]
            row_col!(row,col,r_idx,c_idx)
            # ∇c[idx_con_dyn[i][t],idx_u_vec[t]] = view(∇tmp_u,(i-1)*n .+ (1:n),1:m*N)

            r_idx = idx_con_dyn[i][t]
            c_idx = idx_x[i][t]
            row_col_cartesian!(row,col,r_idx,c_idx)
            # ∇c[CartesianIndex.(idx_con_dyn[i][t],idx_x[i][t])] .= -1.0

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_u[i][t]
            row_col_cartesian!(row,col,r_idx,c_idx)
            # ∇c[CartesianIndex.(idx_con_ctrl[i][t],idx_u[i][t])] .= 1.0

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_k[t]
            row_col!(row,col,r_idx,c_idx)
            # ∇c[idx_con_ctrl[i][t],idx_k[t]] = kron((xs[(i - 1)*n .+ (1:n)] - x_nom[t])',Im)

            if t > 1
                r_idx = idx_con_ctrl[i][t]
                c_idx = idx_x[i][t-1]
                row_col!(row,col,r_idx,c_idx)
                # ∇c[idx_con_ctrl[i][t],idx_x[i][t-1]] = k
            end
        end
    end

    return collect(zip(row,col))
end

z0 = rand(n_nlp)
zl = -Inf*ones(n_nlp)
zu = Inf*ones(n_nlp)

for t = 1:T-1
    for i = 1:N
        z0[idx_x[i][t]] = copy(x_nom[t+1])
        z0[idx_u[i][t]] = copy(u_nom[t])

        zl[idx_u[i][t]] .= ul
        zu[idx_u[i][t]] .= uu
    end
end

prob = Problem(n_nlp,m_nlp,obj,∇obj!,con!,∇con_vec!,true,
    sparsity_jac=_sparsity_jacobian(),
    primal_bounds=(zl,zu))

z_sol_s = solve(z0,prob)

x_sol = [[z_sol_s[idx_x[i][t]] for t = 1:T-1] for i = 1:N]
u_sol = [[z_sol_s[idx_u[i][t]] for t = 1:T-1] for i = 1:N]

plot(hcat(x_sol[1]...)',color=:blue,label="")
plot!(hcat(x_sol[2]...)',color=:green,label="")
plot!(hcat(x_sol[3]...)',color=:red,label="")
plot!(hcat(x_sol[4]...)',color=:orange,label="")
plot!(hcat(x_sol[5]...)',color=:brown,label="")
plot!(hcat(x_sol[6]...)',color=:cyan,label="")
plot!(hcat(x_sol[7]...)',color=:magenta,label="")
plot!(hcat(x_sol[8]...)',color=:yellow,label="")

plot(hcat(u_sol[1]...)',color=:blue,label="")
plot!(hcat(u_sol[2]...)',color=:green,label="")
plot!(hcat(u_sol[3]...)',color=:red,label="")
plot!(hcat(u_sol[4]...)',color=:orange,label="")
plot!(hcat(u_sol[5]...)',color=:brown,label="")
plot!(hcat(u_sol[6]...)',color=:cyan,label="")
plot!(hcat(u_sol[7]...)',color=:magenta,label="")
plot!(hcat(u_sol[8]...)',color=:yellow,label="")

K_sample = [reshape(z_sol_s[idx_k[t]],m,n) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
# println("solution error: $(sum(K_error)/N)")

# plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,width=2.0,label="β=$β",title="Gain matrix error")

# model_unc = Cartpole(0.965,0.2,0.5,9.81)
model_unc = Cartpole(1.0,0.165,0.5,9.81)

model_sim = model_unc

T_sim = 10*T
μ = zeros(n)
Σ = Diagonal(1.0e-3*ones(n))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(n)
Σ0 = Diagonal(1.0e-3*ones(n))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(x_nom[1]) + 1.0*w[:,1])

z_nom_sim, u_nom_sim = nominal_trajectories(x_nom,u_nom,T_sim,Δt)
t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

plt = plot(t_nom,hcat(x_nom...)[1,:],title="States",legend=:bottom,color=:red,label="ref.",width=2.0,xlabel="time (s)")
plt = plot!(t_nom,hcat(x_nom...)[2,:],color=:red,label="",width=2.0)
plt = plot!(t_nom,hcat(x_nom...)[3,:],color=:red,label="",width=2.0)
plt = plot!(t_nom,hcat(x_nom...)[4,:],color=:red,label="",width=2.0)

z_tvlqr, u_tvlqr, J_tvlqr = simulate_linear_controller(K,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,ul=ul*ones(m),uu=uu*ones(m))
plt = plot!(t_sim,hcat(z_tvlqr...)[1,:],color=:purple,label="tvlqr",width=2.0)
plt = plot!(t_sim,hcat(z_tvlqr...)[2,:],color=:purple,label="",width=2.0)
plt = plot!(t_sim,hcat(z_tvlqr...)[3,:],color=:purple,label="",width=2.0)
plt = plot!(t_sim,hcat(z_tvlqr...)[4,:],color=:purple,label="",width=2.0)

z_sample, u_sample, J_sample = simulate_linear_controller(K_sample,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,ul=ul*ones(m),uu=uu*ones(m))
plt = plot!(t_sim,hcat(z_sample...)[1,:],color=:orange,label="sample",width=2.0)
plt = plot!(t_sim,hcat(z_sample...)[2,:],color=:orange,label="",width=2.0)
plt = plot!(t_sim,hcat(z_sample...)[3,:],color=:orange,label="",width=2.0)
plt = plot!(t_sim,hcat(z_sample...)[4,:],color=:orange,label="",width=2.0)

# plot(t_nom[1:end-1],vcat(K...)[:,1],xlabel="time (s)",title="Gains",label="tvlqr",width=2.0,color=:purple,linetype=:steppost)
# plot!(t_nom[1:end-1],vcat(K...)[:,2],label="",width=2.0,color=:purple,linetype=:steppost)

# plot!(t_nom[1:end-1],vcat(K_sample...)[:,1],legend=:bottom,label="sample",color=:orange,width=2.0,linetype=:steppost)
# plot!(t_nom[1:end-1],vcat(K_sample...)[:,2],label="",color=:orange,width=2.0,linetype=:steppost)

plot(t_sim[1:end-1],hcat(u_nom_sim...)[:],title="Controls",xlabel="time (s)",legend=:bottom,color=:red,label="ref.",linetype=:steppost)
plot!(t_sim[1:end-1],hcat(u_tvlqr...)[:],color=:purple,label="tvlqr",linetype=:steppost)
plot!(t_sim[1:end-1],hcat(u_sample...)[:],color=:orange,label="sample",linetype=:steppost)

# objective value
J_tvlqr
J_sample
