using LinearAlgebra, ForwardDiff, Distributions, Plots, StaticArrays, SparseArrays
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
    for k = 1:5
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

    f::Function
end

function dyn_c(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,3:4) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

model_nom = Cartpole(1.0,0.2,0.5,9.81,dyn_c)
n, m = 4,1

# Cartpole discrete-time dynamics (midpoint)
Δt = 0.1
function dynamics(model,x,u,Δt)
    rk3(model,x,u,Δt)
end

# Trajectory optimization
T = 20
x1_nom = zeros(n)
xT_nom = [0.0;π;0.0;0.0]

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
x_nom_ref = linear_interp(x1_nom,xT_nom,T)

Q_nom = [t!=T ? Diagonal(1.0*@SVector [1.0,1.0,1.0,1.0]) : Diagonal(1.0*@SVector ones(n)) for t = 1:T]
R_nom = [Diagonal(1.0e-1*@SVector ones(m)) for t = 1:T-1]

x_nom_idx = [(t-1)*(n+m) .+ (1:n) for t = 1:T]
u_nom_idx = [(t-1)*(n+m) + n .+ (1:m) for t = 1:T-1]

n_nom_nlp = n*T + m*(T-1)
m_nom_nlp = n*(T+1)

zl_nom = -Inf*ones(n_nom_nlp)
zu_nom = Inf*ones(n_nom_nlp)

z0 = 1.0e-5*randn(n_nom_nlp)
for t = 1:T
    z0[x_nom_idx[t]] = x_nom_ref[t]

    if t < T
        zl_nom[u_nom_idx[t]] .= ul
        zu_nom[u_nom_idx[t]] .= uu
    end
end

function obj_nom(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        s += (x-xT_nom)'*Q_nom[t]*(x-xT_nom) + u'*R_nom[t]*u
    end
    x = z[x_nom_idx[T]]
    s += (x-xT_nom)'*Q_nom[T]*(x-xT_nom)

    return s
end

obj_nom(z0)

function ∇obj_nom!(g,z)
    g .= 0.0
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        g[x_nom_idx[t]] += 2.0*Q_nom[t]*(x-xT_nom)
        g[u_nom_idx[t]] += 2.0*R_nom[t]*u
    end
    x = z[x_nom_idx[T]]
    g[x_nom_idx[T]] += 2.0*Q_nom[T]*(x-xT_nom)

    return g
end

g_nom = zeros(n_nom_nlp)
∇obj_nom!(g_nom,z0)
@assert norm(g_nom - ForwardDiff.gradient(obj_nom,z0)) < 1.0e-14

#TODO confirm

# Constraints
function con_nom!(c,z)
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        x⁺ = z[x_nom_idx[t+1]]
        c[(t-1)*n .+ (1:n)] = x⁺ - dynamics(model_nom,x,u,Δt)
    end
    c[(T-1)*n .+ (1:n)] = z[x_nom_idx[1]] - x1_nom
    c[T*n .+ (1:n)] = z[x_nom_idx[T]] - xT_nom
    return c
end

c0 = zeros(m_nom_nlp)
con_nom!(c0,z0)

# Constraints
function ∇con_nom_vec!(c,z)
    shift = 0
    for t = 1:T-1
        x = z[x_nom_idx[t]]
        u = z[u_nom_idx[t]]
        x⁺ = z[x_nom_idx[t+1]]
        # c[(t-1)*n .+ (1:n)] = x⁺ - dynamics(model_nom,x,u,Δt)

        dynx(w) = x⁺ - dynamics(model_nom,w,u,Δt)
        dynu(w) = x⁺ - dynamics(model_nom,x,w,Δt)
        dynx⁺(w) = w - dynamics(model_nom,x,u,Δt)

        r_idx = (t-1)*n .+ (1:n)
        c_idx = x_nom_idx[t]
        len = length(r_idx)*length(c_idx)
        c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(dynx,x))
        shift += len

        r_idx = (t-1)*n .+ (1:n)
        c_idx = u_nom_idx[t]
        len = length(r_idx)*length(c_idx)
        c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(dynu,u))
        shift += len

        r_idx = (t-1)*n .+ (1:n)
        c_idx = x_nom_idx[t+1]
        len = length(r_idx)*length(c_idx)
        c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(dynx⁺,x⁺))
        shift += len

    end
    # c[(T-1)*n .+ (1:n)] = z[x_nom_idx[1]] - x1

    r_idx = (T-1)*n .+ (1:n)
    c_idx = x_nom_idx[1]
    len = length(r_idx)*length(c_idx)
    c[shift .+ (1:len)] = vec(Diagonal(ones(n)))
    shift += len

    # c[T*n .+ (1:n)] = z[x_nom_idx[T]] - xT

    r_idx = T*n .+ (1:n)
    c_idx = x_nom_idx[T]
    len = length(r_idx)*length(c_idx)
    c[shift .+ (1:len)] = vec(Diagonal(ones(n)))
    shift += len
    return nothing
end

function _sparsity_jacobian_nom(;r_shift=0,c_shift=0)
    row = []
    col = []

    for t = 1:T-1

        # c[(t-1)*n .+ (1:n)] = x⁺ - dynamics(model_nom,x,u,Δt)

        r_idx = r_shift + (t-1)*n .+ (1:n)
        c_idx = c_shift .+ x_nom_idx[t]
        row_col!(row,col,r_idx,c_idx)

        r_idx = r_shift + (t-1)*n .+ (1:n)
        c_idx = c_shift .+ u_nom_idx[t]
        row_col!(row,col,r_idx,c_idx)

        r_idx = r_shift + (t-1)*n .+ (1:n)
        c_idx = c_shift .+ x_nom_idx[t+1]
        row_col!(row,col,r_idx,c_idx)

    end
    # c[(T-1)*n .+ (1:n)] = z[x_nom_idx[1]] - x1

    r_idx = r_shift + (T-1)*n .+ (1:n)
    c_idx = c_shift .+ x_nom_idx[1]
    row_col!(row,col,r_idx,c_idx)

    # c[T*n .+ (1:n)] = z[x_nom_idx[T]] - xT

    r_idx = r_shift + T*n .+ (1:n)
    c_idx = c_shift .+ x_nom_idx[T]
    row_col!(row,col,r_idx,c_idx)

    return collect(zip(row,col))
end

sparsity_nom = _sparsity_jacobian_nom()
∇c_nom_vec = zeros(length(sparsity_nom))
∇con_nom_vec!(∇c_nom_vec,z0)
∇c_nom = spzeros(m_nom_nlp,n_nom_nlp)
for (i,idx) in enumerate(sparsity_nom)
    ∇c_nom[idx[1],idx[2]] = ∇c_nom_vec[i]
end

@assert (norm(vec(∇c_nom)-vec(ForwardDiff.jacobian(con_nom!,zeros(m_nom_nlp),z0))) < 1.0e-14)

prob = Problem(n_nom_nlp,m_nom_nlp,obj_nom,∇obj_nom!,con_nom!,∇con_nom_vec!,false,
    sparsity_jac=_sparsity_jacobian_nom(),
    primal_bounds=(zl_nom,zu_nom))

# Solve
z_sol = solve(z0,prob)

x_nom = [z_sol[x_nom_idx[t]] for t = 1:T]
u_nom = [z_sol[u_nom_idx[t]] for t = 1:T-1]

plot(hcat(x_nom...)',xlabel="time step",ylabel="state",label=["x" "θ" "dx" "dθ"],width=2.0,legend=:topleft)
plot(hcat(u_nom...)',linetype=:steppost,xlabel="time step",ylabel="control",label="",width=2.0)

# TVLQR solution
A = []
B = []
for t = 1:T-1
    x = x_nom[t]
    u = u_nom[t]
    fx(z) = dynamics(model_nom,z,u,Δt)
    fu(z) = dynamics(model_nom,x,z,Δt)

    push!(A,ForwardDiff.jacobian(fx,x))
    push!(B,ForwardDiff.jacobian(fu,u))
end

Q = [t < T ? Diagonal(1.0*@SVector [10.0,10.0,1.0,1.0]) : Diagonal([100.0;100.0;100.0;100.0]) for t = 1:T]
R = [Diagonal(1.0e-1*@SVector ones(m)) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]

P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

α = 1.0e-5
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

model1 = Cartpole(1.0,0.18,0.5,9.81,dyn_c)
model2 = Cartpole(1.0,0.185,0.5,9.81,dyn_c)
model3 = Cartpole(1.0,0.19,0.5,9.81,dyn_c)
model4 = Cartpole(1.0,0.195,0.5,9.81,dyn_c)
model5 = Cartpole(1.0,0.205,0.5,9.81,dyn_c)
model6 = Cartpole(1.0,0.21,0.5,9.81,dyn_c)
model7 = Cartpole(1.0,0.215,0.5,9.81,dyn_c)
model8 = Cartpole(1.0,0.22,0.5,9.81,dyn_c)

models_mass = [model_nom for i = 1:N]
# models_mass = [model1,model2,model3,model4,model5,model6,model7,model8]

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1) + n_nom_nlp
m_nlp = N*(n*(T-1) + m*(T-1)) + m_nom_nlp

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

idx_x_vec = [vcat([idx_x[i][t] for i = 1:N]...) for t = 1:T-1]
idx_u_vec = [vcat([idx_u[i][t] for i = 1:N]...) for t = 1:T-1]

idx_x_nom = [N*(n*(T-1) + m*(T-1)) + m*n*(T-1) .+ x_nom_idx[t] for t = 1:T]
idx_u_nom = [N*(n*(T-1) + m*(T-1)) + m*n*(T-1) .+ u_nom_idx[t] for t = 1:T-1]

idx_z_nom = N*(n*(T-1) + m*(T-1)) + m*n*(T-1) .+ (1:n_nom_nlp)
idx_c_nom = N*(n*(T-1) + m*(T-1)) .+ (1:m_nom_nlp)

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

function resample_vec(X; β=1.0,w=1.0)
    xμ = sum([X[(i-1)*n .+ (1:n)] for i = 1:N])./N
    Σμ = (0.5/(β^2))*sum([(X[(i-1)*n .+ (1:n)] - xμ)*(X[(i-1)*n .+ (1:n)] - xμ)' for i = 1:N]) + w*I
    cols = fastsqrt(Σμ)
    Xs = vcat([xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]...)
    return Xs
end

function sample_nonlinear_dynamics_vec(models,X,U; β=1.0,w=1.0)
    # X⁺ = zeros(eltype(X),X)
    # for i = 1:n
    #     X⁺[(i-1)*n .+ (1:n)] = dynamics(models[i],view(X,(i-1)*n .+ (1:n)),view(U,(i-1)*m .+ (1:m)),Δt)
    # end
    X⁺ = vcat([dynamics(models[i],X[(i-1)*n .+ (1:n)],U[(i-1)*m .+ (1:m)],Δt) for i = 1:N]...)
    # return X⁺
    Xs⁺ = resample_vec(X⁺,β=β,w=w)
    return Xs⁺
end

γ = 1.0
function obj(z)
    s = 0.0
    for t = 1:T-1
        x⁺_nom = z[idx_x_nom[t+1]]
        u_nom = view(z,idx_u_nom[t])
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            s += (x - x⁺_nom)'*Q[t+1]*(x - x⁺_nom) + (u - u_nom)'*R[t]*(u - u_nom)
        end
    end
    return γ*s/N + obj_nom(view(z,idx_z_nom))
end

obj(rand(n_nlp))

function ∇obj!(g,z)
    g .= 0.0
    ∇obj_nom!(view(g,idx_z_nom),view(z,idx_z_nom))

    for t = 1:T-1
        x⁺_nom = z[idx_x_nom[t+1]]
        u_nom = view(z,idx_u_nom[t])
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])

            g[idx_x[i][t]] += 2.0*Q[t+1]*(x - x⁺_nom)*γ/N
            g[idx_u[i][t]] += 2.0*R[t]*(u - u_nom)γ/N

            g[idx_x_nom[t+1]] -= 2.0*Q[t+1]*(x - x⁺_nom)*γ/N
            g[idx_u_nom[t]] -= 2.0*R[t]*(u - u_nom)*γ/N

        end
    end
    return nothing
end

x0 = rand(n_nlp)
grad_f = zeros(n_nlp)
∇obj!(grad_f,x0)
@assert norm(grad_f - ForwardDiff.gradient(obj,x0)) < 1.0e-14

function con!(c,z)
    c .= 0.0
    β = 1.0
    w = 1.0e-5
    for t = 1:T-1
        xs = (t==1 ? [x1[i] for i = 1:N] : [view(z,idx_x[i][t-1]) for i = 1:N])
        u = [view(z,idx_u[i][t]) for i = 1:N]
        xs⁺ = sample_nonlinear_dynamics(models_mass,xs,u,β=β,w=w)
        x⁺ = [view(z,idx_x[i][t]) for i = 1:N]
        k = reshape(view(z,idx_k[t]),m,n)

        x_nom = view(z,idx_x_nom[t])
        u_nom = view(z,idx_u_nom[t])

        for i = 1:N
            c[idx_con_dyn[i][t]] = xs⁺[i] - x⁺[i]
            c[idx_con_ctrl[i][t]] = u[i] + k*(xs[i] - x_nom) - u_nom
        end
    end
    con_nom!(view(c,idx_c_nom),view(z,idx_z_nom))
    return c
end
idx_c_nom
con!(zeros(m_nlp),rand(n_nlp))
zeros(m_nlp)[idx_c_nom]

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

            # u nom
            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_u_nom[t]
            row_col!(row,col,r_idx,c_idx)
            # x nom
            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_x_nom[t]
            row_col!(row,col,r_idx,c_idx)
        end
    end

    return collect(zip(row,col))
end

len_jac = length(_sparsity_jacobian())
len_jac_nom = length(_sparsity_jacobian_nom())

function ∇con_vec!(∇c,z)
    ∇c .= 0.0

    β = 1.0
    w = 1.0e-5

    Im = Matrix(I,m,m)
    ∇tmp_x = zeros(n*N,n*N)
    ∇tmp_u = zeros(n*N,m*N)

    shift = 0
    for t = 1:T-1
        x_nom = view(z,idx_x_nom[t])
        u_nom = view(z,idx_u_nom[t])
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
            ∇c[shift .+ (1:len)] = vec(kron((xs[(i - 1)*n .+ (1:n)] - x_nom)',Im))
            shift += len

            if t > 1
                r_idx = idx_con_ctrl[i][t]
                c_idx = idx_x[i][t-1]
                len = length(r_idx)*length(c_idx)
                ∇c[shift .+ (1:len)] = vec(k)
                shift += len
            end

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_u_nom[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(m)))
            shift += len

            r_idx = idx_con_ctrl[i][t]
            c_idx = idx_x_nom[t]
            len = length(r_idx)*length(c_idx)
            ∇c[shift .+ (1:len)] = vec(-1.0*k)
            shift += len
        end
    end
    ∇con_nom_vec!(view(∇c,len_jac .+ (1:len_jac_nom)),view(z,idx_z_nom))

    return ∇c
end
len_jac
len_jac_nom

idx_z_nom
sparsity = collect([_sparsity_jacobian()...,_sparsity_jacobian_nom(r_shift=(N*(n*(T-1) + m*(T-1))),c_shift=(N*(n*(T-1) + m*(T-1)) + m*n*(T-1)))...])

∇c = spzeros(m_nlp,n_nlp)
∇c_jac = zeros(length(sparsity))
∇con_vec!(∇c_jac,x0)
∇c_fd = ForwardDiff.jacobian(con!,zeros(m_nlp),x0)
for (i,k) in enumerate(sparsity)
    # println("$i")
    ∇c[k[1],k[2]] = ∇c_jac[i]
end
@assert norm(vec(∇c) - vec(∇c_fd)) < 1.0e-12

z0 = rand(n_nlp)
zl = -Inf*ones(n_nlp)
zu = Inf*ones(n_nlp)

for t = 1:T-1
    z0[idx_x_nom[t]] = copy(x_nom[t])
    z0[idx_u_nom[t]] = copy(u_nom[t])

    zl[idx_u_nom[t]] .= ul
    zu[idx_u_nom[t]] .= uu

    z0[idx_k[t]] = vec(K[t])

    for i = 1:N
        z0[idx_x[i][t]] = copy(x_nom[t+1])
        z0[idx_u[i][t]] = copy(u_nom[t])

        zl[idx_u[i][t]] .= ul
        zu[idx_u[i][t]] .= uu

    end
    z0[idx_x_nom[T]] = copy(x_nom[T])
end

prob = Problem(n_nlp,m_nlp,obj,∇obj!,con!,∇con_vec!,false,
    sparsity_jac=sparsity,
    primal_bounds=([zl;zl_nom],[zu;zu_nom]))

z_sol_s = solve(z0,prob,max_iter=1000)
# z_sol_s = z0
x_sol = [[z_sol_s[idx_x[i][t]] for t = 1:T-1] for i = 1:N]
u_sol = [[z_sol_s[idx_u[i][t]] for t = 1:T-1] for i = 1:N]
x_sol_nom = [z_sol_s[idx_x_nom[t]] for t = 1:T]
u_sol_nom = [z_sol_s[idx_u_nom[t]] for t = 1:T-1]

plot(hcat(x1[1],x_sol[1]...)',color=:blue,label="")
plot!(hcat(x1[2],x_sol[2]...)',color=:green,label="")
plot!(hcat(x1[3],x_sol[3]...)',color=:purple,label="")
plot!(hcat(x1[4],x_sol[4]...)',color=:orange,label="")
plot!(hcat(x1[5],x_sol[5]...)',color=:brown,label="")
plot!(hcat(x1[6],x_sol[6]...)',color=:cyan,label="")
plot!(hcat(x1[7],x_sol[7]...)',color=:magenta,label="")
plot!(hcat(x1[8],x_sol[8]...)',color=:yellow,label="")

plot!(hcat(x_sol_nom...)',color=:red,label="",width=2.0)
plot!(hcat(x_nom...)',color=:black,label="",width=2.0)

norm(vec(hcat(x_sol_nom...)) - vec(hcat(x_nom...)))

plot(hcat(u_sol[1]...)',color=:blue,label="",linetype=:steppost)
plot!(hcat(u_sol[2]...)',color=:green,label="",linetype=:steppost)
plot!(hcat(u_sol[3]...)',color=:purple,label="",linetype=:steppost)
plot!(hcat(u_sol[4]...)',color=:orange,label="",linetype=:steppost)
plot!(hcat(u_sol[5]...)',color=:brown,label="",linetype=:steppost)
plot!(hcat(u_sol[6]...)',color=:cyan,label="",linetype=:steppost)
plot!(hcat(u_sol[7]...)',color=:magenta,label="",linetype=:steppost)
plot!(hcat(u_sol[8]...)',color=:yellow,label="",linetype=:steppost)
plot!(hcat(u_sol_nom...)',color=:red,label="",linetype=:steppost,width=2.0)
plot!(hcat(u_nom...)',color=:black,label="",linetype=:steppost,width=2.0)

K_sample = [reshape(z_sol_s[idx_k[t]],m,n) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
# println("solution error: $(sum(K_error)/N)")

# plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,width=2.0,label="β=$β",title="Gain matrix error")

model_unc = Cartpole(1.0,0.2,0.5,9.81,dyn_c)

model_sim = model_unc

T_sim = 100T
μ = zeros(n)
Σ = Diagonal(1.0e-3*ones(n))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(n)
Σ0 = Diagonal(1.0e-3*ones(n))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(x_nom[1]) + 1.0*w0[:,1])

z_nom_sim, u_nom_sim = nominal_trajectories(x_nom,u_nom,T_sim,Δt)
z_nom_sol_sim, u_nom_sol_sim = nominal_trajectories(x_sol_nom,u_sol_nom,T_sim,Δt)

t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

plt1 = plot(t_nom,hcat(x_nom...)[1,:],title="States",legend=:topleft,color=:red,label="nominal",width=2.0,xlabel="time (s)")
plt1 = plot!(t_nom,hcat(x_nom...)[2,:],color=:red,label="",width=2.0)
# plt1 = plot!(t_nom,hcat(x_nom...)[3,:],color=:red,label="",width=2.0)
# plt1 = plot!(t_nom,hcat(x_nom...)[4,:],color=:red,label="",width=2.0)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w)#,ul=ul*ones(m),uu=uu*ones(m))
plt1 = plot!(t_sim,hcat(z_tvlqr...)[1,:],color=:purple,label="tvlqr",width=2.0)
plt1 = plot!(t_sim,hcat(z_tvlqr...)[2,:],color=:purple,label="",width=2.0)
# plt1 = plot!(t_sim,hcat(z_tvlqr...)[3,:],color=:purple,label="",width=2.0)
# plt1 = plot!(t_sim,hcat(z_tvlqr...)[4,:],color=:purple,label="",width=2.0)

z_tvlqr_bnds, u_tvlqr_bnds, J_tvlqr_bnds, Jx_tvlqr_bnds, Ju_tvlqr_bnds = simulate_linear_controller(K,x_nom,u_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,ul=ul*ones(m),uu=uu*ones(m))
plt1 = plot!(t_sim,hcat(z_tvlqr_bnds...)[1,:],color=:cyan,label="tvlqr (bnds)",width=2.0)
plt1 = plot!(t_sim,hcat(z_tvlqr_bnds...)[2,:],color=:cyan,label="",width=2.0)
# plt1 = plot!(t_sim,hcat(z_tvlqr...)[3,:],color=:purple,label="",width=2.0)
# plt1 = plot!(t_sim,hcat(z_tvlqr...)[4,:],color=:purple,label="",width=2.0)

plt2 = plot(t_nom,hcat(x_sol_nom...)[1,:],legend=:topleft,color=:red,label="nominal (sample)",width=2.0,xlabel="time (s)")
plt2 = plot!(t_nom,hcat(x_sol_nom...)[2,:],color=:red,label="",width=2.0)
z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_linear_controller(K_sample,x_sol_nom,u_sol_nom,model_sim,Q,R,T_sim,Δt,z0_sim,w,ul=ul*ones(m),uu=uu*ones(m))
plt2 = plot!(t_sim,hcat(z_sample...)[1,:],color=:orange,label="sample (bnds)",width=2.0)
plt2 = plot!(t_sim,hcat(z_sample...)[2,:],color=:orange,label="",width=2.0)
# plt2 = plot!(t_sim,hcat(z_sample...)[3,:],color=:orange,label="",width=2.0)
# plt2 = plot!(t_sim,hcat(z_sample...)[4,:],color=:orange,label="",width=2.0)


plot(plt1,plt2,layout=(2,1))
# plot(t_nom[1:end-1],vcat(K...)[:,1],xlabel="time (s)",title="Gains",label="tvlqr",width=2.0,color=:purple,linetype=:steppost)
# plot!(t_nom[1:end-1],vcat(K...)[:,2],label="",width=2.0,color=:purple,linetype=:steppost)

# plot!(t_nom[1:end-1],vcat(K_sample...)[:,1],legend=:bottom,label="sample",color=:orange,width=2.0,linetype=:steppost)
# plot!(t_nom[1:end-1],vcat(K_sample...)[:,2],label="",color=:orange,width=2.0,linetype=:steppost)

plt3 = plot(t_sim[1:end-1],hcat(u_nom_sim...)[:],title="Controls",xlabel="time (s)",legend=:bottomright,color=:red,label="nominal",linetype=:steppost)
plt3 = plot!(t_sim[1:end-1],hcat(u_tvlqr...)[:],color=:cyan,label="tvlqr",linetype=:steppost)
plt3 = plot!(t_sim[1:end-1],hcat(u_tvlqr_bnds...)[:],color=:purple,label="tvlqr (bnds)",linetype=:steppost)

plt4 = plot(t_sim[1:end-1],hcat(u_nom_sol_sim...)[:],xlabel="time (s)",legend=:bottomright,color=:red,label="nominal (sample)",linetype=:steppost)
plt4 = plot!(t_sim[1:end-1],hcat(u_sample...)[:],color=:orange,label="sample (bnds)",linetype=:steppost)

plot(plt3,plt4,layout=(2,1))

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
