using LinearAlgebra, ForwardDiff, Plots
include("ipopt.jl")

# Pendulum continuous-time dynamics
n = 2
m = 1
function dyn_c(x,u)
    m = 1.0
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    return [x[2];(u[1] - m*g*lc*sin(x[1]) - b*x[2])/I]
end

# Pendulum discrete-time dynamics (midpoint)
Δt = 0.1
function dyn_d(x,u,Δt)
    x + Δt*dyn_c(x + 0.5*Δt*dyn_c(x,u),u)
end

dyn_c(rand(n),rand(m))
dyn_d(rand(n),rand(m),0.2)

# Trajectory optimization
T = 3
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

function obj(z)
    s = 0.0
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        # s += (x-xT)'*Q[t]*(x-xT) + u'*R[t]*u
        s += x'*Q[t]*x + u'*R[t]*u

    end
    x = z[x_idx[T]]
    s += (x-xT)'*Q[T]*(x-xT)

    return s
end

obj(z0)

# Constraints
function con!(c,z)
    for t = 1:T-1
        x = z[x_idx[t]]
        u = z[u_idx[t]]
        x⁺ = z[x_idx[t+1]]
        c[(t-1)*n .+ (1:n)] = x⁺ - dyn_d(x,u,Δt)
    end
    c[(T-1)*n .+ (1:n)] = z[x_idx[1]] - x1
    c[T*n .+ (1:n)] = z[x_idx[T]] - xT
    return c
end

c0 = zeros(m_nlp)
con!(c0,z0)

# NLP problem
prob = Problem(n_nlp,m_nlp,obj,con!,true)

# Solve
z_sol = solve(z0,prob)

x_sol = [z_sol[x_idx[t]] for t = 1:T]
u_sol = [z_sol[u_idx[t]] for t = 1:T-1]

plot(hcat(x_sol...)',xlabel="time step",ylabel="state",label=["θ" "dθ"],width=2.0,legend=:topleft)
plot(hcat(u_sol...)',xlabel="time step",ylabel="control",label="",width=2.0)

# TVLQR solution
A = []
B = []
for t = 1:T-1
    x = x_sol[t]
    u = u_sol[t]
    fx(z) = dyn_d(z,u,Δt)
    fu(z) = dyn_d(x,z,Δt)

    push!(A,ForwardDiff.jacobian(fx,x))
    push!(B,ForwardDiff.jacobian(fu,u))
end

Q = [Matrix((1.0 + 1.0e-1*t)*I,n,n) for t = 1:T]
R = [Matrix((0.1 + 1.0e-1*t)*I,m,m) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

# number of samples
N = 4
# N = 4

# initial state
x11 = [1.0; 0.0]
x12 = [-1.0; 0.0]
x13 = [0.0; 1.0]
x14 = [0.0; -1.0]
# x15 = [1.0; 1.0]
# x16 = [1.0; -1.0]
# x17 = [-1.0; 1.0]
# x18 = [-1.0; -1.0]
# x19 = 2.0*[1.0; 1.0]
# x110 = 2.0*[1.0; -1.0]
# x111 = 2.0*[-1.0; 1.0]
# x112 = 2.0*[-1.0; -1.0]
x1 = [x11,x12,x13,x14]#,x15,x16,x17,x18,x19,x110,x111,x112]

# simulate
xtraj = [[zeros(n) for t = 1:T] for i = 1:N]
xtraj_nom = [zeros(n) for t = 1:T]

xtraj[1][1] = x1[1]
xtraj[2][1] = x1[2]
xtraj[3][1] = x1[3]
xtraj[4][1] = x1[4]

utraj = [[zeros(m) for t = 1:T-1] for i = 1:N]
utraj_nom = [zeros(m) for t = 1:T-1]

for i = 1:N
    for t = 1:T-1
        utraj[i][t] = -K[t]*xtraj[i][t]
        xtraj[i][t+1] = A[t]*xtraj[i][t] + B[t]*utraj[i][t]
        if i == 1
            utraj_nom[t] = -K[t]*xtraj_nom[t]
            xtraj_nom[t+1] = A[t]*xtraj_nom[t] + B[t]*utraj_nom[t]
        end
    end
end

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1) + m*(T-1))
x_idx = [[(i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
u_idx = [[(i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]
K_idx = [N*(n*(T-1) + m*(T-1)) + (t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]

z0 = zeros(n_nlp)
for i = 1:N
    for t = 1:T-1
        z0[x_idx[i][t]] = xtraj[i][t]
        z0[u_idx[i][t]] = utraj[i][t]
        if i == 1
            z0[K_idx[t]] = vec(K[t])
        end
    end
end

z0_nom = zeros(n_nlp)
for i = 1:N
    for t = 1:T-1
        z0_nom[x_idx[i][t]] = xtraj_nom[t]
        z0[u_idx[i][t]] = utraj_nom[t]
    end
end

function obj(z)
    s = 0.0
    for i = 1:N
        for t = 1:T-1
            x = z[x_idx[i][t]]
            u = z[u_idx[i][t]]
            s += x'*Q[t+1]*x + u'*R[t]*u
        end
    end
    return s
end

obj(z0)

function con!(c,z)
    shift1 = 0
    shift2 = 0
    for i = 1:N
        for t = 1:T-1
            x = (t==1 ? xtraj[i][1] : z[x_idx[i][t-1]])
            u = z[u_idx[i][t]]
            x⁺ = z[x_idx[i][t]]
            k = reshape(z[K_idx[t]],m,n)

            c[shift1 .+ (1:n)] = A[t]*x + B[t]*u - x⁺
            shift1 += n

            c[N*(n*(T-1)) + shift2 .+ (1:m)] = u + k*x
            shift2 += m
        end
    end

    return c
end

c0 = zeros(m_nlp)
con!(c0,z0)

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_nom,prob)

K_sample = [reshape(z_sol[K_idx[t]],m,n) for t = 1:T-1]

println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1])/N)")
