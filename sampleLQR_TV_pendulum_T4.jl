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
T = 4
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

A

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

# initial state
x11 = [1.0; 0.0]
x12 = [-1.0; 0.0]
x13 = [0.0; 1.0]
x14 = [0.0; -1.0]

# simulate
xtraj1 = [zeros(n) for t = 1:T]
xtraj2 = [zeros(n) for t = 1:T]
xtraj3 = [zeros(n) for t = 1:T]
xtraj4 = [zeros(n) for t = 1:T]
xtraj_nom = [zeros(n) for t = 1:T]

xtraj1[1] = x11
xtraj2[1] = x12
xtraj3[1] = x13
xtraj4[1] = x14

utraj1 = [zeros(m) for t = 1:T-1]
utraj2 = [zeros(m) for t = 1:T-1]
utraj3 = [zeros(m) for t = 1:T-1]
utraj4 = [zeros(m) for t = 1:T-1]
utraj_nom = [zeros(m) for t = 1:T-1]

for t = 1:T-1
    utraj1[t] = -K[t]*xtraj1[t]
    utraj2[t] = -K[t]*xtraj2[t]
    utraj3[t] = -K[t]*xtraj3[t]
    utraj4[t] = -K[t]*xtraj4[t]
    utraj_nom[t] = -K[t]*xtraj_nom[t]

    xtraj1[t+1] = A[t]*xtraj1[t] + B[t]*utraj1[t]
    xtraj2[t+1] = A[t]*xtraj2[t] + B[t]*utraj2[t]
    xtraj3[t+1] = A[t]*xtraj3[t] + B[t]*utraj3[t]
    xtraj4[t+1] = A[t]*xtraj4[t] + B[t]*utraj4[t]
    xtraj_nom[t+1] = A[t]*xtraj_nom[t] + B[t]*utraj_nom[t]
end

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1)) + N*(m*(T-1))

z0 = zeros(n_nlp)

z0[1:1] = utraj1[1]
z0[2:2] = utraj2[1]
z0[3:3] = utraj3[1]
z0[4:4] = utraj4[1]

z0[5:6] = K[1]

z0[7:8] = xtraj1[2]
z0[9:10] = xtraj2[2]
z0[11:12] = xtraj3[2]
z0[13:14] = xtraj4[2]

z0[15:15] = utraj1[2]
z0[16:16] = utraj2[2]
z0[17:17] = utraj3[2]
z0[18:18] = utraj4[2]

z0[19:20] = K[2]

z0[21:22] = xtraj1[3]
z0[23:24] = xtraj2[3]
z0[25:26] = xtraj3[3]
z0[27:28] = xtraj4[3]

z0[29:29] = utraj1[3]
z0[30:30] = utraj2[3]
z0[31:31] = utraj3[3]
z0[32:32] = utraj4[3]

z0[33:34] = K[3]

z0[35:36] = xtraj1[4]
z0[37:38] = xtraj2[4]
z0[39:40] = xtraj3[4]
z0[41:42] = xtraj4[4]

z0_nom = zeros(n_nlp)

z0_nom[1:1] = utraj_nom[1]
z0_nom[2:2] = utraj_nom[1]
z0_nom[3:3] = utraj_nom[1]
z0_nom[4:4] = utraj_nom[1]

z0_nom[5:6] .= 0.0 #K[1]

z0_nom[7:8] = xtraj_nom[2]
z0_nom[9:10] = xtraj_nom[2]
z0_nom[11:12] = xtraj_nom[2]
z0_nom[13:14] = xtraj_nom[2]

z0_nom[15:15] = utraj_nom[2]
z0_nom[16:16] = utraj_nom[2]
z0_nom[17:17] = utraj_nom[2]
z0_nom[18:18] = utraj_nom[2]

z0_nom[19:20] .= 0.0#K[2]

z0_nom[21:22] = xtraj_nom[3]
z0_nom[23:24] = xtraj_nom[3]
z0_nom[25:26] = xtraj_nom[3]
z0_nom[27:28] = xtraj_nom[3]

z0_nom[29:29] = utraj_nom[3]
z0_nom[30:30] = utraj_nom[3]
z0_nom[31:31] = utraj_nom[3]
z0_nom[32:32] = utraj_nom[3]

z0_nom[33:34] .= 0.0 #K[3]

z0_nom[35:36] = xtraj_nom[4]
z0_nom[37:38] = xtraj_nom[4]
z0_nom[39:40] = xtraj_nom[4]
z0_nom[41:42] = xtraj_nom[4]

function obj(z)
    u11 = z[1:1]
    u12 = z[2:2]
    u13 = z[3:3]
    u14 = z[4:4]

    # k1 = z[5:6]

    x21 = z[7:8]
    x22 = z[9:10]
    x23 = z[11:12]
    x24 = z[13:14]

    u21 = z[15:15]
    u22 = z[16:16]
    u23 = z[17:17]
    u24 = z[18:18]

    # k2 = z[19:20]

    x31 = z[21:22]
    x32 = z[23:24]
    x33 = z[25:26]
    x34 = z[27:28]

    u31 = z[29:29]
    u32 = z[30:30]
    u33 = z[31:31]
    u34 = z[32:32]

    # k3 = z[33:34]

    x41 = z[35:36]
    x42 = z[37:38]
    x43 = z[39:40]
    x44 = z[41:42]

    return (u11'*R[1]*u11 + u12'*R[1]*u12 + u13'*R[1]*u13 + u14'*R[1]*u14
            + u21'*R[2]*u21 + u22'*R[2]*u22 + u23'*R[2]*u23 + u24'*R[2]*u24
            + u31'*R[3]*u31 + u32'*R[3]*u32 + u33'*R[3]*u33 + u34'*R[3]*u34
            + x21'*Q[2]*x21 + x22'*Q[2]*x22 + x23'*Q[2]*x23 + x24'*Q[2]*x24
            + x31'*Q[3]*x31 + x32'*Q[3]*x32 + x33'*Q[3]*x33 + x34'*Q[3]*x34
            + x41'*Q[4]*x41 + x42'*Q[4]*x42 + x43'*Q[4]*x43 + x44'*Q[4]*x44)
end

obj(z0)

function con!(c,z)
    u11 = z[1]
    u12 = z[2]
    u13 = z[3]
    u14 = z[4]

    k1 = z[5:6]

    x21 = z[7:8]
    x22 = z[9:10]
    x23 = z[11:12]
    x24 = z[13:14]

    u21 = z[15]
    u22 = z[16]
    u23 = z[17]
    u24 = z[18]

    k2 = z[19:20]

    x31 = z[21:22]
    x32 = z[23:24]
    x33 = z[25:26]
    x34 = z[27:28]

    u31 = z[29]
    u32 = z[30]
    u33 = z[31]
    u34 = z[32]

    k3 = z[33:34]

    x41 = z[35:36]
    x42 = z[37:38]
    x43 = z[39:40]
    x44 = z[41:42]

    c[1:2] = A[1]*x11 + B[1]*u11 - x21
    c[3:4] = A[1]*x12 + B[1]*u12 - x22
    c[5:6] = A[1]*x13 + B[1]*u13 - x23
    c[7:8] = A[1]*x14 + B[1]*u14 - x24

    c[9:10] = A[2]*x21 + B[2]*u21 - x31
    c[11:12] = A[2]*x22 + B[2]*u22 - x32
    c[13:14] = A[2]*x23 + B[2]*u23 - x33
    c[15:16] = A[2]*x24 + B[2]*u24 - x34

    c[17:18] = A[3]*x31 + B[3]*u31 - x41
    c[19:20] = A[3]*x32 + B[3]*u32 - x42
    c[21:22] = A[3]*x33 + B[3]*u33 - x43
    c[23:24] = A[3]*x34 + B[3]*u34 - x44

    c[25] = u11 + k1'*x11
    c[26] = u12 + k1'*x12
    c[27] = u13 + k1'*x13
    c[28] = u14 + k1'*x14

    c[29] = u21 + k2'*x21
    c[30] = u22 + k2'*x22
    c[31] = u23 + k2'*x23
    c[32] = u24 + k2'*x24

    c[33] = u31 + k3'*x31
    c[34] = u32 + k3'*x32
    c[35] = u33 + k3'*x33
    c[36] = u34 + k3'*x34

    return c
end

c0 = zeros(m_nlp)
con!(c0,z0_nom)

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_nom,prob)

K_sample = [reshape(z_sol[5:6],m,n),
            reshape(z_sol[19:20],m,n),
            reshape(z_sol[33:34],m,n)]

println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1])/N)")
