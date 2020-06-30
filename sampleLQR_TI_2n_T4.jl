using LinearAlgebra, ForwardDiff
include("ipopt.jl")

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

# TVLQR solution
T = 4
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

    xtraj1[t+1] = A*xtraj1[t] + B*utraj1[t]
    xtraj2[t+1] = A*xtraj2[t] + B*utraj2[t]
    xtraj3[t+1] = A*xtraj3[t] + B*utraj3[t]
    xtraj4[t+1] = A*xtraj4[t] + B*utraj4[t]
    xtraj_nom[t+1] = A*xtraj_nom[t] + B*utraj_nom[t]
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

z0_nom[5:6] .= 0.0#K[1]

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

z0_nom[33:34] .= 0.0#K[3]

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

    return (u11'*R*u11 + u12'*R*u12 + u13'*R*u13 + u14'*R*u14
            + u21'*R*u21 + u22'*R*u22 + u23'*R*u23 + u24'*R*u24
            + u31'*R*u31 + u32'*R*u32 + u33'*R*u33 + u34'*R*u34
            + x21'*Q*x21 + x22'*Q*x22 + x23'*Q*x23 + x24'*Q*x24
            + x31'*Q*x31 + x32'*Q*x32 + x33'*Q*x33 + x34'*Q*x34
            + x41'*Q*x41 + x42'*Q*x42 + x43'*Q*x43 + x44'*Q*x44)
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

    c[1:2] = A*x11 + B*u11 - x21
    c[3:4] = A*x12 + B*u12 - x22
    c[5:6] = A*x13 + B*u13 - x23
    c[7:8] = A*x14 + B*u14 - x24

    c[9:10] = A*x21 + B*u21 - x31
    c[11:12] = A*x22 + B*u22 - x32
    c[13:14] = A*x23 + B*u23 - x33
    c[15:16] = A*x24 + B*u24 - x34

    c[17:18] = A*x31 + B*u31 - x41
    c[19:20] = A*x32 + B*u32 - x42
    c[21:22] = A*x33 + B*u33 - x43
    c[23:24] = A*x34 + B*u34 - x44

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

println("solution error: $(norm(z_sol - z0))")
