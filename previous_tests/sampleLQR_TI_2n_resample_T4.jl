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

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1) + N*(n*(T-1) + m*(T-2))
m_nlp = N*(n*(T-1)) + N*(m*(T-1)) + N*(n*(T-1)) + N*(m*(T-2))

# z0 = zeros(n_nlp)
#
# z0[1:1] = utraj1[1]
# z0[2:2] = utraj2[1]
# z0[3:3] = utraj3[1]
# z0[4:4] = utraj4[1]
#
# z0[5:6] = K[1]
#
# z0[7:8] = xtraj1[2]
# z0[9:10] = xtraj2[2]
# z0[11:12] = xtraj3[2]
# z0[13:14] = xtraj4[2]
#
# z0[15:15] = utraj1[2]
# z0[16:16] = utraj2[2]
# z0[17:17] = utraj3[2]
# z0[18:18] = utraj4[2]
#
# z0[19:20] = K[2]
#
# z0[21:22] = xtraj1[3]
# z0[23:24] = xtraj2[3]
# z0[25:26] = xtraj3[3]
# z0[27:28] = xtraj4[3]
#

z0_nom = 0.1*rand(n_nlp)

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

    # # resampled points
    x̃21 = z[29:30]
    x̃22 = z[31:32]
    x̃23 = z[33:34]
    x̃24 = z[35:36]

    ũ21 = z[37:37]
    ũ22 = z[38:38]
    ũ23 = z[39:39]
    ũ24 = z[40:40]

    x̃31 = z[41:42]
    x̃32 = z[43:44]
    x̃33 = z[45:46]
    x̃34 = z[47:48]

    ##
    u31 = z[49:49]
    u32 = z[50:50]
    u33 = z[51:51]
    u34 = z[52:52]

    k3 = z[53:54]

    x41 = z[55:56]
    x42 = z[57:58]
    x43 = z[59:60]
    x44 = z[61:62]

    # resampled points

    ũ31 = z[63:63]
    ũ32 = z[64:64]
    ũ33 = z[65:65]
    ũ34 = z[66:66]

    x̃41 = z[67:68]
    x̃42 = z[69:70]
    x̃43 = z[71:72]
    x̃44 = z[73:74]

    return (u11'*R*u11 + u12'*R*u12 + u13'*R*u13 + u14'*R*u14
            + u21'*R*u21 + u22'*R*u22 + u23'*R*u23 + u24'*R*u24
            + u31'*R*u31 + u32'*R*u32 + u33'*R*u33 + u34'*R*u34
            + x21'*Q*x21 + x22'*Q*x22 + x23'*Q*x23 + x24'*Q*x24
            + x31'*Q*x31 + x32'*Q*x32 + x33'*Q*x33 + x34'*Q*x34
            + x41'*Q*x41 + x42'*Q*x42 + x43'*Q*x43 + x44'*Q*x44
            + ũ21'*R*ũ21 + ũ22'*R*ũ22 + ũ23'*R*ũ23 + ũ24'*R*ũ24
            + ũ31'*R*ũ31 + ũ32'*R*ũ32 + ũ33'*R*ũ33 + ũ34'*R*ũ34
            + x̃21'*Q*x̃21 + x̃22'*Q*x̃22 + x̃23'*Q*x̃23 + x̃24'*Q*x̃24
            + x̃31'*Q*x̃31 + x̃32'*Q*x̃32 + x̃33'*Q*x̃33 + x̃34'*Q*x̃34
            + x̃41'*Q*x̃41 + x̃42'*Q*x̃42 + x̃43'*Q*x̃43 + x̃44'*Q*x̃44)

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

    # resampled points
    x̃21 = z[29:30]
    x̃22 = z[31:32]
    x̃23 = z[33:34]
    x̃24 = z[35:36]

    ũ21 = z[37]
    ũ22 = z[38]
    ũ23 = z[39]
    ũ24 = z[40]

    x̃31 = z[41:42]
    x̃32 = z[43:44]
    x̃33 = z[45:46]
    x̃34 = z[47:48]

    u31 = z[49]
    u32 = z[50]
    u33 = z[51]
    u34 = z[52]

    k3 = z[53:54]

    x41 = z[55:56]
    x42 = z[57:58]
    x43 = z[59:60]
    x44 = z[61:62]

    # resampled points

    ũ31 = z[63]
    ũ32 = z[64]
    ũ33 = z[65]
    ũ34 = z[66]

    x̃41 = z[67:68]
    x̃42 = z[69:70]
    x̃43 = z[71:72]
    x̃44 = z[73:74]

    c[1:2] = A*x11 + B*u11 - x21
    c[3:4] = A*x12 + B*u12 - x22
    c[5:6] = A*x13 + B*u13 - x23
    c[7:8] = A*x14 + B*u14 - x24

    c[9:10] = A*x21 + B*u21 - x31
    c[11:12] = A*x22 + B*u22 - x32
    c[13:14] = A*x23 + B*u23 - x33
    c[15:16] = A*x24 + B*u24 - x34

    c[17] = u11 + k1'*x11
    c[18] = u12 + k1'*x12
    c[19] = u13 + k1'*x13
    c[20] = u14 + k1'*x14

    c[21] = u21 + k2'*x21
    c[22] = u22 + k2'*x22
    c[23] = u23 + k2'*x23
    c[24] = u24 + k2'*x24

    # resampled constraints
    Σ = x21*x21' + x22*x22' + x23*x23' + x24*x24' + 1.0e-8*I
    cols = Array(cholesky(Σ).U)
    β = 0.001

    c[25:26] = β*cols[:,1] - x̃21
    c[27:28] = β*cols[:,1] - x̃22
    c[29:30] = β*cols[:,2] - x̃23
    c[31:32] = β*cols[:,2] - x̃24

    c[33:34] = A*x̃21 + B*ũ21 - x̃31
    c[35:36] = A*x̃22 + B*ũ22 - x̃32
    c[37:38] = A*x̃23 + B*ũ23 - x̃33
    c[39:40] = A*x̃24 + B*ũ24 - x̃34

    c[41] = ũ21 + k2'*x̃21
    c[42] = ũ22 + k2'*x̃22
    c[43] = ũ23 + k2'*x̃23
    c[44] = ũ24 + k2'*x̃24

    ##
    c[45:46] = A*x31 + B*u31 - x41
    c[47:48] = A*x32 + B*u32 - x42
    c[49:50] = A*x33 + B*u33 - x43
    c[51:52] = A*x34 + B*u34 - x44

    c[53] = u31 + k3'*x31
    c[54] = u32 + k3'*x32
    c[55] = u33 + k3'*x33
    c[56] = u34 + k3'*x34

    c[57:58] = A*x̃31 + B*ũ31 - x̃41
    c[59:60] = A*x̃32 + B*ũ32 - x̃42
    c[61:62] = A*x̃33 + B*ũ33 - x̃43
    c[63:64] = A*x̃34 + B*ũ34 - x̃44

    c[65] = ũ31 + k3'*x̃31
    c[66] = ũ32 + k3'*x̃32
    c[67] = ũ33 + k3'*x̃33
    c[68] = ũ34 + k3'*x̃34

    return c
end

c0 = rand(m_nlp)
con!(c0,rand(n_nlp))

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_nom,prob)
K_sample = [reshape(z_sol[5:6],m,n), reshape(z_sol[19:20],m,n),reshape(z_sol[53:54],m,n)]
println("solution error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1]))")
