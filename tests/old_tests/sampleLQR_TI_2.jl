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
T = 3
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
N = 2

# initial state
x11 = [1.0; 0.0]
x12 = [0.0; 1.0]

# simulate
xtraj1 = [zeros(n) for t = 1:T]
xtraj2 = [zeros(n) for t = 1:T]

xtraj1[1] = x11
xtraj2[1] = x12

utraj1 = [zeros(m) for t = 1:T-1]
utraj2 = [zeros(m) for t = 1:T-1]

for t = 1:T-1
    utraj1[t] = -K[t]*xtraj1[t]
    utraj2[t] = -K[t]*xtraj2[t]

    xtraj1[t+1] = A*xtraj1[t] + B*utraj1[t]
    xtraj2[t+1] = A*xtraj2[t] + B*utraj2[t]
end

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1)) + N*(m*(T-1))

z0 = zeros(n_nlp)

z0[1:1] = utraj1[1]
z0[2:2] = utraj2[1]
z0[3:4] = K[1]
z0[5:6] = xtraj1[2]
z0[7:8] = xtraj2[2]
z0[9:9] = utraj1[2]
z0[10:10] = utraj2[2]
z0[11:12] = K[2]
z0[13:14] = xtraj1[3]
z0[15:16] = xtraj2[3]

z0_ = zero(z0) #copy(z0)
# z0_[3:4] .= 0.0
# z0_[11:12] .= 0.0

function obj(z)
    u11 = z[1:1]
    u12 = z[2:2]
    k1 = z[3:4]
    x21 = z[5:6]
    x22 = z[7:8]
    u21 = z[9:9]
    u22 = z[10:10]
    k2 = z[11:12]
    x31 = z[13:14]
    x32 = z[15:16]

    return (u11'*R*u11 + u12'*R*u12 + u21'*R*u21 + u22'*R*u22
            + x21'*Q*x21 + x22'*Q*x22 + x31'*Q*x31 + x32'*Q*x32)
end

obj(z0)

function con!(c,z)
    u11 = z[1]
    u12 = z[2]
    k1 = z[3:4]
    x21 = z[5:6]
    x22 = z[7:8]
    u21 = z[9]
    u22 = z[10]
    k2 = z[11:12]
    x31 = z[13:14]
    x32 = z[15:16]

    c[1:2] = A*x11 + B*u11 - x21
    c[3:4] = A*x12 + B*u12 - x22
    c[5:6] = A*x21 + B*u21 - x31
    c[7:8] = A*x22 + B*u22 - x32
    c[9] = u11 + k1'*x11
    c[10] = u12 + k1'*x12
    c[11] = u21 + k2'*x21
    c[12] = u22 + k2'*x22
    return c
end

c0 = ones(m_nlp)
con!(c0,ones(n_nlp))

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_,prob)

println("solution error: $(norm(z_sol - z0))")
