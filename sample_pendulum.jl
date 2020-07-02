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
    utraj1[t] = u_sol[t] - K[t]*(xtraj1[t] - x_sol[t])
    utraj2[t] = u_sol[t] - K[t]*(xtraj2[t] - x_sol[t])
    utraj3[t] = u_sol[t] - K[t]*(xtraj3[t] - x_sol[t])
    utraj4[t] = u_sol[t] - K[t]*(xtraj4[t] - x_sol[t])
    utraj_nom[t] = u_sol[t] - K[t]*(xtraj_nom[t] - x_sol[t])

    xtraj1[t+1] = dyn_d(xtraj1[t],utraj1[t],Δt)
    xtraj2[t+1] = dyn_d(xtraj2[t],utraj2[t],Δt)
    xtraj3[t+1] = dyn_d(xtraj3[t],utraj3[t],Δt)
    xtraj4[t+1] = dyn_d(xtraj4[t],utraj4[t],Δt)
    xtraj_nom[t+1] = dyn_d(xtraj_nom[t],utraj_nom[t],Δt)
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

z0_nom = zeros(n_nlp)

z0_nom[1:1] = utraj_nom[1]
z0_nom[2:2] = utraj_nom[1]
z0_nom[3:3] = utraj_nom[1]
z0_nom[4:4] = utraj_nom[1]

z0_nom[5:6] = vec(K[1])

z0_nom[7:8] = xtraj_nom[2]
z0_nom[9:10] = xtraj_nom[2]
z0_nom[11:12] = xtraj_nom[2]
z0_nom[13:14] = xtraj_nom[2]

z0_nom[15:15] = utraj_nom[2]
z0_nom[16:16] = utraj_nom[2]
z0_nom[17:17] = utraj_nom[2]
z0_nom[18:18] = utraj_nom[2]

z0_nom[19:20] = vec(K[2])

z0_nom[21:22] = xtraj_nom[3]
z0_nom[23:24] = xtraj_nom[3]
z0_nom[25:26] = xtraj_nom[3]
z0_nom[27:28] = xtraj_nom[3]

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

    return (u11'*R[1]*u11 + u12'*R[1]*u12 + u13'*R[1]*u13 + u14'*R[1]*u14
            + u21'*R[2]*u21 + u22'*R[2]*u22 + u23'*R[2]*u23 + u24'*R[2]*u24
            + (x21-x_sol[2])'*Q[2]*(x21-x_sol[2]) + (x22-x_sol[2])'*Q[2]*(x22-x_sol[2]) + (x23-x_sol[2])'*Q[2]*(x23-x_sol[2]) + (x24-x_sol[2])'*Q[2]*(x24-x_sol[2])
            + (x31-x_sol[3])'*Q[3]*(x31-x_sol[3]) + (x32-x_sol[3])'*Q[3]*(x32-x_sol[3]) + (x33-x_sol[3])'*Q[3]*(x33-x_sol[3]) + (x34-x_sol[3])'*Q[3]*(x34-x_sol[3]))
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

    # resampling
    β = 0.1
    x̂2 = (x21+x22+x23+x24)./N
    Σ2 = (x21-x̂2)*(x21-x̂2)' + (x22-x̂2)*(x22-x̂2)' + (x23-x̂2)*(x23-x̂2)' + (x24-x̂2)*(x24-x̂2)' + 1.0e-8*I
    cols2 = cholesky(Σ2).U

    x21s = x̂2 + β*cols2[:,1]
    x22s = x̂2 - β*cols2[:,1]
    x23s = x̂2 + β*cols2[:,2]
    x24s = x̂2 - β*cols2[:,2]

    if eltype(z) == Float64
        println("x̂2: $(x̂2), x̄2: $(x_sol[2])")
        println("x21: $(x21), x21s: $(x21s)")
        println("x22: $(x22), x22s: $(x22s)")
        println("x23: $(x23), x23s: $(x23s)")
        println("x24: $(x24), x24s: $(x24s)")
        println("\n")
    end

    c[1:2] = dyn_d(x11,u11,Δt) - x21
    c[3:4] = dyn_d(x12,u12,Δt) - x22
    c[5:6] = dyn_d(x13,u13,Δt) - x23
    c[7:8] = dyn_d(x14,u14,Δt) - x24

    c[9:10] = dyn_d(x21,u21,Δt) - x31
    c[11:12] = dyn_d(x22,u22,Δt) - x32
    c[13:14] = dyn_d(x23,u23,Δt) - x33
    c[15:16] = dyn_d(x24,u24,Δt) - x34

    c[17] = u11 + k1'*(x11 - x_sol[1]) - u_sol[1][1]
    c[18] = u12 + k1'*(x12 - x_sol[1]) - u_sol[1][1]
    c[19] = u13 + k1'*(x13 - x_sol[1]) - u_sol[1][1]
    c[20] = u14 + k1'*(x14 - x_sol[1]) - u_sol[1][1]

    c[21] = u21 + k2'*(x21 - x_sol[2]) - u_sol[2][1]
    c[22] = u22 + k2'*(x22 - x_sol[2]) - u_sol[2][1]
    c[23] = u23 + k2'*(x23 - x_sol[2]) - u_sol[2][1]
    c[24] = u24 + k2'*(x24 - x_sol[2]) - u_sol[2][1]

    return c
end
c0 = zeros(m_nlp)
con!(c0,ones(n_nlp))
c0
prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(copy(z0_nom),prob)

K_sample = [reshape(z_sol[5:6],m,n),
            reshape(z_sol[19:20],m,n)]
obj(z_sol)
obj(z0)

println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1])/N)")
eigen(A[1] - B[1]*K_sample[1])
eigen(A[2] - B[2]*K_sample[2])
eigen(A[1] - B[1]*K[1])
eigen(A[2] - B[2]*K[2])

using Plots

function simulate_linear_controller(K,T_sim,Δt,z0)
    T = length(K)+1
    times = [(t-1)*Δt for t = 1:T-1]
    tf = Δt*T
    t_sim = range(0,stop=tf,length=T_sim)
    dt_sim = tf/(T_sim-1)

    z_rollout = [z0]
    u_rollout = []
    for tt = 1:T_sim-1
        t = t_sim[tt]
        k = searchsortedlast(times,t)
        z = z_rollout[end]
        u = u_sol[k] -K[k]*(z - x_sol[k])
        push!(z_rollout,dyn_d(z,u,Δt) + 0.0*randn(n))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout
end

x_sol_tvlqr, u_sol_tvlqr = simulate_linear_controller(K,4*T,Δt,[0.0;0.0])
x_sol_sample, u_sol_sample = simulate_linear_controller(K_sample,4*T,Δt,[0.0;0.0])

plot(hcat(x_sol_tvlqr...)',color=:purple,xlabel="time step",width=2.0,label=["tvlqr" ""])
plot!(hcat(x_sol_sample...)',color=:orange,xlabel="time step",width=1.0,label=["sample" ""])
