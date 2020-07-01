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
T = 5
Q = Matrix(10.0*I,n,n)
R = Matrix(0.1*I,m,m)

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q
for t = T-1:-1:1
    K[t] = (R + B'*P[t+1]*B)\(B'*P[t+1]*A)
    P[t] = Q + K[t]'*R*K[t] + (A-B*K[t])'*P[t+1]*(A-B*K[t])
end

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
x1 = [x11,x12,x13,x14]#,x15,x16,x17,x18]#,x19,x110,x111,x112]
N = length(x1)
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
        xtraj[i][t+1] = A*xtraj[i][t] + B*utraj[i][t]
        if i == 1
            utraj_nom[t] = -K[t]*xtraj_nom[t]
            xtraj_nom[t+1] = A*xtraj_nom[t] + B*utraj_nom[t]
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

z0_nom = 0.001*randn(n_nlp)
# for i = 1:N
#     for t = 1:T-1
#         z0_nom[x_idx[i][t]] = xtraj_nom[t]
#         z0[u_idx[i][t]] = utraj_nom[t]
#     end
# end

function obj(z)
    s = 0.0
    for t = 1:T-1
        for i = 1:N
            x = z[x_idx[i][t]]
            u = z[u_idx[i][t]]
            s += x'*Q*x + u'*R*u
        end
    end
    return s
end

obj(z0)

function con!(c,z)
    shift1 = 0
    shift2 = 0
    xs = []
    β = 0.75
    for t = 1:T-1
        if t > 1
            x̂ = sum([z[x_idx[i][t-1]] for i = 1:N])
            Σ = sum([(z[x_idx[i][t-1]] - x̂)*(z[x_idx[i][t-1]] - x̂)' for i = 1:N]) + 1.0e-8*I
            cols = cholesky(Σ).U
            for k = 1:n
                push!(xs, x̂ + β*cols[:,k] + 0.0*randn(n))
                push!(xs, x̂ - β*cols[:,k] + 0.0*randn(n))
            end
        end
        for i = 1:N
            x = (t==1 ? xtraj[i][1] : xs[i])#z[x_idx[i][t-1]])
            u = z[u_idx[i][t]]
            x⁺ = z[x_idx[i][t]]
            k = reshape(z[K_idx[t]],m,n)

            c[shift1 .+ (1:n)] = A*x + B*u - x⁺
            shift1 += n
            c[N*(n*(T-1)) + shift2 .+ (1:m)] = u + k*x
            shift2 += m
            if eltype(z) == Float64 && t > 1
                println("t $t")
                println("x̂ $(x̂)")
                println("x$i: $(z[x_idx[i][t-1]]), xs$i: $(xs[i])")
            end
        end
        if eltype(z) == Float64
            println("\n")
        end
        empty!(xs)
    end

    return c
end

c0 = rand(m_nlp)
con!(c0,z0)

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_nom,prob)

K_sample = [reshape(z_sol[K_idx[t]],m,n) for t = 1:T-1]

println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1])/N)")

using Plots
x_sol_tvlqr, u_sol_tvlqr = simulate_linear_controller(K,50,Δt,[2.0;-3.0])
x_sol_sample, u_sol_sample = simulate_linear_controller(K_sample,50,Δt,[2.0;-3.0])

plot(hcat(x_sol_tvlqr...)',color=:purple,xlabel="time step",width=2.0,label=["tvlqr" ""])
plot!(hcat(x_sol_sample...)',color=:orange,xlabel="time step",width=1.0,label=["sample" ""])

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
        u = -K[k]*z
        push!(z_rollout,A*z + B*u + 0.001*randn(n))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout
end
