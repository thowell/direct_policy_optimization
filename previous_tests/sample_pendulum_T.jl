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
Δt = 0.01
function dyn_d(x,u,Δt)
    x + Δt*dyn_c(x + 0.5*Δt*dyn_c(x,u),u)
end

dyn_c(rand(n),rand(m))
dyn_d(rand(n),rand(m),0.2)

# Trajectory optimization
T = 50
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

Q = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0;100.0]) for t = 1:T]
R = [Matrix((0.01)*I,m,m) for t = 1:T-1]

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

x1 = [x11,x12,x13,x14]

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
        utraj[i][t] = u_sol[t] -K[t]*(xtraj[i][t] - x_sol[t])
        xtraj[i][t+1] = dyn_d(xtraj[i][t],utraj[i][t],Δt)
        if i == 1
            utraj_nom[t] = u_sol[t] - K[t]*(xtraj_nom[t] - x_sol[t])
            xtraj_nom[t+1] = dyn_d(xtraj_nom[t],utraj_nom[t],Δt)
        end
    end
end

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1) + m*(T-1))
x_idx = [[(i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
u_idx = [[(i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]
K_idx = [N*(n*(T-1) + m*(T-1)) + (t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]

z0 = zeros(n_nlp)
z0_nom = 0.001*randn(n_nlp)
for i = 1:N
    for t = 1:T-1
        z0[x_idx[i][t]] = xtraj[i][t+1]
        z0[u_idx[i][t]] = utraj[i][t]
        z0_nom[x_idx[i][t]] = xtraj[i][t+1]
        z0_nom[u_idx[i][t]] = utraj[i][t]

        if i == 1
            z0[K_idx[t]] = vec(K[t])
        end
    end
end

function obj(z)
    s = 0.0
    for i = 1:N
        for t = 1:T-1
            x = z[x_idx[i][t]]
            u = z[u_idx[i][t]]
            s += (x - x_sol[t+1])'*Q[t+1]*(x - x_sol[t+1]) + u'*R[t]*u
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

            c[shift1 .+ (1:n)] = dyn_d(x,u,Δt) - x⁺
            shift1 += n

            c[N*(n*(T-1)) + shift2 .+ (1:m)] = u + k*(x - x_sol[t]) - u_sol[t]
            shift2 += m
        end
    end

    return c
end

c0 = zeros(m_nlp)
con!(c0,ones(n_nlp))

prob = Problem(n_nlp,m_nlp,obj,con!,true)

# z_sol = solve(z0_nom,prob)

obj(z_sol)
obj(z0)
K_sample = [reshape(z_sol[K_idx[t]],m,n) for t = 1:T-1]

println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1])/N)")


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
        push!(z_rollout,dyn_d(z,u,dt_sim) + 0.05*randn(n))
        push!(u_rollout,u)
    end
    return z_rollout, u_rollout, t_sim
end
n
m
β = 1.0
N_sample = 2*(n+m)
K_ukf = []
z_sol
let K_ukf=K_ukf, H=Q[T]
    tmp = zeros(n+m,n+m)
    z_tmp = zeros(n+m)

    for t = T:-1:2
        println("t: $t")
        tmp[1:n,1:n] = H
        tmp[n .+ (1:m),n .+ (1:m)] = R[t-1]
        L = cholesky(inv(tmp)).U

        z_tmp[1:n] = x_sol[t]
        z_tmp[n .+ (1:m)] = u_sol[t-1]

        z_sample = [z_tmp + β*L[:,i] for i = 1:(n+m)]
        z_sample = [z_sample...,[z_tmp - β*L[:,i] for i = 1:(n+m)]...]

        z_sample_prev = [[dyn_d(zs[1:n],zs[n .+ (1:m)],-Δt);zs[n .+ (1:m)]] for zs in z_sample]

        z_tmp[1:n] = x_sol[t-1]
        M = 0.5/(β^2)*sum([(zsp - z_tmp)*(zsp - z_tmp)' for zsp in z_sample_prev])

        P = inv(M)
        P[1:n,1:n] += Q[t-1]

        A = P[1:n,1:n]
        C = P[n .+ (1:m),1:n]
        B = P[n .+ (1:m), n .+ (1:m)]

        K = -(B + 1.0e-8*I)\C

        push!(K_ukf,-K)

        H = A + K'*B*K + K'*C + C'*K
        H = 0.5*(H + H')
    end
end
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + Δt
end
# for t = 1:T-1
#     println("K_ukf[$t] = $(K_ukf[t])")
#     println("K_sample[$t] = $(K_sample[t])")
#     println("K_tvlqr[$t] = $(K[t])")
# end

x_sol_tvlqr, u_sol_tvlqr, t_sim = simulate_linear_controller(K,10*T,Δt,[0.0;0.0])
# x_sol_sample, u_sol_sample = simulate_linear_controller(K_sample,2*T,Δt,[0.0;0.0])
x_sol_ukf, u_sol_ukf, t_sim = simulate_linear_controller(K_ukf,10*T,Δt,[0.0;0.0])

plot(t_nominal,hcat(x_sol...)[1,:],color=:orange,xlabel="time step",width=2.0,label=["nominal" ""])
plot!(t_nominal,hcat(x_sol...)[2,:],color=:orange,width=2.0,label="")
plot!(t_sim,hcat(x_sol_tvlqr...)[1,:],color=:purple,xlabel="time step",width=2.0,label=["tvlqr" ""])
plot!(t_sim,hcat(x_sol_tvlqr...)[2,:],color=:purple,width=2.0,label="")
plot!(t_sim,hcat(x_sol_ukf...)[1,:],color=:cyan,xlabel="time step",width=1.0,label=["ukf" ""])
plot!(t_sim,hcat(x_sol_ukf...)[2,:],color=:cyan,width=1.0,label="")
# plot!(hcat(x_sol_sample...)',color=:orange,xlabel="time step",width=1.0,label=["sample" ""])
