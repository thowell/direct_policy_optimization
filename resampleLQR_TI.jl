using LinearAlgebra, ForwardDiff, Distributions, Plots
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

function dynamics(x,u,Δt)
    Δt = 0.1
    D = exp(Δt*[Ac Bc; zeros(1,n+m)])
    A = D[1:n,1:n]
    B = D[1:n,n .+ (1:m)]

    return A*x + B*u
end

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
β = 100.0
x11 = β*[1.0; 1.0]
x12 = β*[1.0; -1.0]
x13 = β*[-1.0; 1.0]
x14 = β*[-1.0; -1.0]

x1 = [x11,x12,x13,x14]

N = length(x1)

n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)# + n*(T-1) + m*(T-2)
m_nlp = N*(n*(T-1) + m*(T-1))

idx_k = [(t-1)*(m*n) .+ (1:m*n) for t = 1:T-1]
idx_x = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_u = [[(T-1)*(m*n) + (i-1)*(n*(T-1) + m*(T-1)) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-1] for i = 1:N]

# idx_xs = [N*(n*(T-1) + m*(T-1)) + m*n*(T-1) + (t-1)*(n+m) .+ (1:n) for t = 1:T-1]
# idx_us = [N*(n*(T-1) + m*(T-1)) + m*n*(T-1) + (t-1)*(n+m) + n .+ (1:m) for t = 1:T-2]

idx_con_dyn = [[(i-1)*(n*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-1] for i = 1:N]
idx_con_ctrl = [[(i-1)*(m*(T-1)) + N*(n*(T-1)) + (t-1)*m .+ (1:m) for t = 1:T-1] for i = 1:N]

# idx_con_dyn_s = [N*(n*(T-1) + m*(T-1)) + (t-1)*n .+ (1:n) for t = 1:T-2]
# idx_con_ctrl_s = [N*(n*(T-1) + m*(T-1)) + (T-2)*n + (t-1)*m .+ (1:m) for t = 1:T-2]

function obj(z)
    s = 0
    for t = 1:T-1
        for i = 1:N
            x = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            s += x'*Q*x + u'*R*u
        end
    end
    return s
end

function con!(c,z)
    for t = 1:T-1
        # resample
        β = 1.0
        if t > 1
            xμ = sum([view(z,idx_x[i][t-1]) for i = 1:N])./N
            Σμ = (0.5/(β^2))*sum([(view(z,idx_x[i][t-1]) - xμ)*(view(z,idx_x[i][t-1]) - xμ)' for i = 1:N]) + 1.0e-8*I
            cols = cholesky(Σμ).U
            # cols = fastsqrt(Σμ)
            xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:n]
        end
        for i = 1:N
            if t > 1 && eltype(z) == Float64
                i==1 && println("t: $t")
                println("xi: $(view(z,idx_x[i][t-1]))")
                println("xs: $(xs[i])")
            end
            x = (t==1 ? x1[i] : view(z,idx_x[i][t-1]))
            # x = (t==1 ? x1[i] : xs[i])
            x⁺ = view(z,idx_x[i][t])
            u = view(z,idx_u[i][t])
            k = reshape(view(z,idx_k[t]),m,n)
            c[idx_con_dyn[i][t]] = A*x + B*u - x⁺
            c[idx_con_ctrl[i][t]] = u + k*x
        end

        println("\n")
    end
    return c
end

c0 = rand(m_nlp)
con!(c0,ones(n_nlp))
prob = Problem(n_nlp,m_nlp,obj,con!,true)

z0 = rand(n_nlp)
z_sol = solve(z0,prob)

K_sample = [reshape(z_sol[idx_k[t]],m,n) for t = 1:T-1]
K_error = [norm(vec(K_sample[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("solution error: $(sum(K_error)/N)")

plot(K_error,xlabel="time step",ylabel="norm(Ks-K)/norm(K)",yaxis=:log,width=2.0,label="β=$β",legend=:bottom,title="Gain matrix error")

x2 = [z_sol[idx_x[i][1]] for i = 1:N]
x3 = [z_sol[idx_x[i][2]] for i = 1:N]

plt = plot()
for i = 1:N
    x = x1[i]
    plt = scatter!([x[1]],[x[2]],label="",color=:red)
end
for i = 1:N
    x = x2[i]
    plt = scatter!([x[1]],[x[2]],label="",color=:blue,marker=:star)
end
for i = 1:N
    x = x3[i]
    plt = scatter!([x[1]],[x[2]],label="",color=:green,marker=:square)
end

display(plt)
