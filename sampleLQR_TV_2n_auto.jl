using LinearAlgebra, ForwardDiff
include("ipopt.jl")

# Horizon
T = 4

# continuous-time dynamics
n = 2
m = 1

# discrete-time dynamics (ρ(A) = 1.0)
A = []
B = []
for t = 1:T-1
    tmp = randn(n,n)
    _A = tmp'*tmp
    _B = randn(n,m)

    eig = eigen(_A)

    for (i,v) in enumerate(eig.values)
        eig.values[i] = real(v)
        if abs(v) > 1.0
            # @warn "correcting eigen value: $v -> 1.0"
            eig.values[i] = 1.0
        end
    end
    # println("eig(A): $(eig.values)")
    push!(A,eig.vectors*Diagonal(eig.values)*eig.vectors')
    push!(B,_B)
end

# TVLQR solution
Q = [Matrix(rand(1)[1]*I,n,n) for t = 1:T]
R = [Matrix(rand(1)[1]*I,m,m) for t = 1:T-1]

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q[T]
for t = T-1:-1:1
    K[t] = (R[t] + B[t]'*P[t+1]*B[t])\(B[t]'*P[t+1]*A[t])
    P[t] = Q[t] + K[t]'*R[t]*K[t] + (A[t]-B[t]*K[t])'*P[t+1]*(A[t]-B[t]*K[t])
end

# samples
Σ = Matrix(1.0*I,n,n)
x1 = []
for i = 1:n
    push!(x1,Σ[:,i])
    push!(x1,-1.0*Σ[:,i])
end
N = length(x1)

# simulate
xtraj = [[zeros(n) for t = 1:T] for i = 1:N]
xtraj_nom = [zeros(n) for t = 1:T]
for i = 1:N
    xtraj[i][1] = x1[i]
end

utraj = [[zeros(m) for t = 1:T-1] for i = 1:N]
utraj_nom = [zeros(m) for t = 1:T-1]

for t = 1:T-1
    utraj_nom[t] = -K[t]*xtraj_nom[t]
    xtraj_nom[t+1] = A[t]*xtraj_nom[t] + B[t]*utraj_nom[t]

    for i = 1:N
        utraj[i][t] = -K[t]*xtraj[i][t]
        xtraj[i][t+1] = A[t]*xtraj[i][t] + B[t]*utraj[i][t]
    end
end

# NLP problem dimensions
n_nlp = N*(n*(T-1) + m*(T-1)) + m*n*(T-1)
m_nlp = N*(n*(T-1)) + N*(m*(T-1))

# TVLQR solution
z0 = zeros(n_nlp)

for t = 1:T-1
    for i = 1:N
        z0[(t-1)*(N*m + m*n + N*n) + (i-1)*m .+ (1:m)] = utraj[i][t]
    end

    z0[(t-1)*(N*m + m*n + N*n) + N*m .+ (1:m*n)] = K[t]

    for i = 1:N
        z0[(t-1)*(N*m + m*n + N*n) + N*m + m*n + (i-1)*n .+ (1:n)] = xtraj[i][t+1]
    end
end
z0

# Nominal trajectory initialization
z0_nom = zeros(n_nlp)

for t = 1:T-1
    for i = 1:N
        z0_nom[(t-1)*(N*m + m*n + N*n) + (i-1)*m .+ (1:m)] = utraj_nom[t]
    end

    z0_nom[(t-1)*(N*m + m*n + N*n) + N*m .+ (1:m*n)] .= 0.0 # K[t]

    for i = 1:N
        z0_nom[(t-1)*(N*m + m*n + N*n) + N*m + m*n + (i-1)*n .+ (1:n)] = xtraj_nom[t+1]
    end
end

# Objective
function obj(z)
    s = 0.0
    for t = 1:T-1
        for i = 1:N
            ui  = z[(t-1)*(N*m + m*n + N*n) + (i-1)*m .+ (1:m)]
            xi⁺ = z[(t-1)*(N*m + m*n + N*n) + N*m + m*n + (i-1)*n .+ (1:n)]
            s += xi⁺'*Q[t+1]*xi⁺ + ui'*R[t]*ui
        end
    end

    return s
end

obj(z0)

# Constraints
function con!(c,z)
    for t = 1:T-1
        k = reshape(z[(t-1)*(N*m + m*n + N*n) + N*m .+ (1:m*n)],m,n)
        for i = 1:N
            xi = (t == 1 ? xtraj[i][1] : z[(t-2)*(N*m + m*n + N*n) + N*m + m*n + (i-1)*n .+ (1:n)])
            ui = z[(t-1)*(N*m + m*n + N*n) + (i-1)*m .+ (1:m)]
            xi⁺ = z[(t-1)*(N*m + m*n + N*n) + N*m + m*n + (i-1)*n .+ (1:n)]
            c[(t-1)*(N*n) + (i-1)*(n) .+ (1:n)] = A[t]*xi + B[t]*ui - xi⁺
            c[(T-1)*(N*n) + (t-1)*(N*m) + (i-1)*m .+ (1:m)] = ui + k*xi
        end
    end
    return c
end

c0 = zeros(m_nlp)
con!(c0,z0_nom)

# NLP problem
prob = Problem(n_nlp,m_nlp,obj,con!,true)

# Solve
z_sol = solve(z0_nom,prob,tol=1.0e-6,max_iter=1000)
K_sample = [reshape(z_sol[(t-1)*(N*m + m*n + N*n) + N*m .+ (1:m*n)],m,n) for t = 1:T-1]

# Check solution
println("solution error: $(sum([norm(vec(K_sample[t]) - vec(K[t])) for t = 1:T-1])/N)")
