using LinearAlgebra, Convex, SCS, ECOS

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
T = 2
Q = Matrix(1.0*I,n,n)
R = Matrix(0.1*I,m,m)

P = [Semidefinite(n) for t = 1:T]

problem = minimize(0.0)
problem.constraints += P[T] == Q
# problem.constraints += Q + A'*P[T]*A - A'*P[T]*B*(R + B'*P[T]*B)\(B'*P[T]*A) - P[T-1] == 0
problem.constraints += P[T-1]*A + A'*P[T-1] <= -Matrix(I,n,n)
problem.constraints += P[T-1] >= Matrix(I,n,n)
solve!(problem, SCS.Optimizer)
problem.status
problem.optval

P[T].value
P[T-1].value
