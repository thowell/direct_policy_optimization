using LinearAlgebra, Convex, SCS, ECOS, COSMO

n = 2
m = 1
A = [1.0 0.1; 0.0 1.0]
B = [0.0; 1.0]
K = ones(m,n)
P = Semidefinite(n)
Q = Semidefinite(n)
R = Semidefinite(m)

prob = minimize(0.0)
prob.constraints += Q + A'*P*(A + B*K) - P == 0.
prob.constraints += R*K + B'*P*(A + B*K) == 0.
prob.constraints += P >= 0.
prob.constraints += Q >= 0.
prob.constraints += R >= 0.
solve!(prob,COSMO.Optimizer())
prob.status

P.value
Q.value
R.value
