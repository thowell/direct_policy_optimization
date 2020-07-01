using LinearAlgebra, Convex, SCS, ECOS, COSMO
y = Semidefinite(2)
p = maximize(eigmin(y), tr(y)<=6)
solve!(p, COSMO.Optimizer)
p.optval
p.status
