include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

n = 5
p = 3
tmp = rand(5,5)
Wx = tmp'*tmp#Diagonal(0.1*ones(n))
Ww = Diagonal(0.01*ones(p))

W = Array(cat(Wx,Ww,dims=(1,2)))
L = Array(cholesky(W).L)
L = sqrt(W)
vec(L*L') -  vec(W)

T = 25
n = 10
m = 4
p = 1
Z_dpo = (2*n*(n+m))^3
Z_cond = ((n^2 + n)/2 + n)^3 #+ 2*(n+p)*m)^3

Z_dpo/Z_cond
