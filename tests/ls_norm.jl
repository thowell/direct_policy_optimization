using LinearAlgebra, ForwardDiff, Plots
using StaticArrays, BenchmarkTools, SparseArrays, Distributions
include("ipopt.jl")

n = 100
m = 10

A = randn(m,n)
b = randn(m)

# ℓ1-norm (slack reformulation)
function obj_l1(z)
	t = z[n .+ (1:n)]
	return sum(t)
end

function con_l1!(c,z)
	x = z[1:n]
	t = z[n .+ (1:n)]
	c[1:m] = A*x - b
	c[m .+ (1:n)] = x - t
	c[m+n .+ (1:n)] = -x - t
	return c
end

n_nlp_l1 = n + n
m_nlp_l1 = m + n + n

idx_ineq_l1 = [i for i = m .+ (1:2n)]
prob_l1 = Problem(n_nlp_l1,m_nlp_l1,obj_l1,con_l1!,true,idx_ineq=idx_ineq_l1)
z0_l1 = rand(n_nlp_l1)
z_l1 = solve(z0_l1,prob_l1)
norm(A*z_l1[1:n] - b)

# ℓ2-norm (squared)
function obj_l2(z)
	return z'*z
end

function con_l2!(c,z)
	c[1:m] = A*z - b
	return c
end

n_nlp_l2 = n
m_nlp_l2 = m

prob_l2 = Problem(n_nlp_l2,m_nlp_l2,obj_l2,con_l2!,true)
z0_l2 = rand(n_nlp_l2)
z_l2 = solve(z0_l2,prob_l2)
norm(A*z_l2 - b)

# l∞-norm (slack reformulation)
function obj_l∞(z)
	t = z[n + 1]
	return t
end

function con_l∞!(c,z)
	x = z[1:n]
	t = z[n + 1]
	c[1:m] = A*x - b
	c[m .+ (1:n)] = x - t*ones(n)
	c[m+n .+ (1:n)] = -x - t*ones(n)
	return c
end

n_nlp_l∞ = n + 1
m_nlp_l∞ = m + n + n

idx_ineq_l∞ = [i for i = m .+ (1:2n)]
prob_l∞ = Problem(n_nlp_l∞,m_nlp_l∞,obj_l∞,con_l∞!,true,idx_ineq=idx_ineq_l∞)
z0_l∞ = rand(n_nlp_l∞)
z_l∞ = solve(z0_l∞,prob_l∞)
norm(A*z_l∞[1:n] - b)

scatter(z_l1[1:n],legend=:bottom,width=2.0,label="l1",xlabel="element i",ylabel="x_i")
scatter!(z_l2,width=2.0,label="l2")
scatter!(z_l∞[1:n],width=2.0,label="l∞")
