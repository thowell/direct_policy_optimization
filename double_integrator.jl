using LinearAlgebra, ForwardDiff, Plots, StaticArrays, BenchmarkTools
include("/home/taylor/Research/non-convex_solver/src/non-convex_solver.jl")

# parameterized model
mutable struct DoubleIntegrator2D
    # z = (x,y,ẋ,ẏ)
    mx
    my
end

# double integrator dynamics (2D)
function dynamics(model::DoubleIntegrator2D,z,u)
    @SVector [z[3], z[4], u[1]/model.mx, u[2]/model.my]
end

n = 4 # number of states
m = 2 # number of controls

model = DoubleIntegrator2D(1.0,1.0)
z0 = zeros(4)
u0 = ones(2)
dynamics(model,z0,u0)

# discrete dynamics (midpoint)
Δt = 0.05
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end
midpoint(model,z0,u0,Δt)

T = 40
x_ref = sin.(range(0.0,stop=pi,length=T))
y_ref = range(0.0,stop=pi,length=T)
z_ref = zeros(T*n)
for t = 1:T
    z_ref[(t-1)*n .+ (1:n)] = [x_ref[t];y_ref[t];0.0;0.0]
end
u0 = zeros(m)
u_ref = copy(u0)

# optimize nominal trajectories
z0 = [x_ref[1];y_ref[1];0.0;0.0]
Q = Diagonal(@SVector[1.0,1.0,1.0,1.0])
Qf = Diagonal(@SVector[1.0,1.0,1.0,1.0])
R = 1.0e-5*sparse(I,m,m)

function f_func(x)
    s = 0.0
    for t = 1:T-1
        z = x[(t-1)*(n+m) .+ (1:n)]
        u = x[(t-1)*(n+m)+n .+ (1:m)]
        s += (z - z_ref[(t-1)*n .+ (1:n)])'*Q*(z - z_ref[(t-1)*n .+ (1:n)]) + (u - u_ref)'*R*(u - u_ref)
    end
    z = x[(T-1)*(n+m) .+ (1:n)]
    s += (z - z_ref[(T-1)*n .+ (1:n)])'*Qf*(z - z_ref[(T-1)*n .+ (1:n)])

    return s
end

f, ∇f!, ∇²f! = objective_functions(f_func)

function c_func(x)
    c = zeros(eltype(x),n*T)
    for t = 1:T-1
        z = x[(t-1)*(n+m) .+ (1:n)]
        z⁺ = x[t*(n+m) .+ (1:n)]
        u = x[(t-1)*(n+m)+n .+ (1:m)]
        c[(t-1)*n .+ (1:n)] = z⁺ - midpoint(model,z,u,Δt)
    end
    z = x[1:n]
    c[(T-1)*n .+ (1:n)] = z - z0

    return c
end
c!, ∇c!, ∇²cy! = constraint_functions(c_func)






n_nlp = n*T + m*(T-1)
m_nlp = n*T
xL = -Inf*ones(n_nlp)
xU = Inf*ones(n_nlp)

nlp_model = Model(n_nlp,m_nlp,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=zeros(Bool,m_nlp),cA_idx=zeros(Bool,m_nlp))

x0 = zeros(n_nlp)

for t = 1:T-1
    x0[(t-1)*(n+m) .+ (1:(n+m))] = [x_ref[t];y_ref[t];0.0;0.0;u0]
end
x0[(T-1)*(n+m) .+ (1:(n))] = [x_ref[T];y_ref[T];0.0;0.0]





opts = Options{Float64}(kkt_solve=:symmetric,
                        max_iter=250,
                        ϵ_tol=1.0e-5,
                        ϵ_al_tol=1.0e-5,
                        verbose=true,
                        quasi_newton=:none)

s = NonConvexSolver(x0,nlp_model,opts=opts)
@time solve!(s)
x_sol = get_solution(s)

z_nom = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
x_nom = vec([x_sol[(t-1)*(n+m) .+ (1:n)][1] for t = 1:T])
y_nom = vec([x_sol[(t-1)*(n+m) .+ (1:n)][2] for t = 1:T])
u_nom = [x_sol[(t-1)*(n+m)+n .+ (1:m)] for t = 1:T-1]

plot(x_ref,y_ref,title="reference trajectory",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")

plot(hcat(z_nom...)',width=2.0,label=["x" "y" "ẋ" "ẏ"],title="states")
plot(hcat(u_nom...)',linetype=:steppost,width=2.0,label=["u1" "u2"],title="control")

# discrete TV-LQR
A = []
B = []
for t = 1:T-1
    fz(z) = midpoint(model,z,u_nom[t],Δt)
    fu(u) = midpoint(model,z_nom[t],u,Δt)
    push!(A,ForwardDiff.jacobian(fz,z_nom[t]))
    push!(B,ForwardDiff.jacobian(fu,u_nom[t]))
end

Q_tvlqr = Q#Diagonal(@SVector[100.0,100.0,1.0e-1,1.0e-1])
Qf_tvlqr = Qf#Diagonal(@SVector[100.0,100.0,1.0e-1,1.0e-1])
R_tvlqr = R#1.0e-2*sparse(I,m,m)
P = []
push!(P,Qf_tvlqr)
K = []

for t = T-1:-1:1
    println(t)
    push!(K,(R_tvlqr + B[t]'*P[end]*B[t])\(B[t]'*P[end]*A[t]))
    push!(P,A[t]'*P[end]*A[t] - (A[t]'*P[end]*B[t])*K[end] + Q_tvlqr)
end

# simulation
model_alt = DoubleIntegrator2D(5.0,0.2)

times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
z_tvlqr_rollout = [z0]
u_tvlqr = []
for tt = 1:T_sim-1
    t = t_sim[tt]
    k = searchsortedlast(times,t)
    w = randn(n)*1.0e-2
    z = z_tvlqr_rollout[end] + 1.0*w
    u = u_nom[k] - K[k]*(z - z_nom[k])
    push!(z_tvlqr_rollout,midpoint(model_alt,z,u,dt_sim))
    push!(u_tvlqr,u)
end

z_tvlqr_rollout
x_tvlqr_rollout = [z_tvlqr_rollout[t][1] for t = 1:T_sim]
y_tvlqr_rollout = [z_tvlqr_rollout[t][2] for t = 1:T_sim]

plot(x_ref,y_ref,title="simulation",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")
plot!(x_tvlqr_rollout,y_tvlqr_rollout,color=:purple,label="TVLQR",width=2.0)

plot(hcat(u_nom...)',linetype=:steppost,width=2.0,label=["u1 nom." "u2 nom."],title="control")
plot(hcat(u_tvlqr...)',linetype=:steppost,width=2.0,label=["u1 tvlqr" "u2 tvlqr"],title="control")

# # robust sampling controller
masses = [1.1]
models = [DoubleIntegrator2D(m,1.0) for m in masses]

# N = length(models)
#
# n_nlp_robust = n*N*T + (n*m)*(T-1)
# m_nlp_robust = n*N*T
#
# function f_func(x)
#     s = 0.0
#     for t = 1:T
#
#         z = x[(t-1)*(n*N + n*m) .+ (1:n*N)]
#         for i = 1:N
#             zi = z[(i-1)*n .+ (1:n)]
#             s += (zi - z_nom[t])'*Q*(zi - z_nom[t])/N
#
#             if t < T
#                 K = reshape(x[(t-1)*(n*N + n*m) + n*N .+ (1:n*m)],m,n)
#                 δu_robust = -K*(zi - z_nom[t])
#                 s += δu_robust'*R*δu_robust/N
#             end
#         end
#     end
#     return s
# end
#
# f, ∇f!, ∇²f! = objective_functions(f_func)
#
# function c_func(x)
#     c = zeros(eltype(x),n*N*T)
#     for t = 1:T-1
#         z = x[(t-1)*(n*N + n*m) .+ (1:n*N)]
#         z⁺ = x[t*(n*N + n*m) .+ (1:n*N)]
#         K = reshape(x[(t-1)*(n*N + n*m) + n*N .+ (1:n*m)],m,n)
#         for i = 1:N
#             zi = z[(i-1)*n .+ (1:n)]
#             zi⁺ = z⁺[(i-1)*n .+ (1:n)]
#             c[(t-1)*n*N + (i-1)*n .+ (1:n)] = zi⁺ - midpoint(models[i],zi,u_nom[t] - K*(zi - z_nom[t]),Δt)
#         end
#     end
#     z = x[1:n*N]
#     for i = 1:N
#         zi = z[(i-1)*n .+ (1:n)]
#         c[(T-1)*n*N + (i-1)*n .+ (1:n)] = zi - z0
#     end
#     return c
# end
# c!, ∇c!, ∇²cy! = constraint_functions(c_func)
#
# xL = -Inf*ones(n_nlp_robust)
# xU = Inf*ones(n_nlp_robust)
#
# nlp_model = Model(n_nlp_robust,m_nlp_robust,xL,xU,f,∇f!,∇²f!,c!,∇c!,∇²cy!,cI_idx=zeros(Bool,m_nlp_robust),cA_idx=zeros(Bool,m_nlp_robust))
#
# x0 = zeros(n_nlp_robust)
#
# c_func(x0)
#
# for t = 1:T
#     for i = 1:N
#         x0[(t-1)*(n*N + n*m) + (i-1)*n .+ (1:n)] = z_nom[t]
#     end
#     if t < T
#         x0[(t-1)*(n*N + n*m) + N*n .+ (1:n*m)] = vec(K[t])
#     end
# end
#
# #TODO initialize TVLQR K gains
#
# opts = Options{Float64}(kkt_solve=:symmetric,
#                         max_iter=250,
#                         ϵ_tol=1.0e-3,
#                         ϵ_al_tol=1.0e-3,
#                         verbose=true,
#                         quasi_newton=:none)
#
# s = NonConvexSolver(x0,nlp_model,opts=opts)
#
# @time solve!(s)
# x_sol = get_solution(s)
# x_sol = copy(x0)
# K_robust = []
# for t = 1:T-1
#     push!(K_robust,reshape(x_sol[(t-1)*(n*N + n*m) + n*N .+ (1:n*m)],m,n))
# end
# K_robust
# times = [(t-1)*Δt for t = 1:T-1]
# tf = Δt*T
# T_sim = 1000
# t_sim = range(0,stop=tf,length=T_sim)
# dt_sim = tf/(T_sim-1)
# z_robust = [z0]
# u_robust = []
# for tt = 1:T_sim-1
#     t = t_sim[tt]
#     k = searchsortedlast(times,t)
#     w = randn(n)*0.1e-1
#     z = z_robust[end] + 0.0*w
#     u = u_nom[k] - K_robust[k]*(z - z_nom[k])
#     push!(z_robust,midpoint(models[1],z,u,dt_sim))
#     push!(u_robust,u)
# end
#
# x_robust = [z_robust[t][1] for t = 1:T_sim]
# y_robust = [z_robust[t][2] for t = 1:T_sim]
#
# plot(x_ref,y_ref,title="simulation",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
# scatter!(x_nom,y_nom,color=:orange,label="nominal")
# plot!(x_robust,y_robust,color=:green,label="Robust",width=2.0)


# parameterized controller
# n_nlp_robust = n*T + (m*n)*(T-1)
# m_nlp_robust = n*T

n_nlp_robust = n*T + m*(T-1)
m_nlp_robust = n*T

prob = Problem(n_nlp_robust,
               m_nlp_robust,
               z_nom,
               u_nom,
               T,
               n,
               m,
               1,
               Diagonal(@SVector[10.0,10.0,1.0e-2,1.0e-2]),
               Diagonal(@SVector[10.0,10.0,1.0e-2,1.0e-2]),
               1.0e-2*sparse(I,m,m),
               A,
               B,
               model,
               Δt,
               z0,
               false)
primal_bounds(prob)
constraint_bounds(prob)

x0 = zeros(n_nlp_robust)

for t = 1:T
    x0[(t-1)*(n + m) .+ (1:n)] = z_nom[t]
    if t < T
        x0[(t-1)*(n + m)+n .+ (1:m)] = u_nom[t]
    end
end

x0
MOI.eval_objective(prob,x0)
MOI.eval_objective_gradient(prob,ones(n_nlp_robust),x0)
MOI.eval_constraint(prob,ones(m_nlp_robust),x0)
MOI.eval_constraint_jacobian(prob,ones(m_nlp_robust*n_nlp_robust),x0)
sparsity(prob)
x_sol = solve_ipopt(x_sol,prob)

z_opt = [x_sol[(t-1)*(n+m) .+ (1:n)] for t = 1:T]
x_opt = vec([x_sol[(t-1)*(n+m) .+ (1:n)][1] for t = 1:T])
y_opt = vec([x_sol[(t-1)*(n+m) .+ (1:n)][2] for t = 1:T])
u_opt = [x_sol[(t-1)*(n+m)+m .+ (1:m)] for t = 1:T]

K_robust = []
for t = 1:T-1
    push!(K_robust,reshape(x_sol[(t-1)*(n + m*n).+ (1:n*m)],m,n))
end
K_robust
times = [(t-1)*Δt for t = 1:T-1]
tf = Δt*T
T_sim = 1000
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)
z_robust = [z0]
u_robust = []
for tt = 1:T_sim-1
    t = t_sim[tt]
    k = searchsortedlast(times,t)
    w = randn(n)*0.1e-1
    z = z_robust[end] + 0.0*w
    u = u_nom[k]# - K_robust[k]*(z - z_nom[k])
    push!(z_robust,midpoint(model,z,u,dt_sim))
    push!(u_robust,u)
end

x_robust = [z_robust[t][1] for t = 1:T_sim]
y_robust = [z_robust[t][2] for t = 1:T_sim]

plot(x_ref,y_ref,title="simulation",color=:black,legend=:topleft,label="ref.",xlabel="x",ylabel="y",width=2.0,aspect_ratio=:equal)
scatter!(x_nom,y_nom,color=:orange,label="nominal")
plot!(x_opt,y_opt,color=:green,label="opt",width=2.0)

yy = rand(m*n)
yy_mat = reshape(yy,m,n)
