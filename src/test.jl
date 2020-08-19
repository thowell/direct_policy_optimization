# test problem
Z0_test = rand(prob.N)
tmp_o(z) = eval_objective(prob,z)
∇obj = zeros(prob.N)
eval_objective_gradient!(∇obj,Z0_test,prob)
@assert norm(ForwardDiff.gradient(tmp_o,Z0_test) - ∇obj) < 1.0e-10
c0 = zeros(prob.M)
eval_constraint!(c0,Z0_test,prob)
tmp_c(c,z) = eval_constraint!(c,z,prob)
ForwardDiff.jacobian(tmp_c,c0,Z0_test)

spar = sparsity_jacobian(prob)
∇c_vec = zeros(length(spar))
∇c = zeros(prob.M,prob.N)
eval_constraint_jacobian!(∇c_vec,Z0_test,prob)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_test))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_test)) < 1.0e-10

# test sample problem
Z0_sample = rand(prob_sample.N_nlp)
tmp_o(z) = eval_objective(prob_sample,z)
∇obj_ = zeros(prob_sample.N_nlp)
eval_objective_gradient!(∇obj_,Z0_sample,prob_sample)
@assert norm(ForwardDiff.gradient(tmp_o,Z0_sample) - ∇obj_) < 1.0e-10

tmp_o(z) = sample_objective(z,prob_sample)
∇obj_ = zeros(prob_sample.N_nlp)
∇sample_objective!(∇obj_,Z0_sample,prob_sample)
@assert norm(ForwardDiff.gradient(tmp_o,Z0_sample) - ∇obj_) < 1.0e-10

# include("sample_penalty_objective.jl")
# tmp_o(z) = sample_general_objective(z,prob_sample)
# ∇obj_ = zeros(prob_sample.N_nlp)
# ∇sample_general_objective!(∇obj_,Z0_sample,prob_sample)
# @assert norm(ForwardDiff.gradient(tmp_o,Z0_sample) - ∇obj_) < 1.0e-10

c0 = zeros(prob_sample.M_policy)
sample_policy_constraints!(c0,Z0_sample,prob_sample)
tmp_c(c,z) = sample_policy_constraints!(c,z,prob_sample)
ForwardDiff.jacobian(tmp_c,c0,Z0_sample)

spar = sparsity_jacobian_sample_policy(prob_sample)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_sample.M_policy,prob_sample.N_nlp)
∇sample_policy_constraints!(∇c_vec,Z0_sample,prob_sample)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_sample))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_sample)) < 1.0e-10

# include("../src/general_constraints.jl")
c0 = zeros(prob_sample.M_nlp)
eval_constraint!(c0,Z0_sample,prob_sample)
tmp_c(c,z) = eval_constraint!(c,z,prob_sample)
ForwardDiff.jacobian(tmp_c,c0,Z0_sample)

spar = sparsity_jacobian(prob_sample)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_sample.M_nlp,prob_sample.N_nlp)
eval_constraint_jacobian!(∇c_vec,Z0_sample,prob_sample)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_sample))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_sample)) < 1.0e-10

# policy constraint for simulation
# include("../src/simulate.jl")
#
# tol=1.0e-6
# c_tol=1.0e-6
# α=100.0
# slack_tol=1.0e-5
#
# tf = sum(H_nom)
# times = [(t-1)*H_nom[t] for t = 1:T-2]
# t_sim = range(0,stop=tf,length=T_sim)
# dt_sim = tf/(T_sim-1)
#
# # Bounds
#
# # h = h0 (fixed timestep)
# hu_sim = dt_sim
# hl_sim = dt_sim
#
# model.α = α
# penalty_obj = PenaltyObjective(model.α)
# multi_obj = MultiObjective([penalty_obj])
#
# X_traj = [X_nom[1],X_nom[2]]
# U_traj = []
#
# t = 1
# k = 1
# # xl <= x <= xu
# xu_sim = [X_traj[t],X_traj[t+1],Inf*ones(model.nx)]
# xl_sim = [X_traj[t],X_traj[t+1],-Inf*ones(model.nx)]
#
# # ul <= u <= uu
# uu_sim = Inf*ones(model.nu)
# ul_sim = zeros(model.nu)
#
# uu_sim[model.idx_u] = U_nom[k][model.idx_u]
# ul_sim[model.idx_u] = U_nom[k][model.idx_u]
#
# # policy
# pi = PolicyInfo(X_nom[k:k+2],U_nom[k:k],H_nom[k:k],K_nom_sample[k:k])
#
# general_constraint=true
# m_general=model.nu_ctrl
#
# # Problem
# prob_sim = init_problem(model.nx,model.nu,3,model,multi_obj,
#                         xl=xl_sim,
#                         xu=xu_sim,
#                         ul=[ul_sim],
#                         uu=[uu_sim],
#                         hl=[dt_sim],
#                         hu=[dt_sim],
#                         general_constraints=general_constraint,
#                         m_general=m_general,
#                         general_ineq=(1:0),
#                         policy_info=pi)
# Z0_sim = pack([X_traj[t],X_traj[t+1],X_traj[t+1]],[t == 1 ? U_nom[1] : U_traj[t-1]],dt_sim,prob_sim)
#
# c0 = zeros(model.nu_ctrl)
# general_constraints!(c0,Z0_sim,prob_sim)
# tmp_c(c,z) = general_constraints!(c,z,prob_sim)
# ForwardDiff.jacobian(tmp_c,c0,Z0_sim)
#
# spar = general_constraint_sparsity(prob_sim)
# ∇c_vec = zeros(length(spar))
# ∇c = zeros(model.nu_ctrl,prob_sim.N)
# ∇general_constraints!(∇c_vec,Z0_sim,prob_sim)
# for (i,k) in enumerate(spar)
#     ∇c[k[1],k[2]] = ∇c_vec[i]
# end
# @assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_sim))) < 1.0e-10
# @assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_sim)) < 1.0e-10
#
