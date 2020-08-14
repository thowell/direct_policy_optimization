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

# test contact dynamics constraints
c0 = zeros(prob.M_contact_dynamics)
contact_dynamics_constraints!(c0,Z0_test,prob)
tmp_c(c,z) = contact_dynamics_constraints!(c,z,prob)
ForwardDiff.jacobian(tmp_c,c0,Z0_test)

spar = sparsity_contact_dynamics_jacobian(prob)
∇c_vec = zeros(length(spar))
∇c = zeros(prob.M_contact_dynamics,prob.N)
contact_dynamics_constraints_jacobian!(∇c_vec,Z0_test,prob)
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

include("sample_penalty_objective.jl")
tmp_o(z) = sample_general_objective(z,prob_sample)
∇obj_ = zeros(prob_sample.N_nlp)
∇sample_general_objective!(∇obj_,Z0_sample,prob_sample)
@assert norm(ForwardDiff.gradient(tmp_o,Z0_sample) - ∇obj_) < 1.0e-10

c0 = zeros(prob_sample.M_contact_dynamics)
sample_contact_dynamics_constraints!(c0,Z0_sample,prob_sample)
tmp_c(c,z) = sample_contact_dynamics_constraints!(c,z,prob_sample)
ForwardDiff.jacobian(tmp_c,c0,Z0_sample)

spar = sparsity_jacobian_sample_contact_dynamics(prob_sample)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_sample.M_contact_dynamics,prob_sample.N_nlp)
∇sample_contact_dynamics_constraints!(∇c_vec,Z0_sample,prob_sample)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_sample))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_sample)) < 1.0e-10

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
c0 = zeros(model.nu_ctrl)
general_constraints!(c0,Z0_sim,prob_sim)
tmp_c(c,z) = general_constraints!(c,z,prob_sim)
ForwardDiff.jacobian(tmp_c,c0,Z0_sim)

spar = general_constraint_sparsity(prob_sim)
∇c_vec = zeros(length(spar))
∇c = zeros(model.nu_ctrl,prob_sim.N)
∇general_constraints!(∇c_vec,Z0_sim,prob_sim)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end
@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_sim))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_sim)) < 1.0e-10
