obj_sample(Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
tmp_obj(z) = obj_sample(z,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
ForwardDiff.gradient(tmp_obj,Z0_sample)
_∇obj = zero(Z0_sample)
∇obj_sample!(_∇obj,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
@assert norm(_∇obj - ForwardDiff.gradient(tmp_obj,Z0_sample)) < 1.0e-12

c0 = zeros(prob_sample.M_sample)
con_sample!(c0,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.idx_uw,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N,disturbance_ctrl=prob_sample.disturbance_ctrl)
tmp_con(c,z) = con_sample!(c,z,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.idx_uw,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N,disturbance_ctrl=prob_sample.disturbance_ctrl)
ForwardDiff.jacobian(tmp_con,c0,Z0_sample)
spar = sparsity_jacobian_sample(prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.idx_uw,prob.m_stage,prob.T,prob_sample.N,disturbance_ctrl=prob_sample.disturbance_ctrl)
∇c_vec = zeros(length(spar))
∇con_sample_vec!(∇c_vec,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.idx_uw,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N,disturbance_ctrl=prob_sample.disturbance_ctrl)
∇c = zeros(prob_sample.M_sample,prob_sample.N_nlp)

for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end

@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_con,c0,Z0_sample))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_con,c0,Z0_sample)) < 1.0e-10

Z0_test = rand(prob_sample.N_nlp)
tmp_o(z) = eval_objective(prob_sample,z)
∇obj_ = zeros(prob_sample.N_nlp)
eval_objective_gradient!(∇obj_,Z0_sample,prob_sample)
@assert norm(ForwardDiff.gradient(tmp_o,Z0_sample) - ∇obj_) < 1.0e-10

prob_sample.disturbance_ctrl
c0 = zeros(prob_sample.M_nlp)
eval_constraint!(c0,Z0_test,prob_sample)
tmp_c(c,z) = eval_constraint!(c,z,prob_sample)
ForwardDiff.jacobian(tmp_c,c0,Z0_test)

spar = sparsity_jacobian(prob_sample)
∇c_vec = zeros(length(spar))
∇c = zeros(prob_sample.M_nlp,prob_sample.N_nlp)
eval_constraint_jacobian!(∇c_vec,Z0_test,prob_sample)
for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end

@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_c,c0,Z0_test))) < 1.0e-10
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_c,c0,Z0_test)) < 1.0e-10
