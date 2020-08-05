
obj_sample(Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
tmp_obj(z) = obj_sample(z,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
ForwardDiff.gradient(tmp_obj,Z0_sample)
_∇obj = zero(Z0_sample)
∇obj_sample!(_∇obj,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.Q,prob_sample.R,prob_sample.H,T,prob_sample.N,prob_sample.γ)
@assert norm(_∇obj - ForwardDiff.gradient(tmp_obj,Z0_sample)) < 1.0e-12

c0 = zeros(prob_sample.M_sample)
con_sample!(c0,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N)
tmp_con(c,z) = con_sample!(c,z,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N)
ForwardDiff.jacobian(tmp_con,c0,Z0_sample)
spar = sparsity_jacobian_sample(prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob.m_stage,prob.T,prob_sample.N)
∇c_vec = zeros(length(spar))
∇con_sample_vec!(∇c_vec,Z0_sample,prob_sample.idx_nom,prob_sample.idx_sample,prob_sample.idx_x_tmp,prob_sample.idx_K,prob_sample.Q,prob_sample.R,prob_sample.models,prob_sample.β,prob_sample.w,prob.m_stage,prob.T,prob_sample.N)
∇c = zeros(prob_sample.M_sample,prob_sample.N_nlp)

for (i,k) in enumerate(spar)
    ∇c[k[1],k[2]] = ∇c_vec[i]
end

@assert norm(vec(∇c) - vec(ForwardDiff.jacobian(tmp_con,c0,Z0_sample))) < 1.0e-12
@assert sum(∇c) - sum(ForwardDiff.jacobian(tmp_con,c0,Z0_sample)) < 1.0e-12
