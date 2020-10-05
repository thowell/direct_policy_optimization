function sample_general_objective(z,prob::DPOProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    u_policy = prob.u_policy

    T = prob.prob.T
    N = prob.N

    J = 0.0

    for t = 1:T
        nothing
    end

    return J
end

function ∇sample_general_objective!(∇obj,z,prob::DPOProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    u_policy = prob.u_policy
    T = prob.prob.T
    N = prob.N

    for t = 1:T
        nothing
    end
    return nothing
end
