function sample_general_objective(z,prob::SampleProblem)
    idx_sample = prob.idx_sample
    T = prob.prob.T
    N = prob.N

    J = 0.0

    for t = 1:T-2
        for i = 1:N
            s = z[idx_sample[i].u[t][prob.models[i].idx_s]]
            J += s
        end
    end

    return α_cartpole_friction*J
end

function ∇sample_general_objective!(∇obj,z,prob::SampleProblem)
    idx_sample = prob.idx_sample
    T = prob.prob.T
    N = prob.N

    for t = 1:T-1
        for i = 1:N
            ∇obj[idx_sample[i].u[t][prob.models[i].idx_s]] += α_cartpole_friction
        end
    end
    return nothing
end
