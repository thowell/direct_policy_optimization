function policy_constraints!(c,z,prob::DPOProblem)
    β = prob.β_resample

    shift = 0

    # controller for samples
    for t = 1:T-1
        x_nom = view(z,prob.prob.idx.x[t])
        u_nom = view(z,prob.prob.idx.u[t])
        μ = view(z,prob.idx_μ[t])
        L = view(z,prob.idx_P[t])
        K = view(z,prob.idx_K[t])

        for i = 1:N

            ui = view(z,idx_sample[i].u[t])
            c[shift .+ (1:nu_policy)] =
            shift += prob.nu_policy
        end
    end

    nothing
end

function ∇policy_constraints!(∇c,z,prob::DPOProblem)


    nothing
end

function sparsity_jacobian_policy(prob::DPOProblem;
        r_shift=0)

    row = []
    col = []

    return collect(zip(row,col))
end
