function sample_objective(z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    u_policy = prob.u_policy
    Q = prob.Q
    R = prob.R
    H = prob.H
    T = prob.prob.T
    N = prob.N
    γ = prob.γ

    J = 0.0

    for t = 1:T
        x_nom = view(z,idx_nom.x[t])

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            J += (xi - x_nom)'*Q[t]*(xi - x_nom)
        end

        t > T-2 && continue
        u_nom = view(z,idx_nom.u[t])
        h_nom = z[idx_nom.h[t]]

        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]
            J += (ui - u_nom)'*R[t]*(ui - u_nom)
            J += (hi - h_nom)'*H[t]*(hi - h_nom)
        end
    end

    return γ*J/N
end

function ∇sample_objective!(∇obj,z,prob::SampleProblem)
    idx_nom = prob.idx_nom
    idx_sample = prob.idx_sample
    u_policy = prob.u_policy
    Q = prob.Q
    R = prob.R
    H = prob.H
    T = prob.prob.T
    N = prob.N
    γ = prob.γ

    for t = 1:T
        x_nom = view(z,idx_nom.x[t])
        for i = 1:N
            xi = view(z,idx_sample[i].x[t])

            ∇obj[idx_sample[i].x[t]] += 2.0*Q[t]*(xi - x_nom)*γ/N
            ∇obj[idx_nom.x[t]] -= 2.0*Q[t]*(xi - x_nom)*γ/N
        end

        t > T-2 && continue
        u_nom = view(z,idx_nom.u[t])
        h_nom = z[idx_nom.h[t]]

        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]

            ∇obj[idx_sample[i].u[t]] += 2.0*R[t]*(ui - u_nom)*γ/N
            ∇obj[idx_sample[i].h[t]] += 2.0*H[t]*(hi - h_nom)*γ/N

            ∇obj[idx_nom.u[t]] -= 2.0*R[t]*(ui - u_nom)*γ/N
            ∇obj[idx_nom.h[t]] -= 2.0*H[t]*(hi - h_nom)*γ/N
        end
    end
    return nothing
end
