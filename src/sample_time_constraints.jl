function sample_time_constraints!(c,z,prob::DPOProblem)
    T = prob.prob.T
    N = prob.N

    shift = 0

    for t = 1:T-2
        for i = 1:N
            hi⁺ = view(z,prob.idx_u[t+1])
            hi = view(z,prob.idx_u[t])
            c[shift + 1] = hi⁺[1] - hi[1]
            shift += 1
        end
    end
    nothing
end

function ∇sample_time_constraints!(∇c,z,prob::DPOProblem)
    T = prob.prob.T
    N = prob.N

    shift = 0

    for t = 1:T-2
        for i = 1:N
            r_idx = t
            c_idx = prob.idx_u[t+1]
            len = length(r)*length(c_idx)
            ∇c[shift + len] = 1.0
            shift += len

            c_idx = prob.idx_u[t]
            len = length(r)*length(c_idx)
            ∇c[shift + len] = -1.0
            shift += len
        end
    end

    nothing
end

function sparsity_jacobian_sample_time(prob::DPOProblem;
        r_shift=0)
    T = prob.prob.T
    N = prob.N

    row = []
    col = []

    for t = 1:T-2
        for i = 1:N
            r_idx = shift + t
            c_idx = prob.idx_u[t+1]
            row_col!(row,col,r_idx,c_idx)

            c_idx = prob.idx_u[t]
            row_col!(row,col,r_idx,c_idx)
        end
    end

    return collect(zip(row,col))
end
