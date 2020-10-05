function time_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nu = prob.model.nu
    T = prob.T
    model = prob.model

    for t = 1:T-2
        h = Z[idx.u[t]][end]
        h⁺ = Z[idx.u[t+1]][end]

        c[t] = h⁺ - h
    end

    return nothing
end

function time_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    idx = prob.idx
    nu = prob.model.nu
    T = prob.T
    model = prob.model

    shift = 0

    for t = 1:T-2
        r_idx = t
        ∇c[r_idx,idx.u[t][end]] = -1.0
        ∇c[r_idx,idx.h[t+1][end]] = 1.0
    end

    return nothing
end

function sparse_time_constraints_jacobian!(∇c,Z,prob::TrajectoryOptimizationProblem)
    T = prob.T

    shift = 0

    for t = 1:T-2

        s = 1
        ∇c[shift + s] = -1.0
        shift += s

        s = 1
        ∇c[shift + s] = 1.0
        shift += s
    end

    return nothing
end

function sparsity_time_jacobian(prob::TrajectoryOptimizationProblem;
        r_shift=0)
    idx = prob.idx
    T = prob.T

    row = []
    col = []

    if prob.free_time
        for t = 1:T-2
            r_idx = r_shift + t
            row_col!(row,col,r_idx,idx.u[t][end])
            row_col!(row,col,r_idx,idx.u[t+1][end])
        end
    end

    return collect(zip(row,col))
end
