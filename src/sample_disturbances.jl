function obj_l1(z,prob::SampleProblem)
    idx_slack = prob.idx_slack
    α = prob.α

    J = 0.0
    N = length(idx_slack)
    for t = 1:T-1
        for i = 1:N
            s = view(z,idx_slack[i][t])
            J += sum(s)
        end
    end
    return α*J
end

# obj_l1(Z0_sample,prob_sample.idx_slack,prob_sample.α)

function ∇obj_l1!(∇obj,z,prob::SampleProblem)
    idx_slack = prob.idx_slack
    α = prob.α
    N = length(idx_slack)
    for t = 1:T-1
        for i = 1:N
            ∇obj[idx_slack[i][t]] .+= α
        end
    end
    nothing
end

function sample_disturbance_constraints!(c,z,prob::SampleProblem)
    idx_uw = prob.idx_uw
    idx_slack = prob.idx_slack
    T = prob.prob.T

    shift = 0

    N = length(idx_uw)
    nx = length(idx_uw[1][1])
    for t = 1:T-1
        for i = 1:N
            ti = view(z,idx_slack[i][t])
            uwi = view(z,idx_uw[i][t])

            c[shift .+ (1:nx)] = ti - uwi
            shift += nx

            c[shift .+ (1:nx)] = uwi + ti
            shift += nx
        end
    end
    nothing
end

function ∇sample_disturbance_constraints!(∇c,z,prob::SampleProblem)
    idx_uw = prob.idx_uw
    idx_slack = prob.idx_slack
    T = prob.prob.T

    shift = 0
    s = 0

    N = length(idx_uw)
    nx = length(idx_uw[1][1])

    for t = 1:T-1
        for i = 1:N
            ti = view(z,idx_slack[i][t])
            uwi = view(z,idx_uw[i][t])

            # c[shift .+ (1:nw[i])] = ti - uwi
            r_idx = shift .+ (1:nx)

            c_idx = idx_uw[i][t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(-Diagonal(ones(nx)))
            s += len

            c_idx = idx_slack[i][t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
            s += len

            shift += nx

            # c[shift .+ (1:nw[i])] = uwi + ti
            r_idx = shift .+ (1:nx)

            c_idx = idx_uw[i][t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
            s += len

            c_idx = idx_slack[i][t]
            len = length(r_idx)*length(c_idx)
            ∇c[s .+ (1:len)] = vec(Diagonal(ones(nx)))
            s += len

            shift += nx
        end
    end
    nothing
end

function sparsity_jacobian_sample_disturbance(prob::SampleProblem;
        r_shift=0)

    idx_uw = prob.idx_uw
    idx_slack = prob.idx_slack
    T = prob.prob.T

    shift = 0
    s = 0

    row = []
    col = []

    N = length(idx_uw)
    nx = length(idx_uw[1][1])

    for t = 1:T-1
        for i = 1:N
            # ti = view(z,idx_slack[i][t])
            # uwi = view(z,idx_sample[i].u[t][nu .+ (1:nw[i])])

            # c[shift .+ (1:nw[i])] = ti - uwi
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_uw[i][t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(-Diagonal(ones(nw[i])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_slack[i][t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(ones(nw[i])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            shift += nx

            # c[shift .+ (1:nw[i])] = uwi + ti
            r_idx = r_shift + shift .+ (1:nx)

            c_idx = idx_uw[i][t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(ones(nw[i])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            c_idx = idx_slack[i][t]
            # len = length(r_idx)*length(c_idx)
            # ∇c[s .+ (1:len)] = vec(Diagonal(ones(nw[i])))
            # s += len
            row_col!(row,col,r_idx,c_idx)

            shift += nx
        end
    end
    return collect(zip(row,col))
end

function unpack_disturbance(Z0,prob::SampleProblem)
    T = prob.prob.T
    N = prob.N

    Uw_sample = [[Z0[prob.idx_uw[i][t]] for t = 1:T-1] for i = 1:N]

    return Uw_sample
end
