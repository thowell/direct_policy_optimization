using LinearAlgebra, Plots, ForwardDiff, Ipopt, MathOptInterface
const MOI = MathOptInterface

struct Problem <: MOI.AbstractNLPEvaluator
    n_nlp
    m_nlp
    z_nom
    u_nom
    T
    n
    m
    Q
    Qf
    R
    model
    Δt
    enable_hessian::Bool
end

function primal_bounds(prob::MOI.AbstractNLPEvaluator)
    x_l = -Inf*ones(prob.n_nlp)
    x_u = Inf*ones(prob.n_nlp)
    return x_l, x_u
end

function constraint_bounds(prob::MOI.AbstractNLPEvaluator)
    c_l = zeros(prob.m_nlp)
    c_u = zeros(prob.m_nlp)
    return c_l, c_u
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    s = 0.0
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    R = prob.R
    z_nom = prob.z_nom
    u_nom = prob.u_nom

    for t = 1:T
        z = x[(t-1)*(n + m) .+ (1:n)]
        if t < T
            u = x[(t-1)*(n + m)+n .+ (1:m)]
            s += (z - z_nom[t])'*Q*(z - z_nom[t]) + (u - u_nom[t])'*R*(u - u_nom[t])
        else
            s += (z - z_nom[t])'*Qf*(z - z_nom[t])
        end
    end
    return s
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom

    for t = 1:T
        z = x[(t-1)*(n + m) .+ (1:n)]
        if t < T
            grad_f[(t-1)*(n + m) .+ (1:n)] = 2.0*Q*(z - z_nom[t])
            u = x[(t-1)*(n + m)+n .+ (1:m)]

            grad_f[(t-1)*(n + m) + n .+ (1:m)] = 2.0*R*(u - u_nom[t])
        else
            grad_f[(t-1)*(n + m) .+ (1:n)] = 2.0*Qf*(z - z_nom[t])
        end
    end
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt

    for t = 1:T-1
        z = x[(t-1)*(n + m) .+ (1:n)]
        z⁺ = x[t*(n + m) .+ (1:n)]
        u = x[(t-1)*(n + m) + n .+ (1:m)]
        g[(t-1)*n .+ (1:n)] = z⁺ - midpoint(model,z,u,prob.Δt)
    end

    z = x[1:n]
    g[(T-1)*n .+ (1:n)] = z - z_nom[1]
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt

    JAC = zeros(prob.m_nlp,prob.n_nlp)
    for t = 1:T-1
        r_idx = (t-1)*n .+ (1:n)
        c1_idx = (t-1)*(n + m) .+ (1:n)
        c2_idx = t*(n + m) .+ (1:n)
        c3_idx = (t-1)*(n + m) + n .+ (1:m)

        z = x[(t-1)*(n + m) .+ (1:n)]
        z⁺ = x[t*(n + m) .+ (1:n)]
        u = x[(t-1)*(n + m)+n .+ (1:m)]

        f1(w) = z⁺ - midpoint(model,w,u,prob.Δt)
        # f2(w) = z⁺ - midpoint(model,z,u,prob.Δt)
        f3(w) = z⁺ - midpoint(model,z,w,prob.Δt)

        JAC[r_idx,c1_idx] = ForwardDiff.jacobian(f1,z)
        JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
        JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,u)
    end
    r_idx = (T-1)*n .+ (1:n)
    c_idx = 1:n
    JAC[CartesianIndex.(r_idx,c_idx)] .= 1.0

    jac .= vec(JAC)
    return nothing
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end
    return row, col
end

function row_col_cartesian!(row,col,r,c)
    for i = 1:length(r)
        push!(row,convert(Int,r[i]))
        push!(col,convert(Int,c[i]))
    end
    return row, col
end

function sparsity_jacobian(prob::MOI.AbstractNLPEvaluator)

    row = []
    col = []

    r = 1:prob.m_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    return collect(zip(row,col))
end

MOI.features_available(prob::MOI.AbstractNLPEvaluator) = [:Grad, :Jac]
MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = sparsity_jacobian(prob)
MOI.hessian_lagrangian_structure(prob::MOI.AbstractNLPEvaluator) = []
MOI.eval_hessian_lagrangian(prob::MOI.AbstractNLPEvaluator, H, x, σ, μ) = nothing

function solve_ipopt(x0,prob::MOI.AbstractNLPEvaluator)
    x_l, x_u = primal_bounds(prob)
    c_l, c_u = constraint_bounds(prob)

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = 5000
    # solver.options["tol"] = 1.0e-3

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
        xi = MOI.SingleVariable(x[i])
        MOI.add_constraint(solver, xi, MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, xi, MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res
end

# sample controller
struct ProblemCtrl <: MOI.AbstractNLPEvaluator
    n_nlp
    m_nlp
    z_nom
    u_nom
    T
    n
    m
    Q
    Qf
    R
    A
    B
    model
    Δt
    N
    w
    enable_hessian::Bool
end

function MOI.eval_objective(prob::ProblemCtrl, x)
    s = 0.0
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    R = prob.R
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    N = prob.N

    for t = 1:T
        z = x[(t-1)*(n*N + m*n) .+ (1:n*N)]
        for i = 1:N
            zi = z[(i-1)*n .+ (1:n)]
            if t < T
                K = reshape(x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)],m,n)
                s += (zi - z_nom[t])'*(Q + K'*R*K)*(zi - z_nom[t])./N
            else
                s += (zi - z_nom[t])'*Qf*(zi - z_nom[t])./N
            end
        end
    end
    return s/N
end

function MOI.eval_objective_gradient(prob::ProblemCtrl, grad_f, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    N = prob.N

    for t = 1:T
        z = x[(t-1)*(n*N + m*n) .+ (1:n*N)]
        for i = 1:N
            zi = z[(i-1)*n .+ (1:n)]
            if t < T
                K_vec = x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)]
                K = reshape(K_vec,m,n)
                grad_f[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)] .= 2.0*(Q + K'*R*K)*(zi - z_nom[t])./N
                fk(w) = (zi - z_nom[t])'*(Q + reshape(w,m,n)'*R*reshape(w,m,n))*(zi - z_nom[t])
                grad_f[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)] .= ForwardDiff.gradient(fk,K_vec)./N
            else
                grad_f[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)] .= 2.0*Qf*(zi - z_nom[t])./N
            end
        end
    end
    return nothing
end

function MOI.eval_constraint(prob::ProblemCtrl,g,x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt
    A = prob.A
    B = prob.B
    N = prob.N
    w = prob.w

    for t = 1:T-1
        z = x[(t-1)*(n*N + m*n) .+ (1:n*N)]
        z⁺ = x[t*(n*N + m*n) .+ (1:n*N)]
        K = reshape(x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)],m,n)

        for i = 1:N
            zi = z[(i-1)*n .+ (1:n)]
            zi⁺ = z⁺[(i-1)*n .+ (1:n)]
            δz = zi - z_nom[t]
            δz⁺ = zi⁺ - z_nom[t+1]
            g[(t-1)*n*N + (i-1)*n .+ (1:n)] .= δz⁺ - (A[t] - B[t]*K)*δz - w[i][t+1]
        end
    end
    z = x[1:n*N]
    for i = 1:N
        zi = z[(i-1)*n .+ (1:n)]
        g[(T-1)*n*N + (i-1)*n .+ (1:n)] .= zi - (z_nom[1] + w[i][1])
    end
    return nothing
end

function MOI.eval_constraint_jacobian(prob::ProblemCtrl, jac, x)
    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt
    N = prob.N

    # JAC = zeros(prob.m_nlp,prob.n_nlp)

    shift = 0

    for t = 1:T-1
        z = x[(t-1)*(n*N + m*n) .+ (1:n*N)]
        z⁺ = x[t*(n*N + m*n) .+ (1:n*N)]
        K_vec = x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)]
        K = reshape(K_vec,m,n)

        for i = 1:N
            r_idx = (t-1)*n*N + (i-1)*n .+ (1:n)
            c1_idx = (t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)
            c2_idx = t*(n*N + m*n) + (i-1)*n .+ (1:n)
            c3_idx = (t-1)*(n*N + m*n) + n*N .+ (1:m*n)

            zi = z[(i-1)*n .+ (1:n)]
            zi⁺ = z⁺[(i-1)*n .+ (1:n)]

            δz = zi - z_nom[t]
            δz⁺ = zi⁺ - z_nom[t+1]

            f1(w) = δz⁺ - (A[t] - B[t]*K)*(w - z_nom[t])
            f2(w) = (w - z_nom[t+1]) - (A[t] - B[t]*K)*δz
            f3(w) = δz⁺ - (A[t] - B[t]*reshape(w,m,n))*δz

            # JAC[r_idx,c1_idx] = -(A[t] - B[t]*K)
            shift1 = n*n
            jac[shift .+ (1:shift1)] .= vec(-(A[t] - B[t]*K))
            shift += shift1

            # JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
            shift2 = n
            jac[shift .+ (1:shift2)] .= 1.0
            shift += shift2

            # JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,K_vec)
            shift3 = n*m*n
            jac[shift .+ (1:shift3)] .= vec(ForwardDiff.jacobian(f3,K_vec))
            shift += shift3
        end
    end

    for i = 1:N
        r_idx = (T-1)*n*N + (i-1)*n .+ (1:n)
        c_idx = (i-1)*n .+ (1:n)
        # JAC[CartesianIndex.(r_idx,c_idx)] .= 1.0
        shift4 = n
        jac[shift .+ (1:shift4)] .= 1.0
        shift += shift4
    end

    # jac .= vec(JAC)
    return nothing
end

function sparsity_jacobian(prob::ProblemCtrl)

    row = []
    col = []

    # r = 1:prob.m_nlp
    # c = 1:prob.n_nlp
    #
    # row_col!(row,col,r,c)

    n = prob.n
    m = prob.m
    T = prob.T
    Q = prob.Q
    Qf = prob.Qf
    z_nom = prob.z_nom
    u_nom = prob.u_nom
    model = prob.model
    Δt = prob.Δt
    N = prob.N

    for t = 1:T-1
        for i = 1:N
            r_idx = (t-1)*n*N + (i-1)*n .+ (1:n)
            c1_idx = (t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)
            c2_idx = t*(n*N + m*n) + (i-1)*n .+ (1:n)
            c3_idx = (t-1)*(n*N + m*n) + n*N .+ (1:m*n)

            # JAC[r_idx,c1_idx] = -(A[t] - B[t]*K)
            row_col!(row,col,r_idx,c1_idx)

            # JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
            row_col_cartesian!(row,col,r_idx,c2_idx)


            # JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,K_vec)
            row_col!(row,col,r_idx,c3_idx)

        end
    end

    for i = 1:N
        r_idx = (T-1)*n*N + (i-1)*n .+ (1:n)
        c_idx = (i-1)*n .+ (1:n)
        # JAC[CartesianIndex.(r_idx,c_idx)] .= 1.0
        row_col_cartesian!(row,col,r_idx,c_idx)
    end

    return collect(zip(row,col))
end

function sparsity_hessian_lagrangian(prob::ProblemCtrl)

    row = []
    col = []

    r = 1:prob.n_nlp
    c = 1:prob.n_nlp

    row_col!(row,col,r,c)

    # n = prob.n
    # m = prob.m
    # T = prob.T
    # Q = prob.Q
    # Qf = prob.Qf
    # z_nom = prob.z_nom
    # u_nom = prob.u_nom
    # model = prob.model
    # Δt = prob.Δt
    # N = prob.N

    # for t = 1:T-1
    #     for i = 1:N
    #         r_idx = (t-1)*n*N + (i-1)*n .+ (1:n)
    #         c1_idx = (t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)
    #         c2_idx = t*(n*N + m*n) + (i-1)*n .+ (1:n)
    #         c3_idx = (t-1)*(n*N + m*n) + n*N .+ (1:m*n)
    #
    #         # JAC[r_idx,c1_idx] = -(A[t] - B[t]*K)
    #         row_col!(row,col,r_idx,c1_idx)
    #
    #         # JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
    #         row_col_cartesian!(row,col,r_idx,c2_idx)
    #
    #
    #         # JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,K_vec)
    #         row_col!(row,col,r_idx,c3_idx)
    #
    #     end
    # end
    #
    # for i = 1:N
    #     r_idx = (T-1)*n*N + (i-1)*n .+ (1:n)
    #     c_idx = (i-1)*n .+ (1:n)
    #     # JAC[CartesianIndex.(r_idx,c_idx)] .= 1.0
    #     row_col_cartesian!(row,col,r_idx,c_idx)
    # end

    return collect(zip(row,col))
end

# MOI.features_available(prob::Problem) = [:Grad, :Jac]
# MOI.initialize(prob::Problem, features) = nothing
# MOI.jacobian_structure(prob::Problem) = sparsity(prob)
MOI.hessian_lagrangian_structure(prob::ProblemCtrl) = sparsity_hessian_lagrangian(prob)
function MOI.eval_hessian_lagrangian(prob::ProblemCtrl, H, x, σ, y)

    return nothing
end
