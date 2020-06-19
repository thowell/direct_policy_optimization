using LinearAlgebra, Plots, ForwardDiff, Ipopt, MathOptInterface
const MOI = MathOptInterface

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
    z0
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
                s += (zi - z_nom[t])'*(Q + K'*R*K)*(zi - z_nom[t])
            else
                s += (zi - z_nom[t])'*Qf*(zi - z_nom[t])
            end
        end
    end
    return s
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
                K = reshape(x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)],m,n)
                grad_f[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)] .= 2.0*(Q + K'*R*K)*(zi - z_nom[t])

                fk(w) = (zi - z_nom[t])'*(Q + reshape(w,m,n)'*R*reshape(w,m,n))*(zi - z_nom[t])
                grad_f[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)] .= ForwardDiff.gradient(fk,x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)])
            else
                grad_f[(t-1)*(n*N + m*n) + (i-1)*n .+ (1:n)] .= 2.0*Qf*(zi - z_nom[t])
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
    z0 = prob.z0
    N = prob.N

    for t = 1:T-1
        z = x[(t-1)*(n*N + m*n) .+ (1:n*N)]
        z⁺ = x[t*(n*N + m*n) .+ (1:n*N)]
        K = reshape(x[(t-1)*(n*N + m*n) + n*N .+ (1:m*n)],m,n)

        for i = 1:N
            zi = z[(i-1)*n .+ (1:n)]
            zi⁺ = z⁺[(i-1)*n .+ (1:n)]
            δz = zi - z_nom[t]
            δz⁺ = zi⁺ - z_nom[t+1]
            δu = -K*δz
            g[(t-1)*n*N + (i-1)*n .+ (1:n)] .= δz⁺ - (A[t] - B[t]*K)*δz
        end
    end
    # z = x[1:n*N]
    # for i = 1:N
    #     zi = z[(i-1)*n .+ (1:n)]
    #     g[(T-1)*n*N + (i-1)*n .+ (1:n)] .= zi - z0[i]
    # end
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

    JAC = zeros(prob.m_nlp,prob.n_nlp)
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

            JAC[r_idx,c1_idx] = -(A[t] - B[t]*K)
            JAC[CartesianIndex.(r_idx,c2_idx)] .= 1.0
            JAC[r_idx,c3_idx] = ForwardDiff.jacobian(f3,K_vec)
        end
    end

    # for i = 1:N
    #     r_idx = (T-1)*n*N + (i-1)*n .+ (1:n)
    #     c_idx = (i-1)*n .+ (1:n)
    #     JAC[CartesianIndex.(c_idx,r_idx)] .= 1.0
    # end

    jac .= vec(JAC)
    return nothing
end

# function sparsity(prob::Problem)
#
#     row = []
#     col = []
#
#     r = 1:prob.m_nlp
#     c = 1:prob.n_nlp
#
#     row_col!(row,col,r,c)
#
#     return collect(zip(row,col))
# end

# MOI.features_available(prob::Problem) = [:Grad, :Jac]
# MOI.initialize(prob::Problem, features) = nothing
# MOI.jacobian_structure(prob::Problem) = sparsity(prob)
# MOI.hessian_lagrangian_structure(prob::Problem) = []
# MOI.eval_hessian_lagrangian(prob::Problem, H, x, σ, μ) = nothing
