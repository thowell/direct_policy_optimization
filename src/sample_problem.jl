function obj_sample(z,idx_nom,idx_sample,Q,R,T,N)
    J = 0.0

    # sample
    for t = 1:T-1
        u_nom = view(z,idx_nom.u[t])
        x⁺_nom = view(z,idx_nom.x[t+1])

        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            xi⁺ = view(z,idx_sample[i].x[t+1])
            println((xi⁺ - x⁺_nom)'*Q[t+1]*(xi⁺ - x⁺_nom))
            J += (xi⁺ - x⁺_nom)'*Q[t+1]*(xi⁺ - x⁺_nom) + (ui - u_nom)'*R[t]*(ui - u_nom)
        end
    end

    return J
end

function con_sample!(c,z,idx_nom,idx_sample,idx_x_tmp,idx_K,Q,R,models,β,w,con,m_con,T,N,integration)
    shift = 0

    # dynamics + resampling (x1 is taken care of w/ primal bounds)
    β = 1.0
    w = 1.0e-1
    for t = 2:T-1
        h = view(z,idx_nom.h[t])
        x⁺_tmp = [view(z,idx_x_tmp[i].x[t]) for i = 1:N]
        xs⁺ = resample(x⁺_tmp,β=β,w=w) # resample

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            xi⁺ = view(z,idx_sample[i].x[t+1])

            c[shift .+ (1:nx)] = integration(models[i],x⁺_tmp[i],xi,ui,h)
            shift += nx
            c[shift .+ (1:nx)] = xs⁺[i] - xi⁺
            shift += nx
        end
    end

    # controller for samples
    for t = 1:T-1
        x_nom = view(z,idx_nom.x[t])
        u_nom = view(z,idx_nom.u[t])
        K = reshape(view(z,idx_K[t]),nu,nx)

        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])
            c[shift .+ (1:nu)] = ui + K*(xi - x_nom) - u_nom
            shift += nu
        end
    end

    # stage constraints samples
    for t = 2:T-1
        for i = 1:N
            xi = view(z,idx_sample[i].x[t])
            ui = view(z,idx_sample[i].u[t])

            con(view(c,shift .+ (1:m_con)),xi,ui)
            shift += m_con
        end
    end

    nothing
end

mutable struct SampleProblem <: Problem
    prob::TrajectoryOptimizationProblem
    N_nlp::Int # number of decision variables
    M_nlp::Int # number of constraints

    idx_nom
    idx_nom_z
    idx_sample
    idx_x_tmp
    idx_K

    x1

    Q
    R

    N::Int # number of samples
    models
    β
    w
end

function init_sample_problem(prob::TrajectoryOptimizationProblem,models,x1,Q,R;
        time=true,β=1.0,w=1.0)

    nx = prob.n
    nu = prob.m
    T = prob.T
    N = length(models)
    @assert N == 2*nx

    N_nlp = prob.N + N*(nx*T + nu*(T-1)) + N*(nx*(T-1)) + nu*nx*(T-1)
    M_nlp = prob.M + N*2*nx*(T-2) + N*nu*(T-1) + N*prob.m_con*(T-2)

    idx_nom = init_indices(nx,nu,T,time=time,shift=0)
    idx_nom_z = vcat(idx_nom.x...,idx_nom.u...,idx_nom.h...)
    shift = nx*T + nu*(T-1) + (T-1)
    idx_sample = [init_indices(nx,nu,T,time=false,shift=shift + (i-1)*(nx*T + nu*(T-1))) for i = 1:N]
    shift += N*(nx*T + nu*(T-1))
    idx_x_tmp = [init_indices(nx,0,T-1,time=false,shift=shift + (i-1)*(nx*(T-1))) for i = 1:N]
    shift += N*(nx*(T-1))
    idx_K = [shift + (t-1)*(nu*nx) .+ (1:nu*nx) for t = 1:T-1]

    return SampleProblem(prob,N_nlp,M_nlp,idx_nom,idx_nom_z,
        idx_sample,idx_x_tmp,idx_K,x1,Q,R,N,models,β,w)
end

function pack(X0,U0,h0,prob::SampleProblem)
    Z0 = zeros(prob.N_nlp)
    Z0[prob.idx_nom_z] = pack(X0,U0,h0,prob.prob)
    T = prob.prob.T
    N = prob.N

    for t = 1:T
        for i = 1:N
            Z0[prob.idx_sample[i].x[t]] = X0[t]
            t==T && continue
            Z0[prob.idx_sample[i].u[t]] = U0[t]
            Z0[prob.idx_x_tmp[i].x[t]] = X0[t+1]
        end
    end
    return Z0
end

function unpack(Z0,prob::SampleProblem)
    T = prob.prob.T
    N = prob.N

    X_nom = [Z0[prob.idx_nom.x[t]] for t = 1:T]
    U_nom = [Z0[prob.idx_nom.u[t]] for t = 1:T-1]
    H_nom = [Z0[prob.idx_nom.h[t]] for t = 1:T-1]

    X_sample = [[Z0[prob.idx_sample[i].x[t]] for t = 1:T] for i = 1:N]
    U_sample = [[Z0[prob.idx_sample[i].u[t]] for t = 1:T-1] for i = 1:N]

    return X_nom, U_nom, H_nom, X_sample, U_sample
end

function init_MOI_Problem(prob::SampleProblem)
    return MOIProblem(prob.N_nlp,prob.M_nlp,prob,false)
end


function primal_bounds(prob::SampleProblem)
    Zl = -Inf*ones(prob.N_nlp)
    Zu = Inf*ones(prob.N_nlp)

    # nominal bounds
    Zl_nom,Zu_nom = primal_bounds(prob.prob)
    Zl[prob.idx_nom_z] = Zl_nom
    Zu[prob.idx_nom_z] = Zu_nom

    # sample initial conditions
    for i = 1:prob.N
        Zl[prob.idx_sample[i].x[1]] = prob.x1[i]
        Zu[prob.idx_sample[i].x[1]] = prob.x1[i]
    end

    # sample state and control bounds
    for t = 1:prob.prob.T-1
        for i = 1:prob.N
            Zl[prob.idx_sample[i].u[t]] = prob.prob.ul[t]
            Zu[prob.idx_sample[i].u[t]] = prob.prob.uu[t]

            t==1 && continue
            Zl[prob.idx_sample[i].x[t]] = prob.prob.xl[t]
            Zu[prob.idx_sample[i].x[t]] = prob.prob.xu[t]
        end
    end

    return Zl,Zu
end

function constraint_bounds(prob::SampleProblem)
    cl = zeros(prob.M_nlp)
    cu = zeros(prob.M_nlp)

    # nominal constraints
    M_nom = prob.prob.M
    cl_nom, cu_nom = constraint_bounds(prob.prob)

    cl[1:M_nom] = cl_nom
    cu[1:M_nom] = cu_nom

    # sample stage constraints
    cu[prob.N*2*prob.prob.n*(prob.prob.T-2) + prob.N*prob.prob.m*(prob.prob.T-1) .+ (1:prob.N*prob.prob.m_con*(prob.prob.T-2))] .= Inf

    return cl,cu
end

function eval_objective(prob::SampleProblem,Z)
    (eval_objective(prob.prob,view(Z,prob.idx_nom_z))
        + obj_sample(Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.prob.T,prob.N))
end

function eval_objective_gradient!(∇l,Z,prob::SampleProblem)
    eval_objective_gradient!(view(∇l,prob.idx_nom_z),view(Z,prob.idx_nom_z),
        prob.prob)
    tmp_obj(z) = obj_sample(Z,prob.idx_nom,prob.idx_sample,prob.Q,prob.R,prob.prob.T,prob.N)

    ∇l += ForwardDiff.gradient(tmp_obj,Z)
    return nothing
end

function eval_constraint!(c,Z,prob::SampleProblem)
   M_nom = prob.prob.M
   mm = prob.M_nlp - M_nom
   eval_constraint!(view(c,1:M_nom),view(Z,prob.idx_nom_z),prob.prob)
   con_sample!(view(c,M_nom .+ (1:mm)),Z,prob.idx_nom,prob.idx_sample,prob.idx_x_tmp,
        prob.idx_K,prob.Q,prob.R,prob.models,prob.β,prob.w,prob.prob.con,
        prob.prob.m_con,prob.prob.T,prob.N,prob.prob.integration)
   return nothing
end

function eval_constraint_jacobian!(∇c,Z,prob::SampleProblem)
    len = length(sparsity_jacobian(prob.prob))
    eval_constraint_jacobian!(view(∇c,1:len),view(Z,prob.idx_nom_z),prob.prob)

    M_nom = prob.prob.M
    mm = prob.M_nlp - M_nom

    con_tmp(c,z) = con_sample!(c,z,prob.idx_nom,prob.idx_sample,prob.idx_x_tmp,
         prob.idx_K,prob.Q,prob.R,prob.models,prob.β,prob.w,prob.prob.con,
         prob.prob.m_con,prob.prob.T,prob.N,prob.prob.integration)

    ∇c[len .+ (1:prob.M_nlp*prob.N_nlp)] = vec(ForwardDiff.jacobian(con_tmp,zeros(prob.M_nlp),Z))
    return nothing
end

function sparsity_jacobian(prob::SampleProblem)
    M_nom = prob.prob.M
    mm = prob.M_nlp - M_nom
    collect([sparsity_jacobian(prob.prob)...,
        sparsity_jacobian(prob.N_nlp,mm; shift_r=prob.prob.M)...])
end
