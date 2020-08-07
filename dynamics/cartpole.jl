mutable struct Cartpole{T}
    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
    nx::Int   # state dimension
    nu::Int   # control dimension
end

function dynamics(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,3:4) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

nx,nu = 4,1
model = Cartpole(1.0,0.2,0.5,9.81,nx,nu)
model_nominal = model

mutable struct CartpoleFriction{T}
    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
    μ::T      # friction coefficient
    nx::Int   # state dimension
    nu::Int   # control dimension
end

function dynamics(model::CartpoleFriction, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]

    qdd = SVector{2}(-H\(C*view(x,3:4) + G - B*(u[1] - (u[2] - u[3]))))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

nx_friction,nu_friction = 4,7
model_friction = CartpoleFriction(1.0,0.2,0.5,9.81,0.1,nx_friction,nu_friction)

function c_stage!(c,x,u,t,model::CartpoleFriction)
    v = x[3]
    β = u[2:3]
    ψ = u[4]
    η = u[5:6]
    s = u[7]

    n = (model.mc + model.mp)*model.g

    c[1] = v + ψ - η[1]
    c[2] = -v + ψ - η[2]
    c[3] = model.μ*n - sum(β)
    c[4] = s - ψ*(model.μ*n - sum(β))
    c[5] = s - β'*η

    nothing
end

m_stage_friction = 5

ul_friction = zeros(nu_friction)
ul_friction[1] = -10.0
uu_friction = Inf*ones(nu_friction)
uu_friction[1] = 10.0

stage_friction_ineq = (3:5)
(model.mc + model.mp)*model.g

const α_cartpole_friction = 100.0

mutable struct PenaltyObjective{T} <: Objective
    α::T
end

function objective(Z,l::PenaltyObjective,model::CartpoleFriction,idx,T)
    J = 0
    for t = 1:T-1
        s = Z[idx.u[t][7]]
        J += s
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model::CartpoleFriction,idx,T)
    for t = 1:T-1
        u = Z[idx.u[t][7]]
        ∇l[idx.u[t][7]] += l.α
    end
    return nothing
end

function sample_general_objective(z,prob::SampleProblem)
    idx_sample = prob.idx_sample
    T = prob.prob.T
    N = prob.N

    J = 0.0

    for t = 1:T-1
        for i = 1:N
            s = z[idx_sample[i].u[t][7]]
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
            ∇obj[idx_sample[i].u[t][7]] += α_cartpole_friction
        end
    end
    return nothing
end
