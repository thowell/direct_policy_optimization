# particle
mutable struct Particle{T}
    m::T
    μ::T
    Δt::T

    nx
    nu
    nu_ctrl
    nc
    nf
    nb

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s

    m_contact
end

# Dimensions
nq = 3 # configuration dim
nu_ctrl = 3
nc = 1 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc*nf

nx = nq
nu = nu_ctrl + nc + nb + nc + nb + 1

m_contact = nb + nc + nc + 3

idx_u = (1:nu_ctrl)
idx_λ = nu_ctrl .+ (1:nc)
idx_b = nu_ctrl + nc .+ (1:nb)
idx_ψ = nu_ctrl + nc + nb .+ (1:nc)
idx_η = nu_ctrl + nc + nb + nc .+ (1:nb)
idx_s = nu_ctrl + nc + nb + nc + nb + 1

# Parameters
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
m = 1.   # mass

# Methods
function M_func(p::Particle,q)
    @SMatrix [p.m 0. 0.;
              0. p.m 0.;
              0. 0. p.m]
end

G_func(p::Particle,q) = @SVector [0., 0., p.m*9.8]

C_func(::Particle,qk,qn) = @SVector zeros(nq)

function ϕ_func(::Particle,q)
    @SVector[q[3]]
end

B_func(::Particle,q) = @SMatrix [1. 0. 0.;
                                 0. 1. 0.;
                                 0. 0. 1.]

N_func(::Particle,q) = @SMatrix [0. 0. 1.]

P_func(::Particle,q) = @SMatrix [1. 0. 0.;
                                 -1. 0. 0.;
                                 0. 1. 0.;
                                 0. -1. 0.]

function discrete_dynamics(model,x1,x2,x3,u,h,t)
    u_ctrl = u[model.idx_u]
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    (1/h[1])*(M_func(model,x1)*(x2 - x1) - M_func(model,x2)*(x3 - x2)) + h[1]*(0.5*C_func(model,x2,x3) - G_func(model,x2)) + transpose(B_func(model,x3))*u_ctrl + transpose(N_func(model,x3))*λ + transpose(P_func(model,x3))*b
end

function friction_cone(model,u)
    @SVector [model.μ*u[model.idx_λ[1]] - sum(u[model.idx_b])]
end

function maximum_energy_dissipation(model,x2,x3,u,h)
    ψ_stack = u[model.idx_ψ][1]*ones(model.nb)
    η = u[model.idx_η]
    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

model = Particle(m,μ,Δt,
                 nx,nu,nu_ctrl,
                 nc,nf,nb,
                 idx_u,
                 idx_λ,
                 idx_b,
                 idx_ψ,
                 idx_η,
                 idx_s,
                 m_contact)

function visualize!(vis,p::Particle,q; r=0.25)

    setobject!(vis["particle"], HyperRectangle(Vec(0,0,0),Vec(2r,2r,2r)))

    anim = MeshCat.Animation(convert(Int,floor(1/p.Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim,t) do
            settransform!(vis["particle"], Translation(q[t][1:3]...))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end