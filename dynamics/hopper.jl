# 2D Hopper - Dynamically Stable Legged Locomotion. Raibert 1989
mutable struct Hopper
    mb
    ml
    Jb
    Jl
    r
    μ
    g
    Δt

    qL
    qU

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
nq = 5 # configuration dim
nu_ctrl = 2 # control dim
nc = 1 # number of contact points
nf = 2 # number of faces for friction cone
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
g = 9.81 # gravity
Δt = 0.1 # time step
μ = 0.5  # coefficient of friction
mb = 10. # body mass
ml = 1.  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

# Kinematics
r = 0.7
kinematics(::Hopper,q) = [q[1] + q[3]*sin(q[5]), q[2] - q[3]*cos(q[5])]

# Methods
M_func(h::Hopper,q) = Diagonal(@SVector [h.mb+h.ml, h.mb+h.ml, h.ml, h.Jb, h.Jl])

G_func(h::Hopper,q) = @SVector [0., (h.mb+h.ml)*h.g, 0., 0., 0.]

C_func(h::Hopper,qk,qn) = @SVector [0.0, 0.0, 0.0, 0.0, 0.0]

function ϕ_func(::Hopper,q)
    @SVector [q[2] - q[3]*cos(q[5])]
end

N_func(::Hopper,q) = @SMatrix [0. 1. -cos(q[5]) 0. q[3]*sin(q[5])]

# function P(::Hopper2D,q)
#     @SMatrix [1. 0. sin(q[5]) 0. q[3]*cos(q[5])]
# end
function P_func(::Hopper,q)
    @SMatrix [1. 0. sin(q[5]) 0. q[3]*cos(q[5]);
              -1. 0. -sin(q[5]) 0. -q[3]*cos(q[5])]
end

B_func(::Hopper,q) = @SMatrix [0. 0. 0. 1. -1.;
                            0. 0. 1. 0. 0.]

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

qL = -Inf*ones(nq)
qU = Inf*ones(nq)
qL[3] = r/2.0
qU[3] = r


model = Hopper(mb,ml,Jb,Jl,r,μ,g,Δt,qL,qU,
                nx,nu,nu_ctrl,
                nc,nf,nb,
                idx_u,
                idx_λ,
                idx_b,
                idx_ψ,
                idx_η,
                idx_s,
                m_contact)

# Visualization
function visualize!(vis,model::Hopper,q;verbose=false)
    r_foot = 0.05
    r_leg = 0.5*r_foot
    setobject!(vis["body"], HyperSphere(Point3f0(0),
        convert(Float32,0.1)),
        MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))
    setobject!(vis["foot"], HyperSphere(Point3f0(0),
        convert(Float32,r_foot)),
        MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))

    n_leg = 100
    for i = 1:n_leg
        setobject!(vis["leg$i"], HyperSphere(Point3f0(0),
            convert(Float32,r_leg)),
            MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
    end
    p_leg = [zeros(3) for i = 1:n_leg]
    anim = MeshCat.Animation(convert(Int,floor(1/model.Δt)))

    for t = 1:length(q)
        p_body = [q[t][1],0.,q[t][2]]
        p_foot = [kinematics(model,q[t])[1], 0., kinematics(model,q[t])[2]]

        if verbose
            println("foot height: $(p_foot[3])")
        end

        q_tmp = Array(copy(q[t]))
        r_range = range(0,stop=q[t][3],length=n_leg)
        for i = 1:n_leg
            q_tmp[3] = r_range[i]
            p_leg[i] = [kinematics(model,q_tmp)[1], 0., kinematics(model,q_tmp)[2]]
        end
        q_tmp[3] = q[t][3]
        p_foot = [kinematics(model,q_tmp)[1], 0., kinematics(model,q_tmp)[2]]

        z_shift = [0.;0.;r_foot]
        MeshCat.atframe(anim,t) do
            settransform!(vis["body"], Translation(p_body + z_shift))
            settransform!(vis["foot"], Translation(p_foot + z_shift))

            for i = 1:n_leg
                settransform!(vis["leg$i"], Translation(p_leg[i] + z_shift))
            end
        end
    end
    MeshCat.setanimation!(vis,anim)
end
