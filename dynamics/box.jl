using Rotations

# Box
mutable struct Box{T}
    m::T
    J::T
    μ::T

    r

    nx
    nu
    nu_ctrl
    nc
    nf
    nb

    n_corners
    corner_offset

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s

    α
end

# Dimensions
nq = 6 # configuration dim
nu_ctrl = 3
nc = 8 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc*nf

nx = nq
nu = nu_ctrl + nc + nb + nc + nb + 1

idx_u = (1:nu_ctrl)
idx_λ = nu_ctrl .+ (1:nc)
idx_b = nu_ctrl + nc .+ (1:nb)
idx_ψ = nu_ctrl + nc + nb .+ (1:nc)
idx_η = nu_ctrl + nc + nb + nc .+ (1:nb)
idx_s = nu_ctrl + nc + nb + nc + nb + 1

# Parameters
μ = 1.0  # coefficient of friction
m = 1.   # mass
J = 1.0/12.0*m*((2.0*r)^2 + (2.0*r)^2)


# Kinematics
r = 0.5
c1 = @SVector [r, r, r]
c2 = @SVector [r, r, -r]
c3 = @SVector [r, -r, r]
c4 = @SVector [r, -r, -r]
c5 = @SVector [-r, r, r]
c6 = @SVector [-r, r, -r]
c7 = @SVector [-r, -r, r]
c8 = @SVector [-r, -r, -r]

corner_offset = @SVector [c1,c2,c3,c4,c5,c6,c7,c8]

# Methods
M_func(p::Box,q) = Diagonal(@SVector [p.m, p.m, p.m, p.J, p.J, p.J])

G_func(p::Box,q) = @SVector [0., 0., p.m*9.81, 0., 0., 0.]

C_func(::Box,qk,qn) = @SVector zeros(nq)

function ϕ_func(m::Box,q)
    v = q[4:6]
    # ss = zeros(eltype(v),3,3)
    # ss[1,2] = -v[3]
    # ss[1,3] = v[2]
    # ss[2,1] = v[3]
    # ss[2,3] = -v[1]
    # ss[3,1] = -v[2]
    # ss[3,2] = v[1]
    tmp = MRP(v...)

    @SVector [(q[1:3]+tmp*m.corner_offset[1])[3],
              (q[1:3]+tmp*m.corner_offset[2])[3],
              (q[1:3]+tmp*m.corner_offset[3])[3],
              (q[1:3]+tmp*m.corner_offset[4])[3],
              (q[1:3]+tmp*m.corner_offset[5])[3],
              (q[1:3]+tmp*m.corner_offset[6])[3],
              (q[1:3]+tmp*m.corner_offset[7])[3],
              (q[1:3]+tmp*m.corner_offset[8])[3]]
end

function B_func(::Box,q)
    mrp = MRP(q[4:6]...)
    @SMatrix [0. 0. 0. mrp[1,1] mrp[1,2] mrp[1,3];
              0. 0. 0. mrp[2,1] mrp[2,2] mrp[2,3];
              0. 0. 0. mrp[3,1] mrp[3,2] mrp[3,3]]
#     @SMatrix [0. 0. 0. 1. 0. 0.;
#               0. 0. 0. 0. 1. 0.;
#               0. 0. 0. 0. 0. 1.]
end

function N_func(m::Box,q)
    tmp(z) = ϕ_func(m,z)
    ForwardDiff.jacobian(tmp,q)
end


function P_func(m::Box,q)
    map = [1. 0.;
           0. 1.;
           -1. 0.;
           0. -1.]

    function p(x)
        v = x[4:6]
        # ss = zeros(eltype(v),3,3)
        # ss[1,2] = -v[3]
        # ss[1,3] = v[2]
        # ss[2,1] = v[3]
        # ss[2,3] = -v[1]
        # ss[3,1] = -v[2]
        # ss[3,2] = v[1]
        # tmp = (I + 4*((1-v'*v)*I + 2*ss)*ss/(1+v'*v)^2)
        tmp = MRP(v...)

        [map*(x[1:3]+tmp*m.corner_offset[1])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[2])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[3])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[4])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[5])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[6])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[7])[1:2];
         map*(x[1:3]+tmp*m.corner_offset[8])[1:2]]
    end
    ForwardDiff.jacobian(p,q)
end

function friction_cone(model,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]

    @SVector [model.μ*λ[1] - sum(b[1:4]),
              model.μ*λ[2] - sum(b[5:8]),
              model.μ*λ[3] - sum(b[9:12]),
              model.μ*λ[4] - sum(b[13:16]),
              model.μ*λ[5] - sum(b[17:20]),
              model.μ*λ[6] - sum(b[21:24]),
              model.μ*λ[7] - sum(b[25:28]),
              model.μ*λ[8] - sum(b[29:32])]
end

function maximum_energy_dissipation(model,x2,x3,u,h)
    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1]*ones(4);
               ψ[2]*ones(4);
               ψ[3]*ones(4);
               ψ[4]*ones(4);
               ψ[5]*ones(4);
               ψ[6]*ones(4);
               ψ[7]*ones(4);
               ψ[8]*ones(4)]

    η = u[model.idx_η]

    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

α_Box = 1.0

model = Box(m,J,μ,
             r,
             nx,nu,nu_ctrl,
             nc,nf,nb,
             8,corner_offset,
             idx_u,
             idx_λ,
             idx_b,
             idx_ψ,
             idx_η,
             idx_s,
             α_Box)


function visualize!(vis,m::Box,q;
        Δt=0.1)

    setobject!(vis["box"], HyperRectangle(Vec(-r,-r,-r),Vec(2r,2r,2r)))

    for i = 1:m.n_corners
        setobject!(vis["corner$i"],HyperSphere(Point3f0(0),
            convert(Float32,0.05)),
            MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))
    end

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim,t) do

            settransform!(vis["box"], compose(Translation(q[t][1:3]...),LinearMap(MRP(q[t][4:6]...))))

            for i = 1:m.n_corners
                settransform!(vis["corner$i"],
                    Translation((q[t][1:3]+MRP(q[t][4:6]...)*(corner_offset[i]))...))

            end
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
