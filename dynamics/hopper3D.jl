# 2D Hopper - Dynamically Stable Legged Locomotion. Raibert 1989
mutable struct Hopper
    mb
    ml
    Jb
    Jl
    r
    μ
    g

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

    α
end

# Dimensions
nq = 7 # configuration dim
nu_ctrl = 3 # control dim
nc = 3 # number of contact points
nf = 4 # number of faces for friction cone
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
g = 9.81 # gravity
μ = 1.0  # coefficient of friction
mb = 10. # body mass
ml = 1.  # leg mass
Jb = 2.5 # body inertia
Jl = 0.25 # leg inertia

# Kinematics
r = 0.7

M_func(h::Hopper,q) = Diagonal(@SVector [h.mb + h.ml, h.mb + h.ml, h.mb + h.ml, h.Jb + h.Jl, h.Jb + h.Jl, h.Jb, h.ml])
G_func(h::Hopper,q) = @SVector [0., 0., (h.mb + h.ml)*h.g, 0., 0., 0., 0.]
C_func(::Hopper,qk,qn) = @SVector zeros(nq)
B_func(::Hopper,q) = @SMatrix [0. 0. 0. 1. 0. 0. 0.;
                               0. 0. 0. 0. 1. 0. 0.;
                               0. 0. 0. 0. 0. 0. 1.]

function kinematics(::Hopper,q)
    v = q[4:6]

    a = v[1]
    b = v[2]
    c = v[3]

    x = 1 - (a*a + b*b + c*c)
    y = (1 + a*a + b*b + c*c)^2

    [q[1]-q[7]*4*b*x*y^(-1)-q[7]*4*2*a*c*y^(-1);
     q[2]-q[7]*4*(2*b*c)*y^(-1)-q[7]*4*(-a*x)*y^(-1);
     q[3]-q[7]-q[7]*4*(-2*b^2)*y^(-1)-q[7]*4*(-2*a^2)*y^(-1)]
end

function ϕ_func(m::Hopper,q)
    # v = q[4:6]
    #
    # a = v[1]
    # b = v[2]
    # c = v[3]
    #
    # x = 1 - (a*a + b*b + c*c)
    # y = (1 + a*a + b*b + c*c)^2

    m1w = 2
    m2w = -2
    bw = 0.2

    pxz = kinematics(m,q)[[1;3]]
    wall1 = [q[1];m1w*q[1] - bw]
    diff1 = pxz - wall1
    wall2 = [q[1];m2w*q[1] - bw]
    diff2 = pxz - wall2


    @SVector [kinematics(m,q)[3] - (kinematics(m,q)[2] >= 0.75 ? 1.0 : 0.0),
              diff1'*diff1,
              diff2'*diff2]
end

# function ϕ_stairs(::Hopper,q)
#     v = q[4:6]
#
#     a = v[1]
#     b = v[2]
#     c = v[3]
#
#     x = 1 - (a*a + b*b + c*c)
#     y = (1 + a*a + b*b + c*c)^2
#
#     h = 0.25
#     w = 1.0
#     px = q[1]-q[7]*4*b*x*y^(-1)-q[7]*4*2*a*c*y^(-1)
#
#     smooth_step(rr) = rr - sin(rr)
#
#     sh = h/(2*pi)*smooth_step.(smooth_step.(smooth_step.(smooth_step.(2*pi/w*px))))
#     return @SVector [q[3]-q[7]-q[7]*4*(-2*b^2)*y^(-1)-q[7]*4*(-2*a^2)*y^(-1) - sh]
#
#     # if px < 0.5
#     #     return @SVector [q[3]-q[7]-q[7]*4*(-2*b^2)*y^(-1)-q[7]*4*(-2*a^2)*y^(-1)]
#     # elseif 0.5 <= px && px < 1.5
#     #     return @SVector [q[3]-q[7]-q[7]*4*(-2*b^2)*y^(-1)-q[7]*4*(-2*a^2)*y^(-1) - h]
#     # elseif 1.5 <= px
#     #     return @SVector [q[3]-q[7]-q[7]*4*(-2*b^2)*y^(-1)-q[7]*4*(-2*a^2)*y^(-1) - 2h]
#     # end
# end

function N_func(m::Hopper,q)
    # v = q[4:6]
    #
    # a = v[1]
    # b = v[2]
    # c = v[3]
    #
    # x = 1 - (a*a + b*b + c*c)
    # y = (1 + a*a + b*b + c*c)^2
    #
    # xa = -2a
    # xb = -2b
    # xc = -2c
    #
    # ya = 2*(1 + a*a + b*b + c*c)*2*a
    # yb = 2*(1 + a*a + b*b + c*c)*2*b
    # yc = 2*(1 + a*a + b*b + c*c)*2*c
    #
    # p34 = q[7]*4*(-2*b^2)*y^(-2)*ya -q[7]*4*(-4*a)*y^(-1) +q[7]*4*(-2*a^2)*y^(-2)*ya
    # p35 = -q[7]*4*(-4*b)*y^(-1) + q[7]*4*(-2*b^2)*y^(-2)*yb + q[7]*4*(-2*a^2)*y^(-2)*yb
    # p36 = q[7]*4*(-2*b^2)*y^(-2)*yc + q[7]*4*(-2*a^2)*y^(-2)*yc
    # p37 = -1.0 -4*(-2*b^2)*y^(-1)-4*(-2*a^2)*y^(-1)
    #
    # @SMatrix [0.0 0.0 1.0 p34 p35 p36 p37;
    #           ]
    tmp(z) = ϕ_func(m,z)
    ForwardDiff.jacobian(tmp,q)
end

function P_func(m::Hopper,q)
    # v = q[4:6]
    #
    # a = v[1]
    # b = v[2]
    # c = v[3]
    #
    # x = 1 - (a*a + b*b + c*c)
    # y = (1 + a*a + b*b + c*c)^2
    #
    # xa = -2a
    # xb = -2b
    # xc = -2c
    #
    # ya = 2*(1 + a*a + b*b + c*c)*2*a
    # yb = 2*(1 + a*a + b*b + c*c)*2*b
    # yc = 2*(1 + a*a + b*b + c*c)*2*c
    # p14 = -q[7]*4*b*xa*y^(-1) + q[7]*4*b*x*y^(-2)*ya -q[7]*4*2*c*y^(-1) + q[7]*4*2*a*c*y^(-2)*ya
    # p15 = -q[7]*4*x*y^(-1) -q[7]*4*b*xb*y^(-1) + q[7]*4*b*x*y^(-2)*yb + q[7]*4*2*a*c*y^(-2)*yb
    # p16 = -q[7]*4*b*xc*y^(-1) + q[7]*4*b*x*y^(-2)*yc -q[7]*4*2*a*y^(-1) + q[7]*4*2*a*c*y^(-2)*yc
    # p17 = -4*b*x*y^(-1)-4*2*a*c*y^(-1)
    #
    # p24 = q[7]*4*(2*b*c)*y^(-2)*ya -q[7]*4*(-x)*y^(-1) -q[7]*4*(-a*xa)*y^(-1) + q[7]*4*(-a*x)*y^(-2)*ya
    # p25 = -q[7]*4*(2*c)*y^(-1) + q[7]*4*(2*b*c)*y^(-2)*yb -q[7]*4*(-a*xb)*y^(-1) +q[7]*4*(-a*x)*y^(-2)*yb
    # p26 = -q[7]*4*(2*b)*y^(-1) +q[7]*4*(2*b*c)*y^(-2)*yc -q[7]*4*(-a*xc)*y^(-1) +q[7]*4*(-a*x)*y^(-2)*yc
    # p27 = -4*(2*b*c)*y^(-1)-4*(-a*x)*y^(-1)
    #
    # @SMatrix [1.0 0.0 0.0 p14 p15 p16 p17;
    #           -1.0 0.0 0.0 -p14 -p15 -p16 -p17;
    #           0.0 1.0 0.0 p24 p25 p26 p27;
    #           0.0 -1.0 0.0 -p24 -p25 -p26 -p27]
    map = [1.0 0.0;
           -1.0 0.0;
           0.0 1.0;
           0.0 -1.0]

    tmp(z) = kinematics(m,z)
    _P = ForwardDiff.jacobian(tmp,q)
    [map*_P[1:2,:];
     map*_P[2:3,:];
     map*_P[2:3,:]]
end

function friction_cone(model,u)
    λ = u[model.idx_λ]
    b = u[model.idx_b]
    @SVector [model.μ*λ[1] - sum(b[1:4]),
              model.μ*λ[2] - sum(b[5:8]),
              model.μ*λ[3] - sum(b[9:12]),]
end

function maximum_energy_dissipation(model,x2,x3,u,h)
    ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1]*ones(4);ψ[2]*ones(4);ψ[3]*ones(4)]
    η = u[model.idx_η]
    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

qL = -Inf*ones(nq)
qU = Inf*ones(nq)
qL[3] = r/100.0
qU[3] = r

α_hopper = 1.0

model = Hopper(mb,ml,Jb,Jl,r,μ,g,qL,qU,
                nx,nu,nu_ctrl,
                nc,nf,nb,
                idx_u,
                idx_λ,
                idx_b,
                idx_ψ,
                idx_η,
                idx_s,
                α_hopper)

#Visualization
function visualize!(vis,h::Hopper,q;
        verbose=false,Δt=0.1)
   r_foot = 0.05
   setobject!(vis["body"], HyperSphere(Point3f0(0),
       convert(Float32,0.1)),
       MeshPhongMaterial(color=RGBA(0, 1, 0, 1.0)))
   setobject!(vis["foot"], HyperSphere(Point3f0(0),
       convert(Float32,r_foot)),
       MeshPhongMaterial(color=RGBA(1, 0, 0, 1.0)))

   r_leg = 0.5*r_foot
   n_leg = 100
   for i = 1:n_leg
       setobject!(vis["leg$i"], HyperSphere(Point3f0(0),
           convert(Float32,r_leg)),
           MeshPhongMaterial(color=RGBA(0, 0, 0, 1.0)))
   end
   p_leg = [zeros(3) for i = 1:n_leg]
   anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

   z_shift = [0.;0.;r_foot]
   for t = 1:length(q)
       p_body = q[t][1:3]

       if verbose
           println("foot height: $(p_foot[3])")
       end

       q_tmp = Array(copy(q[t]))
       r_range = range(0.,stop=q[t][7],length=n_leg)
       for i = 1:n_leg
           q_tmp[7] = r_range[i]
           p_leg[i] = kinematics(model,q_tmp)
       end
       q_tmp[7] = q[t][7]
       p_foot = kinematics(model,q_tmp)

       MeshCat.atframe(anim,t) do
           settransform!(vis["body"], Translation(p_body+z_shift))
           settransform!(vis["foot"], Translation(p_foot+z_shift))

           for i = 1:n_leg
               settransform!(vis["leg$i"], Translation(p_leg[i]+z_shift))
           end
       end
   end
   MeshCat.setanimation!(vis,anim)
end
