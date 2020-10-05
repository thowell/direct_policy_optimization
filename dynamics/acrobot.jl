mutable struct Acrobot{T}
    m1::T    # mass
    J1::T    # inertia
    l1::T    # length
    lc1::T   # length to COM

    m2::T
    J2::T
    l2::T
    lc2::T

    g::T    # gravity

    b1::T   # joint friction
    b2::T

    nx::Int # state dimension
    nu::Int # control dimension
    nw::Int
end

function M(model::Acrobot,q)

    a = (model.J1 + model.J2 + model.m2*model.l1*model.l1 + 2.0*model.m2*model.l1*model.lc2*cos(q[2]))
    b = model.J2 + model.m2*model.l1*model.lc2*cos(q[2])
    c = model.J2

    @SMatrix [a b;
              b c]
end

function τ(model::Acrobot,q)
    a = -model.m1*model.g*model.lc1*sin(q[1]) - model.m2*model.g*(model.l1*sin(q[1]) + model.lc2*sin(q[1]+q[2]))
    b = -model.m2*model.g*model.lc2*sin(q[1] + q[2])
    @SVector [a, b]
end

function C(model::Acrobot,x)
    a = -2.0*model.m2*model.l1*model.lc2*sin(x[2])*x[4]
    b = -model.m2*model.l1*model.lc2*sin(x[2])*x[4]
    c = model.m2*model.l1*model.lc2*sin(x[2])*x[3]
    d = 0.0
    @SMatrix [a b; c d]
end

function B(model::Acrobot,q)
    @SMatrix [0.0; 1.0]
end

function dynamics(model::Acrobot,x,u,w)
    q = view(x,1:2)
    v = view(x,3:4)
    qdd = M(model,q)\(-C(model,x)*v + τ(model,q) + B(model,q)*u - [model.b1;model.b2].*v)
    @SVector [x[3],x[4],qdd[1],qdd[2]]
end

function kinematics_mid(model::Acrobot,q)
    @SVector [model.l1*sin(q[1]),
              -model.l1*cos(q[1])]
end
function kinematics_ee(model::Acrobot,q)
    @SVector [model.l1*sin(q[1]) + model.l2*sin(q[1]+q[2]),
              -model.l1*cos(q[1]) - model.l2*cos(q[1]+q[2])]
end

nx = 4
nu = 1
nw = 0
model = Acrobot(1.0,0.33,1.0,0.5,1.0,0.33,1.0,0.5,9.81,0.1,0.1,nx,nu,nw)

function visualize!(vis,model::Acrobot,q;
        color=[RGBA(0,0,0,1.0) for i = 1:length(q)],r=0.1,Δt=0.1)

    N = length(q)

    for i = 1:N
        l1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l1),convert(Float32,0.025))
        setobject!(vis["l1$i"],l1,MeshPhongMaterial(color=color[i]))
        l2 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l2),convert(Float32,0.025))
        setobject!(vis["l2$i"],l2,MeshPhongMaterial(color=color[i]))

        setobject!(vis["elbow$i"], HyperSphere(Point3f0(0),
            convert(Float32,0.05)),
            MeshPhongMaterial(color=color[i]))
        setobject!(vis["ee$i"], HyperSphere(Point3f0(0),
            convert(Float32,0.05)),
            MeshPhongMaterial(color=color[i]))
    end

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q[1])

        MeshCat.atframe(anim,t) do
            for i = 1:N
                p_mid = [kinematics_mid(model,q[i][t])[1], 0.0, kinematics_mid(model,q[i][t])[2]]
                p_ee = [kinematics_ee(model,q[i][t])[1], 0.0, kinematics_ee(model,q[i][t])[2]]

                settransform!(vis["l1$i"], cable_transform(zeros(3),p_mid))
                settransform!(vis["l2$i"], cable_transform(p_mid,p_ee))

                settransform!(vis["elbow$i"], Translation(p_mid))
                settransform!(vis["ee$i"], Translation(p_ee))
            end
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
