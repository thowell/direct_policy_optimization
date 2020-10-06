mutable struct Car
    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::Car,x,u,w)
    @SVector [u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]
end

function discrete_dynamics(model::Car,x⁺,x,u,h,w,t)
    midpoint_implicit(model,x⁺,x,u,h,w) - w
end

function discrete_dynamics(model::Car,x,u,h,w,t)
    midpoint(model,x,u,h,w) + w
end

nx,nu,nw = 3,2,3
model = Car(nx,nu,nw)
