mutable struct car
    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::car,x,u,w)
    @SVector [u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]
end

nx,nu,nw = 3,2,0
model = car(nx,nu,nw)
