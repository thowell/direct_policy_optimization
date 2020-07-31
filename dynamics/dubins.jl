mutable struct Dubins
    nx::Int
    nu::Int
end

function dynamics(model::Dubins,x,u)
    @SVector [u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]
end

nx,nu = 3,2
model = Dubins(nx,nu)
