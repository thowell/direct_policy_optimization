mutable struct Dubins
    nx::Int
    nu::Int
end

function dynamics(model::Dubins,x,u)
    @SVector [u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]
end

nx,nu = 3,2
model = Dubins(nx,nu)

function circle_obs(x,y,xc,yc,r)
    (x-xc)^2 + (y-yc)^2 - r^2
end
