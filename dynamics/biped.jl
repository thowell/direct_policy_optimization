include("biped/f.jl")
include("biped/g.jl")
include("biped/utils.jl")

mutable struct Biped
    l1
    l2
    nx::Int
    nu::Int
end

function dynamics(model::Biped,x,u)
    f(x) + g(x)*u
end


nx = 10
nu = 4
model = Biped(0.2755,0.288,nx,nu)
