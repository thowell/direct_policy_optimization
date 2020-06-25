# parameterized model
mutable struct DoubleIntegrator2D
    # z = (x,y,ẋ,ẏ)
    mx
    my
end

# double integrator dynamics (2D)
n = 4 # number of states
m = 2 # number of controls

function dynamics(model::DoubleIntegrator2D,z,u)
    @SVector [z[3],z[4],u[1]/model.mx,u[2]/model.my]
end

# discrete dynamics (midpoint)
Δt = 0.1
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end

model = DoubleIntegrator2D(1.0,1.0) # nominal model
