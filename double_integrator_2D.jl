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

Δt = 0.1

model = DoubleIntegrator2D(1.0,1.0) # nominal model
