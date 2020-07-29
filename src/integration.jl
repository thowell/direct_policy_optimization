function xm_rk3_implicit(model,x⁺,x,u,h)
    0.5*(x⁺ + x) + h[1]/8.0*(dynamics(model,x,u) - dynamics(model,x⁺,u))
end

function rk3_implicit(model,x⁺,x,u,h)
    xm = xm_rk3_implicit(model,x⁺,x,u,h)
    x⁺ - x - h[1]/6*dynamics(model,x,u) - 4*h[1]/6*dynamics(model,xm,u) - h[1]/6*dynamics(model,x⁺,u)
end

function discrete_linear(model,x⁺,x,u,h)
    D = exp(h*[model.Ac model.Bc; zeros(1,model.nx+model.nu)])
    A = D[1:model.nx,1:model.nx]
    B = D[1:model.nx,model.nx .+ (1:model.nu)]
    x⁺ - (A*x + B*u)
end

function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end