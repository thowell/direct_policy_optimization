function midpoint(model,z,u,Δt)
    z + Δt*model.f(model,z + 0.5*Δt*model.f(model,z,u),u)
end

function rk3(model,z,u,Δt)
    k1 = k2 = k3 = zero(z)
    k1 = Δt*model.f(model,z,u)
    k2 = Δt*model.f(model,z + 0.5*k1,u)
    k3 = Δt*model.f(model,z - k1 + 2.0*k2,u)
    z + (k1 + 4.0*k2 + k3)/6.0
end

function xm_rk3_implicit(model,x⁺,x,u,h)
    0.5*(x⁺ + x) + h[1]/8.0*(model.f(model,x,u) - model.f(model,x⁺,u))
end

function rk3_implicit(model,x⁺,x,u,h)
    xm = xm_rk3_implicit(model,x⁺,x,u,h)
    x⁺ - x - h[1]/6*model.f(model,x,u) - 4*h[1]/6*model.f(model,xm,u) - h[1]/6*model.f(model,x⁺,u)
end
