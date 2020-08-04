function obj_sample(z,idx_nom,idx_sample,Q,R,H,T,N,γ)
    J = 0.0
    #TODO consider no initial condition -> fix this cost
    # sample
    for t = 1:T-1
        u_nom = view(z,idx_nom.u[t])
        h_nom = z[idx_nom.h[t]]
        x⁺_nom = view(z,idx_nom.x[t+1])

        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]
            xi⁺ = view(z,idx_sample[i].x[t+1])
            J += (xi⁺ - x⁺_nom)'*Q[t+1]*(xi⁺ - x⁺_nom) + (ui - u_nom)'*R[t]*(ui - u_nom) + (hi - h_nom)'*H[t]*(hi - h_nom)
        end
    end

    return γ*J/N
end

function ∇obj_sample!(∇obj,z,idx_nom,idx_sample,Q,R,H,T,N,γ)
    for t = 1:T-1
        u_nom = view(z,idx_nom.u[t])
        h_nom = z[idx_nom.h[t]]
        x⁺_nom = view(z,idx_nom.x[t+1])
        for i = 1:N
            ui = view(z,idx_sample[i].u[t])
            hi = z[idx_sample[i].h[t]]
            xi⁺ = view(z,idx_sample[i].x[t+1])

            ∇obj[idx_sample[i].x[t+1]] += 2.0*Q[t+1]*(xi⁺ - x⁺_nom)*γ/N
            ∇obj[idx_sample[i].u[t]] += 2.0*R[t]*(ui - u_nom)*γ/N
            ∇obj[idx_sample[i].h[t]] += 2.0*H[t]*(hi - h_nom)*γ/N

            ∇obj[idx_nom.x[t+1]] -= 2.0*Q[t+1]*(xi⁺ - x⁺_nom)*γ/N
            ∇obj[idx_nom.u[t]] -= 2.0*R[t]*(ui - u_nom)*γ/N
            ∇obj[idx_nom.h[t]] -= 2.0*H[t]*(hi - h_nom)*γ/N
        end
    end
    return nothing
end
