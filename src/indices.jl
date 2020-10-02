mutable struct Indices
    x
    u
end

function init_indices(nx,nu,T;
    time=true,shift=0)
    x = []
    u = []

    for t = 1:T
        push!(x,shift+(t-1)*(nx+nu) .+ (1:nx))
        t==T && continue
        push!(u,shift+(t-1)*(nx+nu)+nx .+ (1:nu))
    end

    return Indices(x,u)
end
