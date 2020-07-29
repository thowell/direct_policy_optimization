mutable struct Indices
    x
    u
    h
end

function init_indices(nx,nu,T;
    time=true,shift=0)
    x = []
    u = []
    h = []

    if time
        for t = 1:T
            push!(x,shift+(t-1)*(nx+nu+1) .+ (1:nx))
            t==T && continue
            push!(u,shift+(t-1)*(nx+nu+1)+nx .+ (1:nu))
            push!(h,shift+(t-1)*(nx+nu+1)+nx+nu + 1)
        end
    else
        for t = 1:T
            push!(x,shift+(t-1)*(nx+nu) .+ (1:nx))
            t==T && continue
            push!(u,shift+(t-1)*(nx+nu)+nx .+ (1:nu))
        end
    end

    return Indices(x,u,h)
end
