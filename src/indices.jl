mutable struct Indices
    x
    u
    h
end

function init_indices(nx,nu,T;
    shift=0)
    x = []
    u = []
    h = []

    for t = 1:T-2
        push!(x,shift+(t-1)*(nx+nu+1) .+ (1:nx))
        push!(u,shift+(t-1)*(nx+nu+1)+nx .+ (1:nu))
        push!(h,shift+(t-1)*(nx+nu+1)+nx+nu + 1)
    end
    push!(x,shift+(T-2)*(nx+nu+1) .+ (1:nx))
    push!(x,shift+(T-2)*(nx+nu+1)+nx .+ (1:nx))

    return Indices(x,u,h)
end
