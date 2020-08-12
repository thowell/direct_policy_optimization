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
        for t = 1:T-2
            push!(x,shift+(t-1)*(nx+nu+1) .+ (1:nx))
            push!(u,shift+(t-1)*(nx+nu+1)+nx .+ (1:nu))
            push!(h,shift+(t-1)*(nx+nu+1)+nx+nu + 1)
        end
        push!(x,shift+(T-2)*(nx+nu+1) .+ (1:nx))
        push!(x,shift+(T-2)*(nx+nu+1)+nx .+ (1:nx))
    else
        for t = 1:T-2
            push!(x,shift+(t-1)*(nx+nu) .+ (1:nx))
            push!(u,shift+(t-1)*(nx+nu)+nx .+ (1:nu))
        end
        push!(x,shift+(T-2)*(nx+nu) .+ (1:nx))
        push!(x,shift+(T-2)*(nx+nu)+nx .+ (1:nx))
    end

    return Indices(x,u,h)
end
