using LinearAlgebra, Distributions, Plots
include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/pendulum.jl"))

function error_ellipse(μ,Σ; p=0.95)
    s = -2.0*log(1.0 - p)
    e = eigen(Σ*s)
    t = range(0,stop=2π,length=100)
    a = [μ + e.vectors*Diagonal(sqrt.(e.values))*[cos(t[i]); sin(t[i])] for i = 1:100]
end

function plot_error_ellipse(μ,Σ; p=0.95,plt=plot(),color=:red,label="")
    a = error_ellipse(μ,Σ,p=p)
    plt = plot!(hcat(a...)[1,:],hcat(a...)[2,:],width=2.0,color=color,label=label,linestyle=:dash)
end

function resample_axis(X; β=1.0,w=ones(length(X[1])))
    N = length(X)
    nx = length(X[1])
    xμ = sample_mean(X)
    Σμ = sample_covariance(X,β=β,w=w)
    cols = matrix_sqrt_axis(Σμ)
    Xs = [xμ + s*β*cols[:,i] for s in [-1.0,1.0] for i = 1:nx]
    return Xs
end

# Pendulum model
nx = model.nx
nu = model.nu
Δt = 0.1

# samples
x0 = [0.0;0.0]
N = 4
α = 1.0
x11 = x0 + α*[1.0; 1.0]
x12 = x0 + α*[-1.0; 1.0]
x13 = x0 + α*[1.0; -1.0]
x14 = x0 + α*[-1.0; -1.0]

shift = 4.0 # x shift for visualization

# resampling parameters
β = 1.0
w = 0.0*ones(nx)

# initial distribution
x1 = [x11,x12,x13,x14]

u1 = 1.0*rand(nu)
μ1 = sample_mean(x1)
Σ1 = sample_covariance(x1,β=1.0,w=zeros(nx))

plt = plot([μ1[1]],[μ1[2]],marker=:square,color=:black,aspect_ratio=:equal,
    grid=false,markersize=10.0,label="")
plot_error_ellipse(μ1,Σ1,plt=plt,p=0.395,color=:black)
for i = 1:N
    plt = scatter!([x1[i][1]] ,[x1[i][2]],color=:black,label="",markersize=10.0)
end
display(plt)

# propagate distribution
x̂⁺ = [discrete_dynamics(model,x1[i],u1,Δt,0) for i = 1:N]

μ⁺ = sample_mean(x̂⁺)
Σ⁺ = sample_covariance(x̂⁺,β=β,w=w)
x⁺ = resample_axis(x̂⁺,β=β,w=w)

cholesky(Σ⁺)
sqrt(Σ⁺)
matrix_sqrt(Σ⁺)

plt = plot!([μ⁺[1]+shift],[μ⁺[2]],marker=:square,color=:cyan,markersize=10.0,
    label="")
plot_error_ellipse(μ⁺+[shift;0.0],Σ⁺,plt=plt,p=0.395,color=:cyan)
for i = 1:N
    plt = scatter!([x̂⁺[i][1]+shift] ,[x̂⁺[i][2]],color=:red,label="",
        markersize=10.0)
    plt = scatter!([x⁺[i][1]+shift] ,[x⁺[i][2]],color=:cyan,label="",
        markersize=10.0)
end
display(plt)

# PGFplots version
using PGFPlots
const PGF = PGFPlots

# nominal trajectory
px1= PGF.Plots.Scatter([x1[i][1] for i = 1:N],
					   [x1[i][2] for i = 1:N],
					   mark="*",
					   style="color=purple, line width=3pt",
					   )
pμ1= PGF.Plots.Scatter([μ1[1]],
					   [μ1[2]],
					   mark="square*",
					   style="color=purple, line width=3pt",
					   )
ee1 = hcat(error_ellipse(μ1,Σ1,p=0.395)...)
pe1= PGF.Plots.Linear(ee1[1,:],
					   ee1[2,:],
					   mark="",
					   style="color=purple, densely dashed, line width=2pt",
					   )

px̂⁺= PGF.Plots.Scatter([x̂⁺[i][1]+shift for i = 1:N],
					   [x̂⁺[i][2] for i = 1:N],
					   mark="triangle*",
					   style="color=cyan, line width=3pt")
px⁺= PGF.Plots.Scatter([x⁺[i][1]+shift for i = 1:N],
					   [x⁺[i][2] for i = 1:N],
					   mark="*",
					   style="color=orange, line width=3pt",)
pμ⁺= PGF.Plots.Scatter([μ⁺[1]+shift],
					   [μ⁺[2]],
					   mark="square*",
					   style="color=orange, line width=3pt",)
ee⁺ = hcat(error_ellipse(μ⁺+[shift;0.0],Σ⁺,p=0.395)...)
pe⁺= PGF.Plots.Linear(ee⁺[1,:],
					   ee⁺[2,:],
					   mark="",
					   style="color=orange, densely dashed, line width=2pt",
					   )

a = Axis([pe1;px1;pμ1;pe⁺;px̂⁺;px⁺;pμ⁺],
    axisEqualImage=true,
    hideAxis=false,
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"unscented.tikz"), a, include_preamble=false)
