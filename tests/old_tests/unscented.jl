using LinearAlgebra, Distributions, Plots

function error_ellipse(μ,Σ; p=0.95,plt=plot(),color=:red,label="")
    s = -2.0*log(1.0 - p)
    e = eigen(Σ*s)
    t = range(0,stop=2π,length=100)
    a = [μ + e.vectors*Diagonal(sqrt.(e.values))*[cos(t[i]); sin(t[i])] for i = 1:100]
    plt = plot!(hcat(a...)[1,:],hcat(a...)[2,:],width=2.0,color=color,label=label)
end

μ = zeros(2)
Σ = Diagonal(rand(2))
X = Distributions.MvNormal(μ,Σ)
N = 100
x = rand(X,N)

plt = plot(xlims=(-3,6),ylims=(-4,4),aspect_ratio=:equal)
for i = 1:N
    plt = scatter!([x[1,i]],[x[2,i]],color=:black,label="")
end

plt = error_ellipse(μ,Σ,label="dist.",plt=plt)

# pendulum dynamics
n = 2
m = 1
function dyn_c(x,u)
    m = 1.0
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    return [x[2];(u[1] - m*g*lc*sin(x[1]) - b*x[2])/I]
end

# Pendulum discrete-time dynamics (midpoint)
Δt = 0.1
function dyn_d(x,u,Δt)
    x + Δt*dyn_c(x + 0.5*Δt*dyn_c(x,u),u)
end

u0 = rand(m)

# sample through dynamics
x⁺ = zero(x)
for i = 1:N
    x⁺[:,i] = dyn_d(x[:,i],u0,Δt)
end

for i = 1:N
    plt = scatter!([x⁺[1,i]+ 3.0] ,[x⁺[2,i]],color=:cyan,label="")
end
μ⁺_est = sum(x⁺,dims=2)./N
Σ⁺_est = sum([(x⁺[:,i] - μ⁺_est)*(x⁺[:,i] - μ⁺_est)' for i = 1:N])./N
plt = error_ellipse(μ⁺_est + [3.0;0.0],Σ⁺_est,label="dyn. (N=$N)",color=:green,plt=plt)

display(plt)

β = 1.5
x_sigma = hcat([μ + β*s*Array(cholesky(Σ).U)[:,i] for s in [-1.0,1.0] for i = 1:2]...)
N_sigma = 4

for i = 1:N_sigma
    plt = scatter!([x_sigma[1,i]],[x_sigma[2,i]],color=:orange,label="")
end
display(plt)

x⁺_sigma = zero(x_sigma)
for i = 1:N_sigma
    x⁺_sigma[:,i] = dyn_d(x_sigma[:,i],u0,Δt)
end

for i = 1:N_sigma
    plt = scatter!([x⁺_sigma[1,i] + 3.0],[x⁺_sigma[2,i]],color=:purple,label="")
end
display(plt)

μ⁺_sigma_est = sum(x⁺_sigma,dims=2)./N_sigma
Σ⁺_sigma_est = sum([(x⁺_sigma[:,i] - μ⁺_sigma_est)*(x⁺_sigma[:,i] - μ⁺_sigma_est)' for i = 1:N_sigma])./N_sigma# + 0.1*I
plt = error_ellipse(μ⁺_sigma_est + [3.0;0.0],Σ⁺_sigma_est,label="dyn. (unscented)",color=:brown,plt=plt)
