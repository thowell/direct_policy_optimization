mutable struct Cartpole{T}
    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
    nx::Int   # state dimension
    nu::Int   # control dimension
end

function dynamics(model::Cartpole, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]
    qdd = SVector{2}(-H\(C*view(x,3:4) + G - B*u[1]))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

nx,nu = 4,1
model = Cartpole(1.0,0.2,0.5,9.81,nx,nu)
model_nominal = model

mutable struct CartpoleFriction{T}
    mc::T     # mass of the cart in kg
    mp::T     # mass of the pole (point mass at the end) in kg
    l::T      # length of the pole in m
    g::T      # gravity m/s^2
    μ::T      # friction coefficient
    nx::Int   # state dimension
    nu::Int   # control dimension
	nu_policy
end

function dynamics(model::CartpoleFriction, x, u)
    H = @SMatrix [model.mc+model.mp model.mp*model.l*cos(x[2]); model.mp*model.l*cos(x[2]) model.mp*model.l^2]
    C = @SMatrix [0.0 -model.mp*x[4]*model.l*sin(x[2]); 0.0 0.0]
    G = @SVector [0.0, model.mp*model.g*model.l*sin(x[2])]
    B = @SVector [1.0, 0.0]

    qdd = SVector{2}(-H\(C*view(x,3:4) + G - B*(u[1] + (u[2] - u[3]))))

    return @SVector [x[3],x[4],qdd[1],qdd[2]]
end

nx_friction,nu_friction = 4,7
nu_policy_friction = nu
model_friction = CartpoleFriction(1.0,0.2,0.5,9.81,0.1,
	nx_friction,nu_friction,nu_policy_friction)

function maximum_energy_dissipation(x⁺,u,model)

    v = x⁺[3]
    β = u[2:3]
    ψ = u[4]
    η = u[5:6]
    s = u[7]

    n = (model.mc + model.mp)*model.g

    c1 = v + ψ - η[1]
    c2 = -v + ψ - η[2]
    c3 = model.μ*n - sum(β)
    c4 = s - ψ*(model.μ*n - sum(β))
    c5 = s - β'*η

    return @SVector [c1, c2, c3, c4, c5]
end

m_med = 5

function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	idx = prob.idx
	model = prob.model
	T = prob.T
	for t = 1:T-1
		x⁺ = view(Z,idx.x[t+1])
		u = view(Z,idx.u[t])
		c[(t-1)*m_med .+ (1:m_med)] = maximum_energy_dissipation(x⁺,u,model)
	end
	nothing
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	shift = 0
	idx = prob.idx
	model = prob.model
	T = prob.T
	for t = 1:T-1
		x⁺ = view(Z,idx.x[t+1])
		u = view(Z,idx.u[t])

		med_x⁺(y) = maximum_energy_dissipation(y,u,model)
		med_u(y) = maximum_energy_dissipation(x⁺,y,model)

		r_idx = (t-1)*m_med .+ (1:m_med)

		c_idx = idx.x[t+1]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(med_x⁺,x⁺))
		shift += len

		c_idx = idx.u[t]
		len = length(r_idx)*length(c_idx)
		∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(med_u,u))
		shift += len
	end
	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0,)
	row = []
	col = []

	idx = prob.idx
	T = prob.T

	for t = 1:T-1

		r_idx = r_shift + (t-1)*m_med .+ (1:m_med)

		c_idx = idx.x[t+1]
		row_col!(row,col,r_idx,c_idx)

		c_idx = idx.u[t]
		row_col!(row,col,r_idx,c_idx)
	end

	return collect(zip(row,col))
end

function general_constraints!(c,Z,prob::SampleProblem)
	idx_sample = prob.idx_sample
	model = prob.prob.model
	T = prob.prob.T

	shift = 0
	for t = 1:T-1
		for i = 1:N
			x⁺ = view(Z,idx_sample[i].x[t+1])
			u = view(Z,idx_sample[i].u[t])
			c[shift .+ (1:m_med)] = maximum_energy_dissipation(x⁺,u,models[i])
			shift += m_med
		end
	end
	nothing
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	shift = 0
	s = 0

	idx_sample = prob.idx_sample
	model = prob.prob.model
	T = prob.prob.T

	for t = 1:T-1
		for i = 1:N
			x⁺ = view(Z,idx_sample[i].x[t+1])
			u = view(Z,idx_sample[i].u[t])

			med_x⁺(y) = maximum_energy_dissipation(y,u,models[i])
			med_u(y) = maximum_energy_dissipation(x⁺,y,models[i])

			r_idx = s .+ (1:m_med)

			c_idx = idx_sample[i].x[t+1]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(med_x⁺,x⁺))
			shift += len

			c_idx = idx_sample[i].u[t]
			len = length(r_idx)*length(c_idx)
			∇c[shift .+ (1:len)] = vec(ForwardDiff.jacobian(med_u,u))
			shift += len

			s += m_med
		end
	end
	nothing
end

function general_constraint_sparsity(prob::SampleProblem;
		r_shift=0,)
	row = []
	col = []

	shift = 0
	s = 0

	idx_sample = prob.idx_sample
	model = prob.prob.model
	T = prob.prob.T

	for t = 1:T-1
		for i = 1:N
			# x⁺ = view(Z,idx_sample[i].x[t+1])
			# u = view(Z,idx_sample[i].u[t])
			#
			# med_x⁺(y) = maximum_energy_dissipation(y,u,model)
			# med_u(y) = maximum_energy_dissipation(x⁺,y,model)

			r_idx = r_shift + s .+ (1:m_med)

			c_idx = idx_sample[i].x[t+1]
			row_col!(row,col,r_idx,c_idx)

			c_idx = idx_sample[i].u[t]
			row_col!(row,col,r_idx,c_idx)

			s += m_med
		end
	end

	return collect(zip(row,col))
end

m_stage_friction = 5

ul_friction = zeros(nu_friction)
ul_friction[1] = -10.0
uu_friction = Inf*ones(nu_friction)
uu_friction[1] = 10.0

stage_friction_ineq = (3:5)
(model.mc + model.mp)*model.g

const α_cartpole_friction = 1000.0 #5.0

mutable struct PenaltyObjective{T} <: Objective
    α::T
end

function objective(Z,l::PenaltyObjective,model::CartpoleFriction,idx,T)
    J = 0
    for t = 1:T-1
        s = Z[idx.u[t][7]]
        J += s
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model::CartpoleFriction,idx,T)
    for t = 1:T-1
        u = Z[idx.u[t][7]]
        ∇l[idx.u[t][7]] += l.α
    end
    return nothing
end

function sample_general_objective(z,prob::SampleProblem)
    idx_sample = prob.idx_sample
    T = prob.prob.T
    N = prob.N

    J = 0.0

    for t = 1:T-1
        for i = 1:N
            s = z[idx_sample[i].u[t][7]]
            J += s
        end
    end

    return α_cartpole_friction*J
end

function ∇sample_general_objective!(∇obj,z,prob::SampleProblem)
    idx_sample = prob.idx_sample
    T = prob.prob.T
    N = prob.N

    for t = 1:T-1
        for i = 1:N
            ∇obj[idx_sample[i].u[t][7]] += α_cartpole_friction
        end
    end
    return nothing
end

function policy(model::CartpoleFriction,K,x,u,x_nom,u_nom)
	u_nom[1:model.nu_policy] - reshape(K,model.nu_policy,model.nx)*(x - x_nom)
end

function visualize!(vis,model::Cartpole,q;
       Δt=0.1,color=[RGBA(1,0,0,1.0) for i = 1:length(q)])

   l2 = Cylinder(Point3f0(-model.l*2,0,0),Point3f0(model.l*2,0,0),convert(Float32,0.025))
   setobject!(vis["slider"],l2,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))

	N = length(q)

	for i = 1:N
	    l1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,model.l),convert(Float32,0.025))
	    setobject!(vis["arm$i"],l1,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))

	    setobject!(vis["base$i"], HyperSphere(Point3f0(0),
	        convert(Float32,0.1)),
	        MeshPhongMaterial(color=color[i]))
	    setobject!(vis["ee$i"], HyperSphere(Point3f0(0),
	        convert(Float32,0.05)),
	        MeshPhongMaterial(color=color[i]))
	end

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q[1])


        MeshCat.atframe(anim,t) do
			for i = 1:N
				x = q[i][t]
				px = x[1] + model.l*sin(x[2])
				pz = -model.l*cos(x[2])
	            settransform!(vis["arm$i"], cable_transform([x[1];0;0],[px;0.0;pz]))
	            settransform!(vis["base$i"], Translation([x[1];0.0;0.0]))
	            settransform!(vis["ee$i"], Translation([px;0.0;pz]))
			end
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
