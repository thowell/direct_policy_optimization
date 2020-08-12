function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nothing
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0,)
	row = []
	col = []
	r = (1:0)
	c = (1:0)

	return collect(zip(row,col))
end

function general_constraints!(c,Z,prob::SampleProblem)
	nothing
end

function ∇general_constraints!(∇c,Z,prob::SampleProblem)
	nothing
end

function general_constraint_sparsity(prob::SampleProblem;
		r_shift=0)
	row = []
	col = []
	r = (1:0)
	c = (1:0)

	return collect(zip(row,col))
end
