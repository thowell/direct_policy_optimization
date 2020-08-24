include("../src/sample_motion_planning.jl")
using Plots

x1 = zeros(2)
plt = plot([x1[1]],[x1[2]],marker=:circle,label="x1",color=:black)

β=1.0e-2
w=0.0*ones(2)
x1_sample = resample([x1 for i = 1:4],β=β,w=w)
@show maximum(hcat(x1_sample...))

for i = 1:4
	plt = plot!([x1_sample[i][1]],[x1_sample[i][2]],marker=:circle,
		label="",color=:cyan)
end
display(plt)

β=1.0e-1
w=0.0*ones(2)
x1_sample = resample([x1 for i = 1:4],β=β,w=w)
@show maximum(hcat(x1_sample...))

for i = 1:4
	plt = plot!([x1_sample[i][1]],[x1_sample[i][2]],marker=:circle,
		label="",color=:orange)
end
display(plt)

β=1.0
w=0.0*ones(2)
x1_sample = resample([x1 for i = 1:4],β=β,w=w)
@show maximum(hcat(x1_sample...))

for i = 1:4
	plt = plot!([x1_sample[i][1]],[x1_sample[i][2]],marker=:circle,
		label="",color=:red)
end
display(plt)

β=10.0
w=0.0*ones(2)
x1_sample = resample([x1 for i = 1:4],β=β,w=w)
@show maximum(hcat(x1_sample...))

for i = 1:4
	plt = plot!([x1_sample[i][1]],[x1_sample[i][2]],marker=:circle,
		label="",color=:blue)
end
display(plt)

β=100.0
w=0.0*ones(2)
x1_sample = resample([x1 for i = 1:4],β=β,w=w)
@show maximum(hcat(x1_sample...))

for i = 1:4
	plt = plot!([x1_sample[i][1]],[x1_sample[i][2]],marker=:circle,
		label="",color=:green)
end
display(plt)
