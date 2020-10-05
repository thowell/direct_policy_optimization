# linear full-state-feedback policy
function policy(model,K,x,x_nom,u_nom)
	u_nom - reshape(K,model.nu,model.nx)*(x - x_nom)
end

# function no_policy(model,K,x,u,x_nom,u_nom)
# 	# u_nom - reshape(K,model.nu,model.nx)*(x - x_nom)
# 	u
# end

function output(model,x)
	x
end
