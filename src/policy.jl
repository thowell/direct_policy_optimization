# linear full-state-feedback policy
function policy(model,K,x1,x2,x3,ū,h,x1_nom,x2_nom,x3_nom,u_nom,ū_nom,h_nom)
	u_nom - reshape(K,model.nu_ctrl,model.nx)*(x3 - x3_nom)
end

#TODO add velocity
