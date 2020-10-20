# eigen value analysis
C = Diagonal(ones(model_slosh.nx))[1:model_nom.nx,:]
# nominal
A_nom, B_nom = nominal_jacobians(model_nom,X_nom,U_nom,H_nom)
A_nom_cl = [(A_nom[t] - B_nom[t]*K[t]) for t = 1:T-1]
sv_nom = [norm.(eigen(A_nom_cl[t]).values) for t = 1:T-1]
plt_nom = plot(hcat(sv_nom...)',xlabel="time step t",ylabel="eigen value norm",
	title="TVLQR nominal model",linetype=:steppost,
	ylims=(-1,3),labels="")

# slosh nominal
X_nom_slosh = [[copy(X_nom[t]);0.0;0.0] for t = 1:T]
A_nom_slosh, B_nom_slosh = nominal_jacobians(model_slosh,X_nom_slosh,U_nom,H_nom)
A_nom_slosh_cl = [(A_nom_slosh[t] - B_nom_slosh[t]*K[t]*C) for t = 1:T-1]
sv_nom_slosh = [norm.(eigen(A_nom_slosh_cl[t]).values) for t = 1:T-1]
plt_nom_slosh = plot(hcat(sv_nom_slosh...)',xlabel="time step t",ylabel="eigen value norm",
	title="TVLQR slosh model",linetype=:steppost,
	ylims=(-1,3),labels="")

# slosh
A_dpo, B_dpo = nominal_jacobians(model_nom,X_nom_sample,U_nom_sample,H_nom_sample)
A_dpo_cl = [(A_dpo[t] - B_dpo[t]*Θ_mat[t]) for t = 1:T-1]
sv_dpo = [norm.(eigen(A_dpo_cl[t]).values) for t = 1:T-1]
plt_dpo_nom = plot(hcat(sv_dpo...)',xlabel="time step t",ylabel="singular value",
	title="DPO nominal model",linetype=:steppost,
	ylims=(-1,3),labels="")

X_dpo_slosh = [[copy(X_nom_sample[t]);0.0;0.0] for t = 1:T]
A_dpo_slosh, B_dpo_slosh = nominal_jacobians(model_slosh,X_dpo_slosh,U_nom_sample,H_nom_sample)
A_dpo_slosh_cl = [(A_dpo_slosh[t] - B_dpo_slosh[t]*Θ_mat[t]*C) for t = 1:T-1]
sv_dpo_slosh = [norm.(eigen(A_dpo_slosh_cl[t]).values) for t = 1:T-1]
plt_dpo_slosh = plot(hcat(sv_dpo_slosh...)',xlabel="time step t",ylabel="eigen value norm",
	title="DPO slosh model",linetype=:steppost,
	ylims=(-1.0,3.0),labels="")

plot(plt_nom,plt_dpo_nom,layout=(2,1))

plot(plt_nom_slosh,plt_dpo_slosh,layout=(2,1))
