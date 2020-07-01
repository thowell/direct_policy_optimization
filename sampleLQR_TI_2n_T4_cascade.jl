using LinearAlgebra, ForwardDiff
include("ipopt.jl")

# continuous-time dynamics
n = 2
m = 1
Ac = [0.0 1.0; 0.0 0.0]
Bc = [0.0; 1.0]

# discrete-time dynamics
Δt = 0.1
D = exp(Δt*[Ac Bc; zeros(1,n+m)])
A = D[1:n,1:n]
B = D[1:n,n .+ (1:m)]

# TVLQR solution
T = 4
Q = Matrix(1.0*I,n,n)
R = Matrix(0.1*I,m,m)

P = [zeros(n,n) for t = 1:T]
K = [zeros(m,n) for t = 1:T-1]
P[T] = Q
for t = T-1:-1:1
    K[t] = (R + B'*P[t+1]*B)\(B'*P[t+1]*A)
    P[t] = Q + K[t]'*R*K[t] + (A-B*K[t])'*P[t+1]*(A-B*K[t])
end

# number of samples
N = 4

# initial state
x11 = [1.0; 0.0]
x12 = [-1.0; 0.0]
x13 = [0.0; 1.0]
x14 = [0.0; -1.0]

n_nlp =  m*n*(T-1) + N*(n*(T-1) + m*(T-1)) + N*(n*(T-1) + m*(T-2)) + + N*(n*(T-2) + m*(T-3))
m_nlp = N*(n*(T-1)) + N*(m*(T-1)) + N*(n*(T-1)) + N*(m*(T-2)) + N*(n*(T-2)) + N*(m*(T-3))

function obj(z)
    k1 = z[1:2]
    k2 = z[3:4]
    k3 = z[5:6]

    # a
    ua11 = z[7:7]
    ua12 = z[8:8]
    ua13 = z[9:9]
    ua14 = z[10:10]

    xa21 = z[11:12]
    xa22 = z[13:14]
    xa23 = z[15:16]
    xa24 = z[17:18]

    ua21 = z[19:19]
    ua22 = z[20:20]
    ua23 = z[21:21]
    ua24 = z[22:22]

    xa31 = z[23:24]
    xa32 = z[25:26]
    xa33 = z[27:28]
    xa34 = z[29:30]

    ua31 = z[31:31]
    ua32 = z[32:32]
    ua33 = z[33:33]
    ua34 = z[34:34]

    xa41 = z[35:36]
    xa42 = z[37:38]
    xa43 = z[39:40]
    xa44 = z[41:42]

    # b
    xb21 = z[43:44]
    xb22 = z[45:46]
    xb23 = z[47:48]
    xb24 = z[49:50]

    ub21 = z[51:51]
    ub22 = z[52:52]
    ub23 = z[53:53]
    ub24 = z[54:54]

    xb31 = z[55:56]
    xb32 = z[57:58]
    xb33 = z[59:60]
    xb34 = z[61:62]

    ub31 = z[63:63]
    ub32 = z[64:64]
    ub33 = z[65:65]
    ub34 = z[66:66]

    xb41 = z[67:68]
    xb42 = z[69:70]
    xb43 = z[71:72]
    xb44 = z[73:74]

    # c
    xc31 = z[75:76]
    xc32 = z[77:78]
    xc33 = z[79:80]
    xc34 = z[81:82]

    uc31 = z[83:83]
    uc32 = z[84:84]
    uc33 = z[85:85]
    uc34 = z[86:86]

    xc41 = z[87:88]
    xc42 = z[89:90]
    xc43 = z[91:92]
    xc44 = z[93:94]

    return (ua11'*R*ua11 + ua12'*R*ua12 + ua13'*R*ua13 + ua14'*R*ua14
            + ua21'*R*ua21 + ua22'*R*ua22 + ua23'*R*ua23 + ua24'*R*ua24
            + ua31'*R*ua31 + ua32'*R*ua32 + ua33'*R*ua33 + ua34'*R*ua34
            + xa21'*Q*xa21 + xa22'*Q*xa22 + xa23'*Q*xa23 + xa24'*Q*xa24
            + xa31'*Q*xa31 + xa32'*Q*xa32 + xa33'*Q*xa33 + xa34'*Q*xa34
            + xa41'*Q*xa41 + xa42'*Q*xa42 + xa43'*Q*xa43 + xa44'*Q*xa44
            + ub21'*R*ub21 + ub22'*R*ub22 + ub23'*R*ub23 + ub24'*R*ub24
            + ub31'*R*ub31 + ub32'*R*ub32 + ub33'*R*ub33 + ub34'*R*ub34
            + xb21'*Q*xb21 + xb22'*Q*xb22 + xb23'*Q*xb23 + xb24'*Q*xb24
            + xb31'*Q*xb31 + xb32'*Q*xb32 + xb33'*Q*xb33 + xb34'*Q*xb34
            + xb41'*Q*xb41 + xb42'*Q*xb42 + xb43'*Q*xb43 + xb44'*Q*xb44
            + uc31'*R*uc31 + uc32'*R*uc32 + uc33'*R*uc33 + uc34'*R*uc34
            + xc31'*Q*xc31 + xc32'*Q*xc32 + xc33'*Q*xc33 + xc34'*Q*xc34
            + xc41'*Q*xc41 + xc42'*Q*xc42 + xc43'*Q*xc43 + xc44'*Q*xc44)
end

obj(z0)

function con!(c,z)
    k1 = z[1:2]
    k2 = z[3:4]
    k3 = z[5:6]

    # a
    ua11 = z[7]
    ua12 = z[8]
    ua13 = z[9]
    ua14 = z[10]

    xa21 = z[11:12]
    xa22 = z[13:14]
    xa23 = z[15:16]
    xa24 = z[17:18]

    ua21 = z[19]
    ua22 = z[20]
    ua23 = z[21]
    ua24 = z[22]

    xa31 = z[23:24]
    xa32 = z[25:26]
    xa33 = z[27:28]
    xa34 = z[29:30]

    ua31 = z[31]
    ua32 = z[32]
    ua33 = z[33]
    ua34 = z[34]

    xa41 = z[35:36]
    xa42 = z[37:38]
    xa43 = z[39:40]
    xa44 = z[41:42]

    # b
    xb21 = z[43:44]
    xb22 = z[45:46]
    xb23 = z[47:48]
    xb24 = z[49:50]

    ub21 = z[51]
    ub22 = z[52]
    ub23 = z[53]
    ub24 = z[54]

    xb31 = z[55:56]
    xb32 = z[57:58]
    xb33 = z[59:60]
    xb34 = z[61:62]

    ub31 = z[63]
    ub32 = z[64]
    ub33 = z[65]
    ub34 = z[66]

    xb41 = z[67:68]
    xb42 = z[69:70]
    xb43 = z[71:72]
    xb44 = z[73:74]

    # c
    xc31 = z[75:76]
    xc32 = z[77:78]
    xc33 = z[79:80]
    xc34 = z[81:82]

    uc31 = z[83]
    uc32 = z[84]
    uc33 = z[85]
    uc34 = z[86]

    xc41 = z[87:88]
    xc42 = z[89:90]
    xc43 = z[91:92]
    xc44 = z[93:94]

    # a
    c[1:2] = A*x11 + B*ua11 - xa21
    c[3:4] = A*x12 + B*ua12 - xa22
    c[5:6] = A*x13 + B*ua13 - xa23
    c[7:8] = A*x14 + B*ua14 - xa24

    c[9:10] = A*xa21 + B*ua21 - xa31
    c[11:12] = A*xa22 + B*ua22 - xa32
    c[13:14] = A*xa23 + B*ua23 - xa33
    c[15:16] = A*xa24 + B*ua24 - xa34

    c[17:18] = A*xa31 + B*ua31 - xa41
    c[19:20] = A*xa32 + B*ua32 - xa42
    c[21:22] = A*xa33 + B*ua33 - xa43
    c[23:24] = A*xa34 + B*ua34 - xa44

    c[25] = ua11 + k1'*xa11
    c[26] = ua12 + k1'*xa12
    c[27] = ua13 + k1'*xa13
    c[28] = ua14 + k1'*xa14

    c[29] = ua21 + k2'*xa21
    c[30] = ua22 + k2'*xa22
    c[31] = ua23 + k2'*xa23
    c[32] = ua24 + k2'*xa24

    c[33] = ua31 + k3'*xa31
    c[34] = ua32 + k3'*xa32
    c[35] = ua33 + k3'*xa33
    c[36] = ua34 + k3'*xa34

    # b
    Σb = xa21*xa21' + xa22*xa22' + xa23*xa23' + xa24*xa24' + 1.0e-8*I
    cols_b = Array(cholesky(Σb).U)
    β = 0.001

    c[37:38] = β*cols_b[:,1] - xb21
    c[39:40] = -1.0*β*cols_b[:,1] - xb22
    c[41:42] = β*cols_b[:,2] - xb23
    c[43:44] = -1.0*β*cols_b[:,2] - xb24

    c[45:46] = A*xb21 + B*ub21 - xb31
    c[47:48] = A*xb22 + B*ub22 - xb32
    c[49:50] = A*xb23 + B*ub23 - xb33
    c[51:52] = A*xb24 + B*ub24 - xb34

    c[53:54] = A*xb31 + B*ub31 - xb41
    c[55:56] = A*xb32 + B*ub32 - xb42
    c[57:58] = A*xb33 + B*ub33 - xb43
    c[59:60] = A*xb34 + B*ub34 - xb44

    c[61] = ub21 + k2'*xb21
    c[62] = ub22 + k2'*xb22
    c[63] = ub23 + k2'*xb23
    c[64] = ub24 + k2'*xb24

    c[65] = ub31 + k3'*xb31
    c[66] = ub32 + k3'*xb32
    c[67] = ub33 + k3'*xb33
    c[68] = ub34 + k3'*xb34

    # c
    # Σc = (xa31*xa31' + xa32*xa32' + xa33*xa33' + xa34*xa34'
    #       + xb31*xb31' + xb32*xb32' + xb33*xb33' + xb34*xb34'
    #       + 1.0e-8*I)
    Σc = (xb31*xb31' + xb32*xb32' + xb33*xb33' + xb34*xb34'
          + 1.0e-8*I)


    cols_c = Array(cholesky(Σc).U)
    β = 0.001

    c[69:70] = β*cols_c[:,1] - xc31
    c[71:72] = -1.0*β*cols_c[:,1] - xc32
    c[73:74] = β*cols_c[:,2] - xc33
    c[75:76] = -1.0*β*cols_c[:,2] - xc34

    c[77:78] = A*xc31 + B*uc31 - xc41
    c[79:80] = A*xc32 + B*uc32 - xc42
    c[81:82] = A*xc33 + B*uc33 - xc43
    c[83:84] = A*xc34 + B*uc34 - xc44

    c[85] = uc31 + k3'*xc31
    c[86] = uc32 + k3'*xc32
    c[87] = uc33 + k3'*xc33
    c[88] = uc34 + k3'*xc34

    return c
end

c0 = zeros(m_nlp)
con!(c0,z0_nom)

prob = Problem(n_nlp,m_nlp,obj,con!,true)

z_sol = solve(z0_nom,prob)
K_sample = [reshape(z_sol[1:2],m,n),
            reshape(z_sol[3:4],m,n),
            reshape(z_sol[5:6],m,n)]
println("K error: $(sum([norm(vec(K_sample[t] - K[t])) for t = 1:T-1]))")
