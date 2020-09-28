mutable struct Biped{T}
    I1::T
    m1::T
    d1::T
    l1::T

    I2::T
    m2::T
    d2::T
    l2::T

    I3::T
    m3::T
    d3::T
    l3::T

    I4::T
    m4::T
    d4::T
    l4::T

    I5::T
    m5::T
    d5::T
    l5::T

    g::T

    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::Biped, x, u, w)


    return @SVector
end

nx,nu,nw = 10,4,0
model = Biped(1.0,1.0,1.0,1.0,
              1.0,1.0,1.0,1.0,
              1.0,1.0,1.0,1.0,
              1.0,1.0,1.0,1.0,
              1.0,1.0,1.0,1.0,
              9.81,
              nx,nu,nw)

function pij(model::Biped)
  # model parameters
  I1 = model.I1
  m1 = model.m1
  d1 = model.d1
  l1 = model.l1

  I2 = model.I2
  m2 = model.m2
  d2 = model.d2
  l2 = model.l2

  I3 = model.I3
  m3 = model.m3
  d3 = model.d3
  l3 = model.l3

  I4 = model.I4
  m4 = model.m4
  d4 = model.d4
  l4 = model.l4

  I5 = model.I5
  m5 = model.m5
  d5 = model.d5
  l5 = model.l5

  # pij

  p12 = m2*d2*l1 + (m3 + m4 + m5)*l1*l2

  p13 = m3*l1*d3

  p14 = m4*l1*(l4 - d4) + m5*l1*l4

  p15 = m5*l1*(l5 - d5)

  p23 = m3*l2*d3

  p24 = m4*l2*(l4 - d4) + m5*l2*l4

  p25 = m5*l2*(l5 - d5)

  p45 = m5*l4*(l5 - d5)

  return (p12,p13,p14,p15,
              p23,p24,p25,
                      p45)
end

function Di(model::Biped,θ)
    # configuration
    θ1 = θ[1]
    θ2 = θ[2]
    θ3 = θ[3]
    θ4 = θ[4]
    θ5 = θ[5]

    # model parameters
    I1 = model.I1
    m1 = model.m1
    d1 = model.d1
    l1 = model.l1

    I2 = model.I2
    m2 = model.m2
    d2 = model.d2
    l2 = model.l2

    I3 = model.I3
    m3 = model.m3
    d3 = model.d3
    l3 = model.l3

    I4 = model.I4
    m4 = model.m4
    d4 = model.d4
    l4 = model.l4

    I5 = model.I5
    m5 = model.m5
    d5 = model.d5
    l5 = model.l5

    # pij
    p12,p13,p14,p15,p23,p24,p25,p45 = pij(model)

    # Dij
    D11 = I1 + m1*d1^2 + (m2 + m3 + m4 + m5)*l1^2

    D12 = p12*cos(θ1 - θ2)

    D13 = p13*cos(θ1 - θ3)

    D14 = p14*cos(θ1 + θ4)

    D15 = p15*cos(θ1 + θ5)

    D21 = D12

    D22 = I2 + m2*d2^2 + (m3 + m4 + m5)*l2^2

    D23 = p23*cos(θ2 - θ3)

    D24 = p24*cos(θ2 + θ4)

    D25 = p25*cos(θ2 + θ5)

    D31 = D13

    D32 = D23

    D33 = I3 + m3*d3^2

    D34 = 0.0

    D35 = 0.0

    D41 = D14

    D42 = D24

    D43 = D34

    D44 = I4 + m4*(l4 - d4)^2 + m5*l4^2

    D45 = p45*cos(θ4 - θ5)

    D51 = D15

    D52 = D25

    D53 = D35

    D54 = D45

    D55 = I5 + m5*(l5 - d5)^2

    return (D11, D12, D13, D14, D15,
            D21, D22, D23, D24, D25,
            D31, D32, D33, D34, D35,
            D41, D42, D43, D44, D45,
            D51, D52, D53, D54, D55)
end

pij(model)
Di(model,rand(5))

function hijj(model::Biped,θ)
    # configuration
    θ1 = θ[1]
    θ2 = θ[2]
    θ3 = θ[3]
    θ4 = θ[4]
    θ5 = θ[5]

    # pij
    p12,p13,p14,p15,p23,p24,p25,p45 = pij(model)

    # hijj
    h122 = p12*sin(θ1 - θ2)

    h133 = p13*sin(θ1 - θ3)

    h144 = -p14*sin(θ1 + θ4)

    h155 = -p15*sin(θ1 + θ4)

    h211 = -p12*sin(θ1 - θ2)

    h233 = p23*sin(θ2 - θ3)

    h244 = -p24*sin(θ2 + θ4)

    h255 = -p25*sin(θ2 + θ5)

    h311 = -p13*sin(θ1 - θ3)

    h322 = -p23*sin(θ2 - θ3)

    h344 = 0.0

    h355 = 0.0

    h411 = -p14*sin(θ1 + θ4)

    h422 = -p24*sin(θ2 + θ4)

    h433 = 0.0

    h455 = p45*sin(θ4 - θ5)

    h511 = -p15*sin(θ1 + θ5)

    h522 = -p25*sin(θ2 + θ5)

    h533 = 0.0

    h544 = -p45*sin(θ4 - θ5)

    return (h122,h133,h144,h155,
            h211,h233,h244,h255,
            h311,h322,h344,h355,
            h411,h422,h433,h455,
            h511,h522,h533,h544)
end

function G(model::Biped,θ)
    # configuration
    θ1 = θ[1]
    θ2 = θ[2]
    θ3 = θ[3]
    θ4 = θ[4]
    θ5 = θ[5]

    # model parameters
    I1 = model.I1
    m1 = model.m1
    d1 = model.d1
    l1 = model.l1

    I2 = model.I2
    m2 = model.m2
    d2 = model.d2
    l2 = model.l2

    I3 = model.I3
    m3 = model.m3
    d3 = model.d3
    l3 = model.l3

    I4 = model.I4
    m4 = model.m4
    d4 = model.d4
    l4 = model.l4

    I5 = model.I5
    m5 = model.m5
    d5 = model.d5
    l5 = model.l5

    g = model.g

    G1 = -1.0*(m1*d1 + (m2 + m3 + m4 + m5)*l1)*g*sin(θ1)
    G2 = -1.0*(m2*d2 + (m3 + m4 + m5)*l2)*g*sin(θ2)
    G3 = -1.0*(m3*d3)*g*sin(θ3)
    G4 = (m4*(l4 - d4) + m5*l4)*g*sin(θ4)
    G5 = (m5*(l5 - d5))*g*sin(θ5)

    return (G1,G2,G3,G4,G5)
end

function E(model::Biped)
    @SMatrix [1.0 0.0 0.0 0.0;
              -1.0 1.0 0.0 0.0;
              0.0 -1.0 1.0 0.0;
              0.0 0.0 1.0 1.0;
              0.0 0.0 0.0 -1.0]
end
