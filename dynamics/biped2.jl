mutable struct Biped2{T}

end

function D(model::Biped2,q)
    D11 =

    D12 =

    D13 =

    D14 =

    D15 =

    D22 =

    D23 =

    D24 =

    D25 =

    D33 =

    D34 =

    D35 =

    D44 =

    D45 =

    D55 =

end

function C(model::Biped2,q,q̇)
    C11 =

    C12 =

    C13 =

    C14 =

    C15 =

    C21 =

    C22 =

    C23 =

    C24 =

    C25 =

    C31 =

    C32 =

    C33 =

    C34 =

    C35 =

    C41 =

    C42 =

    C43 =

    C44 =

    C45 =

    C51 =

    C52 =

    C53 =

    C54 =

    C55 =

end

function G(model::Biped2,q)
    G1 =

    G2 =

    G3 =

    G4 =

    G5 =


end

function B(model::Biped2,::)

end

function dynamics(model::Biped2,x,u)
    @SVector []
end

nx,nu =
model = Biped2(,nx,nu)
