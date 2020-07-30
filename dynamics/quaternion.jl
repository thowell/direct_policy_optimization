struct Quaternion{T}
    w::T
    x::T
    y::T
    z::T
end

Quaternion(v::AbstractVector) = Quaternion(v[1],v[2],v[3],v[4])
Quaternion(w::Real, v::AbstractVector) = Quaternion(w,v[1],v[2],v[3])
Base.vec(q::Quaternion) = SVector(q.x, q.y, q.z)
StaticArrays.SVector(q::Quaternion) = SVector(q.w, q.x, q.y, q.z)
scalar(q::Quaternion) = q.w
function Base.:*(q2::Quaternion, q1::Quaternion)
    w = scalar(q1)*scalar(q2) - vec(q1)'vec(q2)
    v = scalar(q1)*vec(q2) + scalar(q2)*vec(q1) + vec(q2) × vec(q1)
    Quaternion(w,v)
end
Base.:*(A::AbstractMatrix, q::Quaternion) = A*SVector(q)
Base.:*(q::Quaternion, A::AbstractMatrix) = SVector(q)'A
Base.:*(a::Real, q::Quaternion) = Quaternion(a*scalar(q), a*vec(q))
function Base.:*(q::Quaternion, r::AbstractVector)
    @assert length(r) == 3
    vec(q*Quaternion(zero(eltype(r)),r)*inv(q))
end
Base.inv(q::Quaternion) = Quaternion(q.w, -q.x, -q.y, -q.z)
LinearAlgebra.normalize(q::Quaternion) = Quaternion(normalize(SVector(q)))
