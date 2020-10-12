rot(x) = [cos(x) -sin(x); sin(x) cos(x)]

orig = [3.0;3.0]
pend_orig = [3.75;2.25]
p = [0.0;-1.0]

ang = 135.0/2.0
rad = ang/180.0*pi

rot(rad)*p + pend_orig

315-22.5
