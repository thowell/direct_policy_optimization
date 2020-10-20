rot(x) = [cos(x) -sin(x); sin(x) cos(x)]

orig = [3.0;3.0]
pend_orig = [3.75;2.25]
p = [0.0;-0.75]

ang = 135.0/2.0
rad = ang/180.0*pi

rot(rad)*p + pend_orig

360-22.5

w = 0.5
h1 = 1.5
h2 = 3.0

wt = 0.3
ht = 0.4
pts = [[w;0.0],
	   [w;h1],
	   [0;h1+0.5],
	   [-w;h1],
	   [-w;-h2],
	   [w;-h2],
	   [w;0.0],
	   [wt;-h2],
	   [w;-h2-ht],
	   [-w;-h2-ht],
	   [-wt;-h2],
	   [0;-h2-ht]]

_pts = [
		[-w-0.25;0],
		[-w-0.25;-h2-ht]
]
pts_rot = [rot(pi/4)*pt + orig for pt in pts]
_pts_rot = [rot(pi/4)*pt + orig for pt in _pts]

t_pts = [
	   	[0.5;0.0]
]

t_pts_rot = [rot(-pi/10.0)*pt + [5.40;0.596] for pt in t_pts]

norm(t_pts_rot[1])

(3-ht+h2)/2.0

rot(pi/4)*[0;-(ht+h2)/2.0] + orig
