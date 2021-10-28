// Parameters
blt = 0.75;
r_head = 32;
tail = 64;
tail_angle = 0 * Pi / 180;     // proportional to Re^-1/2
// in boundary layer...
n_theta = 19;
n_r = 7;

// Calculated parameters
head_x = r_head*Cos(tail_angle);
head_y = r_head*Sin(tail_angle);

// Sphere (_sph)
lc_near = 3.14/(n_theta-1);
origin = newp; Point(origin) = {0, 0, 0, lc_near};
psph1 = newp; Point(psph1) = {0, -1, 0, lc_near};
psph2 = newp; Point(psph2) = {0, 1, 0, lc_near};
lsph1 = newl; Circle(lsph1) = {psph1, origin, psph2};

// Boundary Layer (_bl)
lc_bl = 3.14*(1+blt)/(n_theta-1);
pbl1 = newp; Point(pbl1) = {0, -(1+blt), 0, lc_bl};
pbl2 = newp; Point(pbl2) = {0, 1+blt, 0, lc_bl};
lbl1 = newl; Circle(lbl1) = {pbl1, origin, pbl2};
lbl2 = newl; Line(lbl2) = {psph1, pbl1};
lbl3 = newl; Line(lbl3) = {pbl2, psph2};

// Wake
r_wake = 2.5;
lc_bl = 3.14*r_wake/(n_theta-1);
pw0 = newp; Point(pw0) = {0, -r_wake, 0};
pw1 = newp; Point(pw1) = {0, -r_wake*2, 0, lc_bl};
pw2 = newp; Point(pw2) = {r_wake, -r_wake, 0, lc_bl};
pw3 = newp; Point(pw3) = {r_wake, 0, 0, lc_bl};
pw4 = newp; Point(pw4) = {0, r_wake, 0, lc_bl};
lw1 = newl; Line(lw1) = {pbl1, pw1};
lw2 = newl; Circle(lw2) = {pw1, pw0, pw2};
lw3 = newl; Line(lw3) = {pw2, pw3};
lw4 = newl; Circle(lw4) = {pw3, origin, pw4};
lw5 = newl; Line(lw5) = {pw4, pbl2};

// Head section
lc_head = 0.2*r_head;
ph1 = newp; Point(ph1) = {head_x, head_y, -0, lc_head};
ph2 = newp; Point(ph2) = {0, r_head, 0, lc_head};
lh1 = newl; Circle(lh1) = {ph1, origin, ph2};
lh2 = newl; Line(lh2) = {ph2, pw4};

// Tail section
tail_width = r_head + tail*Tan(tail_angle);
lc_tail = 0.20*tail_width;
pt1 = newp; Point(pt1) = {0, -tail, 0, lc_tail};
pt2 = newp; Point(pt2) = {tail_width, -tail, 0, lc_tail};
lt1 = newl; Line(lt1) = {pw1, pt1};
lt2 = newl; Line(lt2) = {pt1, pt2};
lt3 = newl; Line(lt3) = {pt2, ph1};

// Boundary layer mesh
llbl = newreg; Line Loop(llbl) = {-lsph1, lbl2, lbl1, lbl3};
sbl = news; Plane Surface(sbl) = {llbl};
Transfinite Line{lbl2, -lbl3} = n_r Using Progression 1.35;
Transfinite Line{lsph1, lbl1} = n_theta;
Transfinite Surface{sbl};

// Inner bulk fluid
llibf = newreg; Line Loop(llibf) = {lw1, lw2, lw3, lw4, lw5, -lbl1};
sibf = news; Plane Surface(llibf) = {llibf};

// Outer bulk fluid
llbf = newreg; Line Loop(llbf) = {lt1, lt2, lt3, lh1, lh2, -lw4, -lw3, -lw2};
sbf = news; Plane Surface(sbf) = {llbf};

// Physical groups/regions
Physical Line("symaxis") = {lt1, lw1, lbl2, lbl3, lw5, lh2};
Physical Line("sphere") = {lsph1};
Physical Line("shell") = {lh1, lt2, lt3};
// Physical Line("shell-head") = {lh1};
// Physical Line("shell-wall") = {lt3};
// Physical Line("shell-tail") = {lt2};
Physical Surface("bulk") = {sbl, 17, sbf};
Coherence;
