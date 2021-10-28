// R_o = 1000;  // Outer radius
R_o = 100;   // Outer radius
Point(1) = {0, 0, 0, 1};
Point(2) = {-0, -1, 0, 1};
Point(3) = {0, 1, 0, 1};
Point(4) = {0, -R_o, 0, 1};
Point(5) = {0, R_o, 0, 1};
Circle(1) = {2, 1, 3};
Circle(2) = {4, 1, 5};
Transfinite Line {1, 2} = 10;
Line(3) = {2, 4};
Line(4) = {5, 3};
// Transfinite Line {3, -4} = 24 Using Progression 1.35;  // for R_o = 1000
Transfinite Line {3, -4} = 16 Using Progression 1.35;  // for R_o = 100
Line Loop(6) = {4, -1, 3, 2};
Plane Surface(6) = {6};
Transfinite Surface {6} = {2,4,5,3};
Recombine Surface {6};
Physical Line("sphere") = {1};
Physical Line("shell") = {2};
Physical Line("symaxis") = {3, 4};
Physical Surface("interior") = {6};
