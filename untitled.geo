// Gmsh project created on Thu Aug 08 10:29:37 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {-0.5, 0.5, 0, 1.0};
//+
Point(2) = {0.5, 0.5, 0, 1.0};
//+
Point(3) = {1.1, -0, 0, 1.0};
//+
Point(4) = {0.5, -0.2, 0, 1.0};
//+
Point(5) = {-0.5, -0.2, 0, 1.0};
//+
BSpline(1) = {1, 2, 3, 4, 5, 1};
