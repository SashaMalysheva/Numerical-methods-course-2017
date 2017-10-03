ACTIVITY 8

Given
Two tables of values ​​of the function and its derivative on a uniform ordered grid of nodes (the number of nodes of M). The degrees of polynomials are given for comparison.

TASKS.
1. Construct an interpolation Hermite polynomial and Hermite splines. The interpolation Hermite polynomial is constructed on a uniform grid of nodes separated from the main one. (the given degrees of the polynomial are chosen so that this can be done by preserving the extreme points of the grid).
Conduct comparisons by the following criteria:
Visual comparison of graphs with the application of points of tabular function.
The number of nodes where the value of the polynomial is closer to the value of the table function than the value of the spline
Maximum absolute error for a polynomial and a spline for all nodes
The maximum relative error for a polynomial and a spline for all nodes
The average absolute error for a polynomial and a spline for all nodes
The average relative error for a polynomial and spline for all nodes

2. From the main grid, select an arbitrary grid (not necessarily uniform) from not more than half the nodes of the main grid (M / 2), so as to obtain an interpolation polynomial having the least absolute (or relative) error in the remaining nodes. Compare with a spline built on a dedicated grid.
3. Construct an Hermite spline without using a table of derivatives, i.e. calculate them according to the function value table. Compare with n1
4. Calculate the values ​​of the derivatives of the Hermite polynomial and Hermite splines. Find the maximum absolute and relative error at the nodes.
