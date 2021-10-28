#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "glnodes.c"

typedef unsigned int uint;


double legeval(double x, uint n) {
  // Evaluates the order `n` Legendre polynomial at `x` using a recursion
  // relation.

  double p0 = 1.;
  if (n == 0) {
    return p0;
  }

  double p1 = x;
  if (n == 1) {
    return p1;
  }

  double p2;

  uint i = 1;
  while (true) {
    // Recursion relation
    p2 = ((2*i + 1) * x * p1 - i * p0) / (i + 1);
    if (i == n-1)
      return p2;
    p0 = p1;
    p1 = p2;
    i++;
  }
}


double barycentric_lagrange(double f[], uint n, double x) {

  assert(n>1 && n<10);

  double nodes[n];
  double bary_wts[n];

  double numer = 0;
  double denom = 0;
  double kern;

  uint i1, i2, j;

  if (n % 2 == 1) {
    nodes[n/2] = 0.;
    bary_wts[n/2] = gl_bary_weights_dat[n-2][0];
    i1 = n/2 + 1;
    i2 = n/2 - 1;
    j = 1;
  } else {
    i1 = n/2;
    i2 = n/2 - 1;
    j = 0;
  }

  while (i1 < n) {
    nodes[i1] = gl_nodes_dat[n-2][j];
    nodes[i2] = -gl_nodes_dat[n-2][j];
    bary_wts[i1] = gl_bary_weights_dat[n-2][j];
    bary_wts[i2] = gl_bary_weights_dat[n-2][j];
    if (n % 2 == 0)
      bary_wts[i2] *= -1;
    i1 += 1;
    i2 -= 1;
    j += 1;
  }

  // evaluate the interpolating Lagrange polynomial in barycentric form
  for (uint i=0; i<n; i++) {
    kern = bary_wts[i] / (x - nodes[i]);
    // if interpolating at a data point, we are dividing by zero
    if (!isfinite(kern))
      return f[i];  // just return the data point instead
    /* don't worry... this barycentric formula ensures that potentially large
       floating-point rounding errors due to dividing by almost zero (if we are
       interpolating very near a quadrature node) cancel out */
    numer += kern * f[i];
    denom += kern;
  }

  return numer / denom;
}


int main()
{
  double fvals[] = {-1, -0.65465, 0, 0.65465, 1};
  double interp_f;

  interp_f = barycentric_lagrange(fvals, 5, 0.654);
  printf("f(x) = %g\n", interp_f);
}
