#ifndef RADIATION_RADIATION_TRANSPORT_HPP_
#define RADIATION_RADIATION_TRANSPORT_HPP_
// AthenaXXX: BSD-3-Clause; see LICENSE.
// Shared scalar face operator for Minkowski radiation transport.
#include <cmath>
#include "athena.hpp"

namespace radiation {
// a: signed ray speed; left/right: reconstructed transport intensity;
// tau: dimensionless numerical optical-depth parameter (not simply chi*dx);
// velocity: arithmetic face average of the fluid three-velocity component.
// Signal speeds intentionally use a, not a-adv. This preserves the tested scheme.
// This operator is not positivity preserving; RKUpdate applies the existing floor.
KOKKOS_INLINE_FUNCTION
Real OpticalDepthSplitFlux(Real a, Real left, Real right, Real tau, Real velocity) {
  if (fabs(a) <= 1.e-30 || tau <= 1.e-8) {
    return a * ((a > 0.) ? left : right);
  }
  const Real tau2 = tau * tau;
  const Real weight = -expm1(-tau2);
  const Real aa = sqrt(weight) / tau;
  const Real bb = sqrt(-expm1(-tau2 * tau2)) / tau;
  const Real sp = fabs(a) * ((a > 0.) ? aa : bb);
  const Real sm = -fabs(a) * ((a > 0.) ? bb : aa);
  const Real adv = weight * velocity;
  const Real rel = a - adv;
  return (sp * rel * left - sm * rel * right + sp * sm * (right - left)) / (sp - sm) +
         adv * ((adv > 0.) ? left : right);
}
// Frozen-face GR extension. zeta=e_(0)^d, h=sqrt(gamma^{dd});
// mu and vn are ray/fluid projections onto the spatial unit face normal.
// The returned flux evolves the existing U=n^t q, q=n_t I variable.
KOKKOS_INLINE_FUNCTION
Real GRSpatialSplitFlux(Real zeta, Real h, Real mu, Real vn, Real left, Real right, Real tau) {
  const Real a = zeta + h * mu;
  if (zeta == 0. && h == 1.) {
    return OpticalDepthSplitFlux(mu, left, right, tau, vn);
  }
  // Preserve one-way light-cone transport and avoid tangent/invalid face fans.
  if (tau <= 1.e-8 || fabs(mu) <= 1.e-30 || fabs(vn) >= 1. || zeta + h <= 0. || zeta - h >= 0.) {
    return a * ((a > 0.) ? left : right);
  }
  const Real t2 = tau * tau, weight = -expm1(-t2);
  const Real aa = sqrt(weight) / tau, bb = sqrt(-expm1(-t2 * t2)) / tau;
  const Real drift = (1. - weight) * zeta;
  const Real sp = fmax(0., h * fabs(mu) * ((mu > 0.) ? aa : bb) + drift);
  const Real sm = fmin(0., -h * fabs(mu) * ((mu > 0.) ? bb : aa) + drift);
  if (sp == sm) return a * ((a > 0.) ? left : right);
  const Real adv = weight * (zeta + h * vn), rel = a - adv;
  return (sp * rel * left - sm * rel * right + sp * sm * (right - left)) / (sp - sm) +
         adv * ((adv > 0.) ? left : right);
}
}  // namespace radiation
#endif  // RADIATION_RADIATION_TRANSPORT_HPP_
