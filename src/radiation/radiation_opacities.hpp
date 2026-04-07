#ifndef RADIATION_RADIATION_OPACITIES_HPP_
#define RADIATION_RADIATION_OPACITIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_opacities.hpp
//! \brief implements functions for computing opacities

#include <math.h>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OpacityFunction
//! \brief sets sigma_a, sigma_s, sigma_p in the comoving frame

KOKKOS_INLINE_FUNCTION
void OpacityFunction(// density and density scale
                     const Real dens, const Real density_scale,
                     // temperature and temperature scale
                     const Real temp, const Real temperature_scale,
                     // length scale, adiabatic index minus one, mean molecular weight
                     const Real length_scale, const Real gm1, const Real mu,
                     // power law opacities
                     const bool pow_opacity,
                     const Real rosseland_coef, const Real planck_minus_rosseland_coef,
                     // spatially and temporally constant opacities
                     const Real k_a, const Real k_s, const Real k_p,
                     // output sigma
                     Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  if (pow_opacity) {  // power law opacity (accounting for diff b/w Ross & Planck)
    Real power_law = (dens*density_scale)*pow(gm1*mu/(temp*temperature_scale), 3.5);
    Real k_a_r = rosseland_coef * power_law;
    Real k_a_p = planck_minus_rosseland_coef * power_law;
    sigma_a = dens*k_a_r*density_scale*length_scale;
    sigma_p = dens*k_a_p*density_scale*length_scale;
    sigma_s = dens*k_s  *density_scale*length_scale;
  } else {  // spatially and temporally constant opacity
    sigma_a = dens*k_a*density_scale*length_scale;
    sigma_p = dens*k_p*density_scale*length_scale;
    sigma_s = dens*k_s*density_scale*length_scale;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void GetArrayLocation
//! \brief bracketing indices for piecewise-linear / bilinear lookup on sorted axis

KOKKOS_INLINE_FUNCTION
void GetArrayLocation(const Real value, const DualArray1D<Real> &in_arr,
                      int &loc_l, int &loc_r) {
  int arr_size = in_arr.extent_int(0) - 1;
  loc_l = 0;
  loc_r = 0;
  while ((value > in_arr.d_view(loc_r)) && (loc_r < arr_size)) {
    loc_r++;
  }
  loc_l = loc_r - 1;
  if (loc_l < 0) loc_l = 0;
  if (loc_r == arr_size && (value > in_arr.d_view(loc_r))) {
    loc_l = loc_r;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BilinearInterpolation
//! \brief bilinear interpolation on (x,y) using corner indices

KOKKOS_INLINE_FUNCTION
void BilinearInterpolation(const Real in_x, const Real in_y,
                           const int nx_l, const int nx_r,
                           const int ny_l, const int ny_r,
                           const DualArray1D<Real> &in_xarr,
                           const DualArray1D<Real> &in_yarr,
                           const DualArray2D<Real> &in_table,
                           Real &output) {
  Real data_y1_x1 = in_table.d_view(ny_l, nx_l);
  Real data_y1_x2 = in_table.d_view(ny_l, nx_r);
  Real data_y2_x1 = in_table.d_view(ny_r, nx_l);
  Real data_y2_x2 = in_table.d_view(ny_r, nx_r);

  Real x_1 = in_xarr.d_view(nx_l);
  Real x_2 = in_xarr.d_view(nx_r);
  Real y_1 = in_yarr.d_view(ny_l);
  Real y_2 = in_yarr.d_view(ny_r);

  if (nx_l == nx_r) {
    if (ny_l == ny_r) {
      output = data_y1_x1;
    } else {
      output = data_y1_x1 + (data_y2_x1 - data_y1_x1) * (in_y - y_1)/(y_2 - y_1);
    }
  } else {
    if (ny_l == ny_r) {
      output = data_y1_x1 + (data_y1_x2 - data_y1_x1) * (in_x - x_1)/(x_2 - x_1);
    } else {
      output = data_y1_x1 * (y_2 - in_y) * (x_2 - in_x)
             + data_y2_x1 * (in_y - y_1) * (x_2 - in_x)
             + data_y1_x2 * (y_2 - in_y) * (in_x - x_1)
             + data_y2_x2 * (in_y - y_1) * (in_x - x_1);
      output /= ((y_2 - y_1) * (x_2 - x_1));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void TableOpacity
//! \brief Rosseland and Planck mean opacities from 2D tables (values in cgs); splits
//! Rosseland into absorption vs electron scattering using k_elec (same units as table).

KOKKOS_INLINE_FUNCTION
void TableOpacity(const Real dens, const Real density_scale,
                  const Real temp, const Real temperature_scale,
                  const Real length_scale, const bool use_t_r,
                  const DualArray1D<Real> &ross_rho, const DualArray1D<Real> &ross_t,
                  const DualArray1D<Real> &planck_rho,
                  const DualArray1D<Real> &planck_t,
                  const DualArray2D<Real> &ross_table,
                  const DualArray2D<Real> &planck_table, const Real k_elec,
                  Real& sigma_a, Real& sigma_s, Real& sigma_p) {
  Real log_x = log10(dens * density_scale);
  Real log_t = log10(temp * temperature_scale);
  if (use_t_r) {
    log_x = log_x - 3.0 * log_t + 18.0;
  }

  int nx_l = 0;
  int nx_r = 0;
  GetArrayLocation(log_x, ross_rho, nx_l, nx_r);
  int ny_l = 0;
  int ny_r = 0;
  GetArrayLocation(log_t, ross_t, ny_l, ny_r);

  Real kappa_ross = 0.0;
  BilinearInterpolation(log_x, log_t, nx_l, nx_r, ny_l, ny_r,
                        ross_rho, ross_t, ross_table, kappa_ross);

  if (kappa_ross < k_elec) {
    if (temp * temperature_scale > 1.e4) {
      sigma_s = kappa_ross * dens * density_scale * length_scale;
      sigma_a = 0.0;
    } else {
      sigma_s = 0.0;
      sigma_a = kappa_ross * dens * density_scale * length_scale;
    }
  } else {
    sigma_s = k_elec * dens * density_scale * length_scale;
    sigma_a = (kappa_ross - k_elec) * dens * density_scale * length_scale;
  }

  Real kappa_planck = 0.0;
  nx_l = 0;
  nx_r = 0;
  GetArrayLocation(log_x, planck_rho, nx_l, nx_r);
  ny_l = 0;
  ny_r = 0;
  GetArrayLocation(log_t, planck_t, ny_l, ny_r);
  BilinearInterpolation(log_x, log_t, nx_l, nx_r, ny_l, ny_r,
                        planck_rho, planck_t, planck_table, kappa_planck);
  sigma_p = kappa_planck * dens * density_scale * length_scale;
  return;
}

#endif // RADIATION_RADIATION_OPACITIES_HPP_
