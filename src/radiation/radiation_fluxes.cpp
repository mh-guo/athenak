//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fluxes.cpp
//  \brief Calculate 3D fluxes for radiation

#include <float.h>

#include "athena.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "units/units.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "radiation_opacities.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "radiation.hpp"
#include "radiation_transport.hpp"
#include "reconstruct/plm.hpp"
#include "reconstruct/ppm.hpp"
#include "reconstruct/wenoz.hpp"
#include "reconstruct/teno.hpp"

namespace radiation {
void Radiation::PrepareTransportState() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const Real kappa_s_ = kappa_s;
  const bool gr_split = gr_split_flux;
  DvceArray5D<Real> fluid_w;
  if (is_hydro_enabled) fluid_w = pmy_pack->phydro->w0;
  if (is_mhd_enabled) fluid_w = pmy_pack->pmhd->w0;
  auto &block_size = pmy_pack->pmb->mb_size;
  // Cell transport coefficients use precisely the source-term opacity helper.
  // Tetrad fluid velocity and proper coordinate lengths enter the directional
  // optical depth used by the explicit transport correction.
  if (transport_state.extent(0) != static_cast<size_t>(nmb1 + 1) ||
      transport_state.extent(2) != fluid_w.extent(2) ||
      transport_state.extent(3) != fluid_w.extent(3) ||
      transport_state.extent(4) != fluid_w.extent(4))
    transport_state = DvceArray5D<Real>("transport_state", nmb1 + 1, 11, fluid_w.extent(2),
                                        fluid_w.extent(3), fluid_w.extent(4));
  auto state = transport_state;
  auto nt = norm_to_tet;
  auto excision_flux_ = pmy_pack->pcoord->excision_flux;
  bool flat = pmy_pack->pcoord->coord_data.is_minkowski;
  Real spin = pmy_pack->pcoord->coord_data.bh_spin;
  Real gm1 =
      (is_hydro_enabled ? pmy_pack->phydro->peos : pmy_pack->pmhd->peos)->eos_data.gamma - 1.0;
  Real dfloor = (is_hydro_enabled ? pmy_pack->phydro->peos : pmy_pack->pmhd->peos)->eos_data.dfloor;
  Real density_scale = 1., temperature_scale = 1., length_scale = 1., mmw = 1., rc = 1., pc = 0.;
  if (are_units_enabled) {
    density_scale = pmy_pack->punit->density_cgs();
    temperature_scale = pmy_pack->punit->temperature_cgs();
    length_scale = pmy_pack->punit->length_cgs();
    mmw = pmy_pack->punit->mu();
    rc = pmy_pack->punit->rosseland_coef_cgs;
    pc = pmy_pack->punit->planck_minus_rosseland_coef_cgs;
  }
  bool corr = correct_radsrc_opacity, use_r = op_table_use_r;
  const bool table = table_opacity, power = power_opacity, mhd = is_mhd_enabled;
  DvceArray5D<Real> bcc;
  if (mhd) bcc = pmy_pack->pmhd->bcc0;
  Real dfop = dfloor_opacity, dmax = dens_trunc_max, ttr = tau_truncation;
  Real sigres = sigmoid_residual, ka = kappa_a, kp = kappa_p, kelec = k_elec_opacity;
  auto rr = ross_rho, rt = ross_t, pr = planck_rho, pt = planck_t;
  auto rtab = ross_table, ptab = planck_table;
  par_for(
      "transport_cell_state", DevExeSpace(), 0, nmb1, 0, fluid_w.extent(2) - 1, 0,
      fluid_w.extent(3) - 1, 0, fluid_w.extent(4) - 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        auto sz = block_size.d_view(m);
        Real x = CellCenterX(i - is, indcs.nx1, sz.x1min, sz.x1max);
        Real y = CellCenterX(j - js, indcs.nx2, sz.x2min, sz.x2max);
        Real z = CellCenterX(k - ks, indcs.nx3, sz.x3min, sz.x3max);
        Real g[4][4], gi[4][4];
        ComputeMetricAndInverse(x, y, z, flat, spin, g, gi);
        Real vx = fluid_w(m, IVX, k, j, i), vy = fluid_w(m, IVY, k, j, i),
             vz = fluid_w(m, IVZ, k, j, i);
        Real gamma = sqrt(1. + g[1][1] * vx * vx + g[2][2] * vy * vy + g[3][3] * vz * vz +
                          2. * (g[1][2] * vx * vy + g[1][3] * vx * vz + g[2][3] * vy * vz));
        Real sa, ss, sp;
        const Real bx = mhd ? bcc(m,IBX,k,j,i) : 0.;
        const Real by = mhd ? bcc(m,IBY,k,j,i) : 0.;
        const Real bz = mhd ? bcc(m,IBZ,k,j,i) : 0.;
        const auto op = RadiationFluidCellSigmas(
            fluid_w(m, IDN, k, j, i), vx, vy, vz, fluid_w(m, IEN, k, j, i), gm1, g, gi,
            excision_flux_, m, k, j, i, sz.dx1, sz.dx2, sz.dx3, corr, mhd, bx, by, bz, dfloor,
            dfop, dmax, ttr, sigres, kappa_s_, table, power, density_scale, temperature_scale,
            length_scale, use_r, rr, rt, pr, pt, rtab, ptab, kelec, mmw, rc, pc, ka, kp, sa, ss,
            sp);
        state(m, 0, k, j, i) = op.transport;
        for (int a = 0; a < 4; ++a)
          state(m, a + 1, k, j, i) = nt(m, a, 0, k, j, i) * gamma + nt(m, a, 1, k, j, i) * vx +
                                     nt(m, a, 2, k, j, i) * vy + nt(m, a, 3, k, j, i) * vz;
        for (int a = 1; a <= 3; ++a) {
          // Inverse spatial metric, not inverse spacetime metric.
          const Real hinv2 = gi[a][a] - gi[0][a] * gi[0][a] / gi[0][0];
          state(m, a + 4, k, j, i) = gr_split ? 1. / sqrt(hinv2) : sqrt(g[a][a]);
          // Spatial normal-observer velocity in the common coordinate basis.
          state(m, a + 7, k, j, i) = fluid_w(m, IVX + a - 1, k, j, i) / gamma;
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn  void Radiation::CalculateFluxes
//! \brief Compute radiation fluxes

TaskStatus Radiation::CalculateFluxes(Driver *pdriver, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  const auto &recon_method_ = recon_method;
  const bool corrected_flux = optical_depth_flux;
  const bool gr_split = gr_split_flux;
  const Real optical_depth_scale_ = optical_depth_scale;
  auto &block_size = pmy_pack->pmb->mb_size;

  auto &i0_ = i0;
  auto &nh_c_ = nh_c;
  auto &tet_c_ = tet_c;

  if (corrected_flux) PrepareTransportState();
  auto state = transport_state;

  //--------------------------------------------------------------------------------------

  // i-direction

  auto &t1d1 = tet_d1_x1f;
  auto &flx1 = iflx.x1f;
  par_for(
      "rflux_x1", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is, ie + 1,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        // calculate n^1 (hence determining upwinding direction)
        Real n1 =
            t1d1(m, 0, k, j, i) * nh_c_.d_view(n, 0) + t1d1(m, 1, k, j, i) * nh_c_.d_view(n, 1) +
            t1d1(m, 2, k, j, i) * nh_c_.d_view(n, 2) + t1d1(m, 3, k, j, i) * nh_c_.d_view(n, 3);

        // convert to primitive n_0 I
        Real iim1, iicc, iim2, iip1, iim3, iip2;
        iim1 = i0_(m, n, k, j, i - 1) / tet_c_(m, 0, 0, k, j, i - 1);
        iicc = i0_(m, n, k, j, i) / tet_c_(m, 0, 0, k, j, i);
        if (recon_method_ > 0) {
          iim2 = i0_(m, n, k, j, i - 2) / tet_c_(m, 0, 0, k, j, i - 2);
          iip1 = i0_(m, n, k, j, i + 1) / tet_c_(m, 0, 0, k, j, i + 1);
        }
        if (recon_method_ > 1) {
          iim3 = i0_(m, n, k, j, i - 3) / tet_c_(m, 0, 0, k, j, i - 3);
          iip2 = i0_(m, n, k, j, i + 2) / tet_c_(m, 0, 0, k, j, i + 2);
        }

        // reconstruct primitive intensity
        Real iiu, scr;
        switch (recon_method_) {
          case ReconstructionMethod::dc:
            if (n1 > 0.0)
              iiu = iim1;
            else
              iiu = iicc;
            break;
          case ReconstructionMethod::plm:
            if (n1 > 0.0)
              PLM(iim2, iim1, iicc, iiu, scr);
            else
              PLM(iim1, iicc, iip1, scr, iiu);
            break;
          case ReconstructionMethod::ppm4:
            if (n1 > 0.0)
              PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
            else
              PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
            break;
          case ReconstructionMethod::ppmx:
            if (n1 > 0.0)
              PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
            else
              PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
            break;
          case ReconstructionMethod::wenoz:
            if (n1 > 0.0)
              WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
            else
              WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
            break;
          case ReconstructionMethod::teno:
            if (n1 > 0.0)
              TENO(iim3, iim2, iim1, iicc, iip1, iiu, scr);
            else
              TENO(iim2, iim1, iicc, iip1, iip2, scr, iiu);
            break;
          default:
            break;
        }

        // compute x1flux
        flx1(m, n, k, j, i) = n1 * iiu;
        if (corrected_flux && (gr_split || fabs(n1) > 1.e-30)) {
          int l = i - 1, r = i;
          Real dop_l = state(m, 1, k, j, l), dop_r = state(m, 1, k, j, r);
          for (int a = 1; a <= 3; ++a) {
            dop_l -= state(m, a + 1, k, j, l) * nh_c_.d_view(n, a);
            dop_r -= state(m, a + 1, k, j, r) * nh_c_.d_view(n, a);
          }
          Real tau = 2. * optical_depth_scale_ * block_size.d_view(m).dx1 *
                     (state(m, 0, k, j, l) * state(m, 5, k, j, l) * dop_l +
                      state(m, 0, k, j, r) * state(m, 5, k, j, r) * dop_r);
          Real left = iim1, right = iicc, scratch;
          if (recon_method_ == ReconstructionMethod::plm) {
            PLM(iim2, iim1, iicc, left, scratch);
            PLM(iim1, iicc, iip1, scratch, right);
          }
          if (gr_split) {
            const Real zeta = t1d1(m, 0, k, j, i);
            Real h2 = 0.;
            for (int a = 1; a <= 3; ++a) h2 += t1d1(m, a, k, j, i) * t1d1(m, a, k, j, i);
            const Real h = sqrt(h2);
            const Real vn = 0.5 * (state(m, 8, k, j, l) + state(m, 8, k, j, r)) / h;
            flx1(m, n, k, j, i) =
                GRSpatialSplitFlux(zeta, h, (n1 - zeta) / h, vn, left, right, tau);
          } else {
            const Real velocity = 0.5 * (state(m, 2, k, j, l) / state(m, 1, k, j, l) +
                                         state(m, 2, k, j, r) / state(m, 1, k, j, r));
            flx1(m, n, k, j, i) = OpticalDepthSplitFlux(n1, left, right, tau, velocity);
          }
        }
      });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmy_pack->pmesh->multi_d) {
    auto &t2d2 = tet_d2_x2f;
    auto &flx2 = iflx.x2f;
    par_for(
        "rflux_x2", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je + 1, is, ie,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          // calculate n^2 (hence determining upwinding direction)
          Real n2 =
              t2d2(m, 0, k, j, i) * nh_c_.d_view(n, 0) + t2d2(m, 1, k, j, i) * nh_c_.d_view(n, 1) +
              t2d2(m, 2, k, j, i) * nh_c_.d_view(n, 2) + t2d2(m, 3, k, j, i) * nh_c_.d_view(n, 3);

          // convert to primitive n_0 I
          Real iim1, iicc, iim2, iip1, iim3, iip2;
          iim1 = i0_(m, n, k, j - 1, i) / tet_c_(m, 0, 0, k, j - 1, i);
          iicc = i0_(m, n, k, j, i) / tet_c_(m, 0, 0, k, j, i);
          if (recon_method_ > 0) {
            iim2 = i0_(m, n, k, j - 2, i) / tet_c_(m, 0, 0, k, j - 2, i);
            iip1 = i0_(m, n, k, j + 1, i) / tet_c_(m, 0, 0, k, j + 1, i);
          }
          if (recon_method_ > 1) {
            iim3 = i0_(m, n, k, j - 3, i) / tet_c_(m, 0, 0, k, j - 3, i);
            iip2 = i0_(m, n, k, j + 2, i) / tet_c_(m, 0, 0, k, j + 2, i);
          }

          // reconstruct primitive intensity
          Real iiu, scr;
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              if (n2 > 0.0)
                iiu = iim1;
              else
                iiu = iicc;
              break;
            case ReconstructionMethod::plm:
              if (n2 > 0.0)
                PLM(iim2, iim1, iicc, iiu, scr);
              else
                PLM(iim1, iicc, iip1, scr, iiu);
              break;
            case ReconstructionMethod::ppm4:
              if (n2 > 0.0)
                PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::ppmx:
              if (n2 > 0.0)
                PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::wenoz:
              if (n2 > 0.0)
                WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::teno:
              if (n2 > 0.0)
                TENO(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                TENO(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            default:
              break;
          }

          // compute x2flux
          flx2(m, n, k, j, i) = n2 * iiu;
          if (corrected_flux && (gr_split || fabs(n2) > 1.e-30)) {
            int l = j - 1, r = j;
            Real dop_l = state(m, 1, k, l, i), dop_r = state(m, 1, k, r, i);
            for (int a = 1; a <= 3; ++a) {
              dop_l -= state(m, a + 1, k, l, i) * nh_c_.d_view(n, a);
              dop_r -= state(m, a + 1, k, r, i) * nh_c_.d_view(n, a);
            }
            Real tau = 2. * optical_depth_scale_ * block_size.d_view(m).dx2 *
                       (state(m, 0, k, l, i) * state(m, 6, k, l, i) * dop_l +
                        state(m, 0, k, r, i) * state(m, 6, k, r, i) * dop_r);
            Real left = iim1, right = iicc, scratch;
            if (recon_method_ == ReconstructionMethod::plm) {
              PLM(iim2, iim1, iicc, left, scratch);
              PLM(iim1, iicc, iip1, scratch, right);
            }
            if (gr_split) {
              const Real zeta = t2d2(m, 0, k, j, i);
              Real h2 = 0.;
              for (int a = 1; a <= 3; ++a) h2 += t2d2(m, a, k, j, i) * t2d2(m, a, k, j, i);
              const Real h = sqrt(h2);
              const Real vn = 0.5 * (state(m, 9, k, l, i) + state(m, 9, k, r, i)) / h;
              flx2(m, n, k, j, i) =
                  GRSpatialSplitFlux(zeta, h, (n2 - zeta) / h, vn, left, right, tau);
            } else {
              const Real velocity = 0.5 * (state(m, 3, k, l, i) / state(m, 1, k, l, i) +
                                           state(m, 3, k, r, i) / state(m, 1, k, r, i));
              flx2(m, n, k, j, i) = OpticalDepthSplitFlux(n2, left, right, tau, velocity);
            }
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // k-direction. Note order of k,j loops switched

  if (pmy_pack->pmesh->three_d) {
    auto &t3d3 = tet_d3_x3f;
    auto &flx3 = iflx.x3f;
    par_for(
        "rflux_x3", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke + 1, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          // calculate n^3 (hence determining upwinding direction)
          Real n3 =
              t3d3(m, 0, k, j, i) * nh_c_.d_view(n, 0) + t3d3(m, 1, k, j, i) * nh_c_.d_view(n, 1) +
              t3d3(m, 2, k, j, i) * nh_c_.d_view(n, 2) + t3d3(m, 3, k, j, i) * nh_c_.d_view(n, 3);

          // convert to primitive n_0 I
          Real iim1, iicc, iim2, iip1, iim3, iip2;
          iim1 = i0_(m, n, k - 1, j, i) / tet_c_(m, 0, 0, k - 1, j, i);
          iicc = i0_(m, n, k, j, i) / tet_c_(m, 0, 0, k, j, i);
          if (recon_method_ > 0) {
            iim2 = i0_(m, n, k - 2, j, i) / tet_c_(m, 0, 0, k - 2, j, i);
            iip1 = i0_(m, n, k + 1, j, i) / tet_c_(m, 0, 0, k + 1, j, i);
          }
          if (recon_method_ > 1) {
            iim3 = i0_(m, n, k - 3, j, i) / tet_c_(m, 0, 0, k - 3, j, i);
            iip2 = i0_(m, n, k + 2, j, i) / tet_c_(m, 0, 0, k + 2, j, i);
          }

          // reconstruct primitive intensity
          Real iiu, scr;
          switch (recon_method_) {
            case ReconstructionMethod::dc:
              if (n3 > 0.0)
                iiu = iim1;
              else
                iiu = iicc;
              break;
            case ReconstructionMethod::plm:
              if (n3 > 0.0)
                PLM(iim2, iim1, iicc, iiu, scr);
              else
                PLM(iim1, iicc, iip1, scr, iiu);
              break;
            case ReconstructionMethod::ppm4:
              if (n3 > 0.0)
                PPM4(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                PPM4(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::ppmx:
              if (n3 > 0.0)
                PPMX(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                PPMX(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::wenoz:
              if (n3 > 0.0)
                WENOZ(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                WENOZ(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            case ReconstructionMethod::teno:
              if (n3 > 0.0)
                TENO(iim3, iim2, iim1, iicc, iip1, iiu, scr);
              else
                TENO(iim2, iim1, iicc, iip1, iip2, scr, iiu);
              break;
            default:
              break;
          }

          // compute x3flux
          flx3(m, n, k, j, i) = n3 * iiu;
          if (corrected_flux && (gr_split || fabs(n3) > 1.e-30)) {
            int l = k - 1, r = k;
            Real dop_l = state(m, 1, l, j, i), dop_r = state(m, 1, r, j, i);
            for (int a = 1; a <= 3; ++a) {
              dop_l -= state(m, a + 1, l, j, i) * nh_c_.d_view(n, a);
              dop_r -= state(m, a + 1, r, j, i) * nh_c_.d_view(n, a);
            }
            Real tau = 2. * optical_depth_scale_ * block_size.d_view(m).dx3 *
                       (state(m, 0, l, j, i) * state(m, 7, l, j, i) * dop_l +
                        state(m, 0, r, j, i) * state(m, 7, r, j, i) * dop_r);
            Real left = iim1, right = iicc, scratch;
            if (recon_method_ == ReconstructionMethod::plm) {
              PLM(iim2, iim1, iicc, left, scratch);
              PLM(iim1, iicc, iip1, scratch, right);
            }
            if (gr_split) {
              const Real zeta = t3d3(m, 0, k, j, i);
              Real h2 = 0.;
              for (int a = 1; a <= 3; ++a) h2 += t3d3(m, a, k, j, i) * t3d3(m, a, k, j, i);
              const Real h = sqrt(h2);
              const Real vn = 0.5 * (state(m, 10, l, j, i) + state(m, 10, r, j, i)) / h;
              flx3(m, n, k, j, i) =
                  GRSpatialSplitFlux(zeta, h, (n3 - zeta) / h, vn, left, right, tau);
            } else {
              const Real velocity = 0.5 * (state(m, 4, l, j, i) / state(m, 1, l, j, i) +
                                           state(m, 4, r, j, i) / state(m, 1, r, j, i));
              flx3(m, n, k, j, i) = OpticalDepthSplitFlux(n3, left, right, tau, velocity);
            }
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // Angular Fluxes

  if (angular_fluxes) {
    auto &numn = prgeo->num_neighbors;
    auto &indn = prgeo->ind_neighbors;
    auto &arcl = prgeo->arc_lengths;
    auto &solid_angles_ = prgeo->solid_angles;

    auto &na_ = na;
    auto &divfa_ = divfa;

    par_for(
        "rflux_angular", DevExeSpace(), 0, nmb1, 0, nang1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
          divfa_(m, n, k, j, i) = 0.0;
          for (int nb = 0; nb < numn.d_view(n); ++nb) {
            Real flx_edge = na_(m, n, k, j, i, nb) *
                            ((na_(m, n, k, j, i, nb) < 0.0)
                                 ? i0_(m, indn.d_view(n, nb), k, j, i) / tet_c_(m, 0, 0, k, j, i)
                                 : i0_(m, n, k, j, i) / tet_c_(m, 0, 0, k, j, i));
            divfa_(m, n, k, j, i) += (arcl.d_view(n, nb) * flx_edge / solid_angles_.d_view(n));
          }
        });
  }

  return TaskStatus::complete;
}

}  // namespace radiation
