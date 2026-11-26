//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_update.cpp
//  \brief Performs update of Radiation conserved variables (i0) for each stage of
//   explicit SSP RK integrators (e.g. RK1, RK2, RK3). Update uses weighted average and
//   partial time step appropriate to stage.
//  Explicit (not implicit) radiation source terms are included in this update.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "srcterms/srcterms.hpp"
#include "radiation.hpp"

namespace radiation {
//----------------------------------------------------------------------------------------
//! \fn  void Radiation::RKUpdate
//  \brief Explicit RK update of flux divergence and physical source terms

TaskStatus Radiation::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nang1 = prgeo->nangles - 1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &mbsize  = pmy_pack->pmb->mb_size;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);

  auto &i0_ = i0;
  auto &i1_ = i1;
  auto &flx1 = iflx.x1f;
  auto &flx2 = iflx.x2f;
  auto &flx3 = iflx.x3f;

  auto &nh_c_ = nh_c;
  auto &tt = tet_c;
  auto &tc = tetcov_c;

  auto &angular_fluxes_ = angular_fluxes;
  auto &divfa_ = divfa;

  auto &excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &rad_mask_ = pmy_pack->pcoord->excision_floor;
  Real &n_0_floor_ = n_0_floor;

  auto floor_stats = transport_floor_stats;
  constexpr int nbins = 4096;
  Kokkos::View<Real*[4], DevExeSpace> partial("floor_partial", nbins);
  auto omega = prgeo->solid_angles;
  par_for("r_update",DevExeSpace(),0,nmb1,0,nang1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    // spatial fluxes
    Real divf_s = (flx1(m,n,k,j,i+1) - flx1(m,n,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      divf_s += (flx2(m,n,k,j+1,i) - flx2(m,n,k,j,i))/mbsize.d_view(m).dx2;
    }
    if (three_d) {
      divf_s += (flx3(m,n,k+1,j,i) - flx3(m,n,k,j,i))/mbsize.d_view(m).dx3;
    }
    i0_(m,n,k,j,i) = gam0*i0_(m,n,k,j,i)+gam1*i1_(m,n,k,j,i)-beta_dt*divf_s;

    // angular fluxes
    if (angular_fluxes_) { i0_(m,n,k,j,i) -= beta_dt*divfa_(m,n,k,j,i); }

    // zero intensity if negative
    Real n0  = tt(m,0,0,k,j,i);
    Real n_0 = tc(m,0,0,k,j,i)*nh_c_.d_view(n,0) + tc(m,1,0,k,j,i)*nh_c_.d_view(n,1) +
               tc(m,2,0,k,j,i)*nh_c_.d_view(n,2) + tc(m,3,0,k,j,i)*nh_c_.d_view(n,3);
    Real intensity = i0_(m,n,k,j,i)/(n0*n_0);
    if (!isfinite(intensity))
      Kokkos::abort("Nonfinite radiation intensity before transport floor");
    if (intensity < 0.0) {
      const int bin = (i + 37*j + 101*k + 251*n + 509*m) & (nbins-1);
      Kokkos::atomic_add(&partial(bin,0), 1.0);
      Kokkos::atomic_add(&partial(bin,1), -intensity);
      // Angle/coordinate-volume-weighted stage correction. In Minkowski this
      // is an energy increment, but the sum is NOT a net RK energy budget.
      Real dv=mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
      Real de=-intensity*omega.d_view(n)*dv;
      Kokkos::atomic_add(&partial(bin,2), de);
      Kokkos::atomic_add(&partial(bin,3), de*nh_c_.d_view(n,1));
    }
    i0_(m,n,k,j,i) = n0*n_0*fmax(intensity, 0.0);

    // handle excision
    // NOTE(@pdmullen): excision criterion are not finalized.  The below zeroes all
    // intensities within rks <= 1.0 and zeroes intensities within angles where n_0
    // is about zero.  This needs future attention.
    if (excise) {
      if (rad_mask_(m,k,j,i) || fabs(n_0) < n_0_floor_) { i0_(m,n,k,j,i) = 0.0; }
    }
  });
  Kokkos::parallel_for("floor_stats_merge", Kokkos::RangePolicy<DevExeSpace>(0,4),
    KOKKOS_LAMBDA(const int q) {
      Real sum = 0;
      for (int b=0; b<nbins; ++b) sum += partial(b,q);
      floor_stats(q) += sum;
    });
  return TaskStatus::complete;
}
} // namespace radiation
