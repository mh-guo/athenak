//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_ct.cpp
//  \brief

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "srcterms/srcterms.hpp"
#include "driver/driver.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CT
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//  To be clear, the edge-centered variable 'efld' stores E = -(v X B).
//  Temporal update uses multi-step SSP integrators, e.g. RK2, RK3

TaskStatus MHD::CT(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // capture class variables for the kernels
  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;

  int &ng = indcs.ng;
  bool &multi_zone = pmy_pack->pcoord->multi_zone;
  auto zone_mask = pmy_pack->pcoord->zone_mask;
  if (multi_zone) {
    int &nx1 = indcs.nx1, &nx2 = indcs.nx2, &nx3 = indcs.nx3;
    auto &size = pmy_pack->pmb->mb_size;
    Real zone_r = pmy_pack->pcoord->zone_r;
    Real zr_min = 0.88*zone_r;
    Real zr_max = 0.98*zone_r;
    Real zr_len = zr_max - zr_min;
    // fix E field
    par_for("fix-efld", DevExeSpace(), 0, nmb1, ks-ng, ke+ng, js-ng, je+ng ,is-ng, ie+ng,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      if (zone_mask(m,k,j,i)) {
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

        Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
        Real x = fmin(fmax((rad-zr_min)/zr_len, 0.0), 1.0);
        Real fac = SQR(sin(M_PI_2*x));
        e3(m,k,j,i) *= fac;
        e2(m,k,j,i) *= fac;
        e1(m,k,j,i) *= fac;
        if (k==ke+ng) {
          e2(m,k+1,j,i) *= fac;
          e2(m,k+1,j,i+1) *= fac;
          e1(m,k+1,j,i) *= fac;
          e1(m,k+1,j+1,i) *= fac;
        }
        if (j==je+ng) {
          e3(m,k,j+1,i) *= fac;
          e3(m,k,j+1,i+1) *= fac;
          e1(m,k,j+1,i) *= fac;
          e1(m,k+1,j+1,i) *= fac;
        }
        if (i==ie+ng) {
          e3(m,k,j,i+1) *= fac;
          e3(m,k,j+1,i+1) *= fac;
          e2(m,k,j,i+1) *= fac;
          e2(m,k+1,j,i+1) *= fac;
        }
      }
    });
  }

  //---- update B1 (only for 2D/3D problems)
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))/mbsize.d_view(m).dx2;
      if (three_d) {
        bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))/mbsize.d_view(m).dx3;
      }
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  auto bx2f = b0.x2f;
  auto bx2f_old = b1.x2f;
  par_for("CT-b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
    bx2f(m,k,j,i) += beta_dt*(e3(m,k,j,i+1) - e3(m,k,j,i))/mbsize.d_view(m).dx1;
    if (three_d) {
      bx2f(m,k,j,i) -= beta_dt*(e1(m,k+1,j,i) - e1(m,k,j,i))/mbsize.d_view(m).dx3;
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  auto bx3f = b0.x3f;
  auto bx3f_old = b1.x3f;
  par_for("CT-b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
    bx3f(m,k,j,i) -= beta_dt*(e2(m,k,j,i+1) - e2(m,k,j,i))/mbsize.d_view(m).dx1;
    if (multi_d) {
      bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))/mbsize.d_view(m).dx2;
    }
  });

  return TaskStatus::complete;
}
} // namespace mhd
