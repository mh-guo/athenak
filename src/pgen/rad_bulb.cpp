//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rad_bulb.cpp
//  \brief 3D Bulb tests for radiation

// C++ headers

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "radiation/radiation.hpp"
#include "pgen/pgen.hpp"

// Prototypes for user-defined BCs and history functions
void BulbSource(Mesh* pm, const Real bdt);

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::UserProblem(ParameterInput *pin)
//! \brief Sets initial conditions for GR radiation bulb test

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // capture variables for kernel
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);

  // set source function for bulb test
  user_srcs_func = BulbSource;

  // return if restart
  if (restart) return;

  if (pmbp->phydro != nullptr) {
    auto &w0 = pmbp->phydro->w0;
    par_for("rad_bulb_hyd",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = 1.0;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = 1.0;
    });
    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  auto &i0 = pmbp->prad->i0;
  par_for("rad_bulb",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    if (rad < 1.0) {
      i0(m,n,k,j,i) = -1.0/(4.0*M_PI);
    } else {
      i0(m,n,k,j,i) = 0.0;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void BulbSource(Mesh* pm, const Real bdt)
//! \brief User-defined source function for bulb test

void BulbSource(Mesh* pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is; int ie = indcs.ie;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb1 = (pmbp->nmb_thispack-1);
  int nang1 = (pmbp->prad->prgeo->nangles-1);

  if (pmbp->phydro != nullptr) {
    auto &w0 = pmbp->phydro->w0;
    par_for("rad_bulb_hyd",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      w0(m,IDN,k,j,i) = 1.0;
      w0(m,IVX,k,j,i) = 0.0;
      w0(m,IVY,k,j,i) = 0.0;
      w0(m,IVZ,k,j,i) = 0.0;
      w0(m,IEN,k,j,i) = 1.0;
    });
    // Convert primitives to conserved
    auto &u0 = pmbp->phydro->u0;
    pmbp->phydro->peos->PrimToCons(w0, u0, 0, (n1-1), 0, (n2-1), 0, (n3-1));
  }

  auto &i0 = pmbp->prad->i0;
  par_for("rad_bulb_source",DevExeSpace(),0,nmb1,0,nang1,0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    if (rad < 1.0) {
      i0(m,n,k,j,i) = -1.0/(4.0*M_PI);
    }
  });

  return;
}