//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom.cpp
//  \brief implementation of constructor and functions in Zoom class

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "pgen/zoom.hpp"

namespace zoom {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Zoom::Zoom(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  is_set = pin->GetOrAddBoolean("zoom","is_set",false);
  r_in = pin->GetReal("zoom","r_in");
  d_zoom = pin->GetReal("zoom","d_zoom");
  p_zoom = pin->GetReal("zoom","p_zoom");
}

void Zoom::ZoomBoundaryConditions()
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;

  auto &size = pmy_pack->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;

  // Select either Hydro or MHD
  Real gamma = 0.0;
  DvceArray5D<Real> u0_, w0_;
  if (pmy_pack->phydro != nullptr) {
    gamma = pmy_pack->phydro->peos->eos_data.gamma;
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    gamma = pmy_pack->pmhd->peos->eos_data.gamma;
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }
  Real gm1 = gamma - 1.0;

  Real rin = this->r_in;
  Real dzoom = this->d_zoom;
  Real pzoom = this->p_zoom;
  par_for("fixed_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
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

    // apply initial conditions to boundary cells
    if (rad < rin) {
      // store conserved quantities in 3D array
      u0_(m,IDN,k,j,i) = dzoom;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) = 0.0;
      u0_(m,IEN,k,j,i) = pzoom/gm1;
    }
  });
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::ZoomRefine()
//! \brief User-defined refinement condition(s)

// TODO(@mhguo): Implement this function
void Zoom::ZoomRefine() {
  auto &size = pmy_pack->pmb->mb_size;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto refine_flag_ = pmy_pack->pmesh->pmr->refine_flag;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];

  // TODO(@mhguo): write this as a parameter
  int target_level = 20;

  if (pmy_pack->pmesh->adaptive) {
    int root_level = pmy_pack->pmesh->root_level;
    int ncycle=pmy_pack->pmesh->ncycle;
    int old_level = 0;
    int new_level = target_level;
    Real rin  = this->r_in;
    DualArray1D<int> levels_thisrank("levels_thisrank", nmb);
    for (int m=0; m<nmb; ++m) {
      levels_thisrank.h_view(m) = pmy_pack->pmesh->lloc_eachmb[m+mbs].level;
    }
    levels_thisrank.template modify<HostMemSpace>();
    levels_thisrank.template sync<DevExeSpace>();
    par_for_outer("RefineLevel",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real ax1min = x1min*x1max>0.0? fmin(fabs(x1min), fabs(x1max)) : 0.0;
      Real ax2min = x2min*x2max>0.0? fmin(fabs(x2min), fabs(x2max)) : 0.0;
      Real ax3min = x3min*x3max>0.0? fmin(fabs(x3min), fabs(x3max)) : 0.0;
      Real rad_min = sqrt(SQR(ax1min)+SQR(ax2min)+SQR(ax3min));
      if (levels_thisrank.d_view(m+mbs) > old_level+root_level) {
        if (new_level > old_level) {
          if (rad_min < rin) {
            refine_flag_.d_view(m+mbs) = 1;
          }
        } else if (new_level < old_level) {
          refine_flag_.d_view(m+mbs) = -1;
        }
      }
    });
  }
  return;
}

} // namespace zoom
