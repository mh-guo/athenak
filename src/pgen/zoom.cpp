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
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "pgen/zoom.hpp"

namespace zoom {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Zoom::Zoom(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    efld("efld",1,1,1,1) {
  is_set = pin->GetOrAddBoolean("zoom","is_set",false);
  restart = pin->GetOrAddBoolean("zoom","restart",true);
  fix_efield = pin->GetOrAddBoolean("zoom","fix_efield",false);
  r_in = pin->GetReal("zoom","r_in");
  // TODO(@mhguo): move to struct ZoomBC
  d_zoom = pin->GetOrAddReal("zoom","d_zoom",(FLT_MIN));
  p_zoom = pin->GetOrAddReal("zoom","p_zoom",(FLT_MIN));
  nlevels = pin->GetOrAddInteger("zoom","nlevels",4);
  mzoom = 8*nlevels;
  nvars = pin->GetOrAddInteger("zoom","nvars",5);
  zint.t_run_fac = pin->GetOrAddReal("zoom","t_run_fac",1.0);
  zint.t_run_pow = pin->GetOrAddReal("zoom","t_run_pow",0.0);
  zint.t_run_fac_zone_0 = pin->GetOrAddReal("zoom","t_run_fac_zone_0",zint.t_run_fac);
  zint.t_run_fac_zone_1 = pin->GetOrAddReal("zoom","t_run_fac_zone_1",zint.t_run_fac);
  zint.t_run_fac_zone_2 = pin->GetOrAddReal("zoom","t_run_fac_zone_2",zint.t_run_fac);
  zint.t_run_fac_zone_3 = pin->GetOrAddReal("zoom","t_run_fac_zone_3",zint.t_run_fac);
  zint.t_run_fac_zone_4 = pin->GetOrAddReal("zoom","t_run_fac_zone_4",zint.t_run_fac);
  zint.t_run_fac_zone_max = pin->GetOrAddReal("zoom","t_run_fac_zone_max",zint.t_run_fac);

  auto pmesh = pmy_pack->pmesh;
  zamr.max_level = pmesh->max_level;
  zamr.min_level = zamr.max_level - nlevels + 1;
  zamr.level = pin->GetOrAddInteger("zoom","level",zamr.max_level);
  zamr.zone = zamr.max_level - zamr.level;
  zamr.direction = pin->GetOrAddInteger("zoom","direction",-1);
  zamr.last_time = pmesh->time;
  SetInterval();
  zamr.just_zoomed = false;

  // allocate memory for primitive variables
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(u0, mzoom, nvars, ncells3, ncells2, ncells1);
  Kokkos::realloc(w0, mzoom, nvars, ncells3, ncells2, ncells1);
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(coarse_u0, mzoom, nvars, n_ccells3, n_ccells2, n_ccells1);
  Kokkos::realloc(coarse_w0, mzoom, nvars, n_ccells3, n_ccells2, n_ccells1);

  // allocate electric fields
  Kokkos::realloc(efld.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(efld.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(efld.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  Initialize();

  std::cout << "Zoom: initialized" << std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::Initialize()
//! \brief Initialize Zoom variables

// TODO(@mhguo): Implement this function
// Set density, velocity, pressure
void Zoom::Initialize()
{
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::PrintInfo()
//! \brief Print Zoom information

void Zoom::PrintInfo()
{
  // print level structure
  std::cout << "Zoom: max_level = " << zamr.max_level << " min_level = " << zamr.min_level
            << " level = " << zamr.level << " zone = " << zamr.zone
            << " direction = " << zamr.direction << std::endl;
  std::cout << "Zoom: runtime = " << zamr.runtime << " next time = " << zamr.next_time << std::endl;
  std::cout << "Zoom: t_run_fac = " << zint.t_run_fac << " t_run_pow = " << zint.t_run_pow
            << " t_run_fac_zone_0 = " << zint.t_run_fac_zone_0
            << " t_run_fac_zone_max = " << zint.t_run_fac_zone_max << std::endl;
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::BoundaryConditions()
//! \brief User-defined boundary conditions

// TODO(@mhguo): not including ghost cells now, should determine if we need
// TODO(@Mhguo): decide whehter we still need the sink condition
void Zoom::BoundaryConditions()
{
  if (!is_set) return;
  // put here because BoundaryConditions() is called just after
  // RedistAndRefineMeshBlocks() in AdaptiveMeshRefinement()
  if (zamr.just_zoomed && zamr.direction > 0 && zamr.level != zamr.min_level) {
    ApplyVariables();
    zamr.just_zoomed = false;
    std::cout << "Zoom: Apply variables after zooming" << std::endl;
  }
  if (zamr.zone == 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;

  auto &size = pmy_pack->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;

  // Select either Hydro or MHD
  Real gamma = 0.0;
  DvceArray5D<Real> u0_, w0_, bcc;
  bool is_mhd = (pmy_pack->pmhd != nullptr);
  if (pmy_pack->phydro != nullptr) {
    gamma = pmy_pack->phydro->peos->eos_data.gamma;
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    gamma = pmy_pack->pmhd->peos->eos_data.gamma;
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
    bcc = pmy_pack->pmhd->bcc0;
  }
  Real gm1 = gamma - 1.0;

  auto cu0 = coarse_u0;
  auto cw0 = coarse_w0;

  Real rin = this->r_in;
  Real dzoom = this->d_zoom;
  Real pzoom = this->p_zoom;
  Real rzoom = zamr.radius;
  int zid = 8*(zamr.zone-1);
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  // std::cout << "Zoom: BoundaryConditions: zone = " << zamr.zone << " radius = " << rzoom << std::endl;
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

    // vacuum boundary conditions
    // // apply initial conditions to boundary cells
    // if (rad < rin) {
    //   // store conserved quantities in 3D array
    //   u0_(m,IDN,k,j,i) = dzoom;
    //   u0_(m,IM1,k,j,i) = 0.0;
    //   u0_(m,IM2,k,j,i) = 0.0;
    //   u0_(m,IM3,k,j,i) = 0.0;
    //   u0_(m,IEN,k,j,i) = pzoom/gm1;
    // }
    if (rad < rzoom) {
      int hnx1 = nx1/2; int hnx2 = nx2/2;  int hnx3 = nx3/2;
      bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
      bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
      int leaf_id = 1*x1r + 2*x2r + 4*x3r;
      int zm = zid + leaf_id;
      int ci = i - hnx1 * x1l;
      int cj = j - hnx2 * x2l;
      int ck = k - hnx3 * x3l;
      if (is_mhd) {
        w0_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
        w0_(m,IM1,k,j,i) = cw0(zm,IM1,ck,cj,ci);
        w0_(m,IM2,k,j,i) = cw0(zm,IM2,ck,cj,ci);
        w0_(m,IM3,k,j,i) = cw0(zm,IM3,ck,cj,ci);
        w0_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);
        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

        // Load single state of primitive variables
        MHDPrim1D w;
        w.d  = w0_(m,IDN,k,j,i);
        w.vx = w0_(m,IVX,k,j,i);
        w.vy = w0_(m,IVY,k,j,i);
        w.vz = w0_(m,IVZ,k,j,i);
        w.e  = w0_(m,IEN,k,j,i);

        // load cell-centered fields into primitive state
        w.bx = bcc(m,IBX,k,j,i);
        w.by = bcc(m,IBY,k,j,i);
        w.bz = bcc(m,IBZ,k,j,i);

        // call p2c function
        HydCons1D u;
        SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

        // store conserved quantities in 3D array
        u0_(m,IDN,k,j,i) = u.d;
        u0_(m,IM1,k,j,i) = u.mx;
        u0_(m,IM2,k,j,i) = u.my;
        u0_(m,IM3,k,j,i) = u.mz;
        u0_(m,IEN,k,j,i) = u.e;
      } else {
        u0_(m,IDN,k,j,i) = cu0(zm,IDN,ck,cj,ci);
        u0_(m,IM1,k,j,i) = cu0(zm,IM1,ck,cj,ci);
        u0_(m,IM2,k,j,i) = cu0(zm,IM2,ck,cj,ci);
        u0_(m,IM3,k,j,i) = cu0(zm,IM3,ck,cj,ci);
        u0_(m,IEN,k,j,i) = cu0(zm,IEN,ck,cj,ci);
      }
    }
  });
  return;
}

void Zoom::AMR() {
  zamr.just_zoomed = false;
  if (pmy_pack->pmesh->time >= zamr.next_time) {
    std::cout << "Zoom AMR: old level = " << zamr.level << std::endl;
    if (zamr.direction < 0) {
      UpdateVariables();
    }
    RefineCondition();
    zamr.level += zamr.direction;
    zamr.direction = (zamr.level==zamr.max_level) ? -1 : zamr.direction;
    zamr.direction = (zamr.level==zamr.min_level) ? 1 : zamr.direction;
    zamr.zone = zamr.max_level - zamr.level;
    SetInterval();
    std::cout << "Zoom AMR: new level = " << zamr.level << " zone = " << zamr.zone << std::endl;
    std::cout << "Zoom AMR: runtime = " << zamr.runtime << " next time = " << zamr.next_time << std::endl;
    zamr.just_zoomed = true;
  }
}

void Zoom::SetInterval() {
  zamr.radius = std::pow(2.0,static_cast<Real>(zamr.zone));
  Real timescale = pow(zamr.radius,zint.t_run_pow);
  zamr.runtime = zint.t_run_fac*timescale;
  if (zamr.level==zamr.max_level) {zamr.runtime = zint.t_run_fac_zone_0*timescale;}
  if (zamr.level==zamr.min_level) {zamr.runtime = zint.t_run_fac_zone_max*timescale;}
  if (zamr.zone == 1) {zamr.runtime = zint.t_run_fac_zone_1*timescale;}
  if (zamr.zone == 2) {zamr.runtime = zint.t_run_fac_zone_2*timescale;}
  if (zamr.zone == 3) {zamr.runtime = zint.t_run_fac_zone_3*timescale;}
  if (zamr.zone == 4) {zamr.runtime = zint.t_run_fac_zone_4*timescale;}
  zamr.last_time = pmy_pack->pmesh->time;
  zamr.next_time = zamr.last_time + zamr.runtime;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::RefineCondition()
//! \brief User-defined refinement condition(s)

// TODO(@mhguo): Implement this function
void Zoom::RefineCondition() {
  auto &size = pmy_pack->pmb->mb_size;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto refine_flag_ = pmy_pack->pmesh->pmr->refine_flag;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];

  if (pmy_pack->pmesh->adaptive) {
    int old_level = zamr.level;
    int direction = zamr.direction;
    Real rin  = r_in;
    DualArray1D<int> levels_thisrank("levels_thisrank", nmb);
    for (int m=0; m<nmb; ++m) {
      levels_thisrank.h_view(m) = pmy_pack->pmesh->lloc_eachmb[m+mbs].level;
    }
    levels_thisrank.template modify<HostMemSpace>();
    levels_thisrank.template sync<DevExeSpace>();
    par_for_outer("RefineLevel",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      if (levels_thisrank.d_view(m+mbs) == old_level) {
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
        if (direction > 0) {
          if (rad_min < rin) {
            refine_flag_.d_view(m+mbs) = 1;
          }
        } else if (direction < 0) {
          refine_flag_.d_view(m+mbs) = -1;
        }
      }
    });
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::UpdateVariables()
//! \brief User-defined update of variables

void Zoom::UpdateVariables() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];
  DvceArray5D<Real> u, w;
  if (pmy_pack->phydro != nullptr) {
    u = pmy_pack->phydro->u0;
    w = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    u = pmy_pack->pmhd->u0;
    w = pmy_pack->pmhd->w0;
  }
  auto cu = coarse_u0, cw = coarse_w0;
  int zid = 8*zamr.zone;
  int nvar = nvars;
  Real rin = r_in;
  par_for("zoom-update",DevExeSpace(), 0,nmb-1, 0,nvar-1, cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
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

    if (rad_min < rin) {
      bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
      bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
      int leaf_id = 1*x1r + 2*x2r + 4*x3r;
      int zm = zid + leaf_id;
      int finei = 2*i - cis;  // correct when cis=is
      int finej = 2*j - cjs;  // correct when cjs=js
      int finek = 2*k - cks;  // correct when cks=ks
      cu(zm,n,k,j,i) =
          0.125*(u(m,n,finek  ,finej  ,finei) + u(m,n,finek  ,finej  ,finei+1)
              + u(m,n,finek  ,finej+1,finei) + u(m,n,finek  ,finej+1,finei+1)
              + u(m,n,finek+1,finej,  finei) + u(m,n,finek+1,finej,  finei+1)
              + u(m,n,finek+1,finej+1,finei) + u(m,n,finek+1,finej+1,finei+1));
      cw(zm,n,k,j,i) =
          0.125*(w(m,n,finek  ,finej  ,finei) + w(m,n,finek  ,finej  ,finei+1)
              + w(m,n,finek  ,finej+1,finei) + w(m,n,finek  ,finej+1,finei+1)
              + w(m,n,finek+1,finej,  finei) + w(m,n,finek+1,finej,  finei+1)
              + w(m,n,finek+1,finej+1,finei) + w(m,n,finek+1,finej+1,finei+1));
      // printf("m=%d, i=%d, j=%d, k=%d, finei=%d, finej=%d, finek=%d, den = %f\n",
      //         m, i, j, k, finei, finej, finek, w(m,IDN,finek,finej,finei));
    }
  });
  if (pmy_pack->pmhd != nullptr && fix_efield) {
    DvceEdgeFld4D<Real> emf = pmy_pack->pmhd->efld;
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto ef1 = emf.x1e;
    auto ef2 = emf.x2e;
    auto ef3 = emf.x3e;
    // update coarse electric fields
    par_for("zoom-update-efld",DevExeSpace(), 0,nmb-1, cks,cke+1, cjs,cje+1, cis,cie+1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

      if (rad_min < rin) {
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        int finei = 2*i - cis;  // correct when cis=is
        int finej = 2*j - cjs;  // correct when cjs=js
        int finek = 2*k - cks;  // correct when cks=ks
        e1(zm,k,j,i) = 0.5*(ef1(m,finek,finej,finei) + ef1(m,finek,finej,finei+1));
        e2(zm,k,j,i) = 0.5*(ef2(m,finek,finej,finei) + ef2(m,finek,finej+1,finei));
        e3(zm,k,j,i) = 0.5*(ef3(m,finek,finej,finei) + ef3(m,finek+1,finej,finei));
        
        Real &x1min = size.d_view(m).x1min;
        Real &x1max = size.d_view(m).x1max;
        Real x1v = CellCenterX(finei-cis, nx1, x1min, x1max);
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        Real x2v = CellCenterX(finej-cjs, nx2, x2min, x2max);
        Real &x3min = size.d_view(m).x3min;
        Real &x3max = size.d_view(m).x3max;
        Real x3v = CellCenterX(finek-cks, nx3, x3min, x3max);
        Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
        // print e, ef, rad, dens
        // printf("m=%d, i=%d, j=%d, k=%d, finei=%d, finej=%d, finek=%d, e1 = %f, e2 = %f, e3 = %f, ef1 = %f, ef2 = %f, ef3 = %f, rad = %f, den = %f\n",
        //        m, i, j, k, finei, finej, finek,
        //        e1(zm,k,j,i), e2(zm,k,j,i), e3(zm,k,j,i), ef1(m,finek,finej,finei),
        //        ef2(m,finek,finej,finei), ef3(m,finek,finej,finei), rad, w(m,IDN,finek,finej,finei));
      }
    });
  }
  for (int m=0; m<nmb; ++m) {
    if (pmy_pack->pmesh->lloc_eachmb[m+mbs].level == zamr.level) {
      Real &x1min = size.h_view(m).x1min;
      Real &x1max = size.h_view(m).x1max;
      Real &x2min = size.h_view(m).x2min;
      Real &x2max = size.h_view(m).x2max;
      Real &x3min = size.h_view(m).x3min;
      Real &x3max = size.h_view(m).x3max;
      Real ax1min = x1min*x1max>0.0? fmin(fabs(x1min), fabs(x1max)) : 0.0;
      Real ax2min = x2min*x2max>0.0? fmin(fabs(x2min), fabs(x2max)) : 0.0;
      Real ax3min = x3min*x3max>0.0? fmin(fabs(x3min), fabs(x3max)) : 0.0;
      Real rad_min = sqrt(SQR(ax1min)+SQR(ax2min)+SQR(ax3min));
      if (rad_min < r_in) {    
        auto des_slice = Kokkos::subview(u0, Kokkos::make_pair(zid,zid+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto src_slice = Kokkos::subview(u, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        des_slice = Kokkos::subview(w0, Kokkos::make_pair(zid,zid+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        src_slice = Kokkos::subview(w, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        zid += 1;
      }
    }
  }
  if (zid != 8*(zamr.zone+1)) {
    std::cerr << "Error: Zoom::UpdateVariables() failed: zid = " << zid <<
                 " zone = " << zamr.zone << " level = " << zamr.level << std::endl;
    std::exit(1);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::ApplyVariables()
//! \brief Apply finer level variables to coarser level

void Zoom::ApplyVariables() {
  auto &size = pmy_pack->pmb->mb_size;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];
  DvceArray5D<Real> u, w;
  if (pmy_pack->phydro != nullptr) {
    u = pmy_pack->phydro->u0;
    w = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    u = pmy_pack->pmhd->u0;
    w = pmy_pack->pmhd->w0;
  }
  auto cu = coarse_u0, cw = coarse_w0;
  int zid = 8*zamr.zone;
  for (int m=0; m<nmb; ++m) {
    if (pmy_pack->pmesh->lloc_eachmb[m+mbs].level == zamr.level) {
      Real &x1min = size.h_view(m).x1min;
      Real &x1max = size.h_view(m).x1max;
      Real &x2min = size.h_view(m).x2min;
      Real &x2max = size.h_view(m).x2max;
      Real &x3min = size.h_view(m).x3min;
      Real &x3max = size.h_view(m).x3max;
      Real ax1min = x1min*x1max>0.0? fmin(fabs(x1min), fabs(x1max)) : 0.0;
      Real ax2min = x2min*x2max>0.0? fmin(fabs(x2min), fabs(x2max)) : 0.0;
      Real ax3min = x3min*x3max>0.0? fmin(fabs(x3min), fabs(x3max)) : 0.0;
      Real rad_min = sqrt(SQR(ax1min)+SQR(ax2min)+SQR(ax3min));
      if (rad_min < r_in) {    
        auto src_slice = Kokkos::subview(u0, Kokkos::make_pair(zid,zid+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto des_slice = Kokkos::subview(u, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        src_slice = Kokkos::subview(w0, Kokkos::make_pair(zid,zid+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        des_slice = Kokkos::subview(w, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        zid += 1;
      }
    }
  }
  if (zid != 8*(zamr.zone+1)) {
    std::cerr << "Error: Zoom::ApplyVariables() failed: zid = " << zid <<
                 " zone = " << zamr.zone << " level = " << zamr.level << std::endl;
    std::exit(1);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::FixEField()
//! \brief Fix E field on the zoomed grid

void Zoom::FixEField(DvceEdgeFld4D<Real> emf) {
  ApplyEField(emf);
  return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack-1;
  auto e1 = emf.x1e;
  auto e2 = emf.x2e;
  auto e3 = emf.x3e;
  Real rzoom = zamr.radius;

  // print basic information
  // std::cout << "Zoom: FixEField" << std::endl;

  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  // array_sum::GlobalSum tnc1, tnc2, tnc3, tem1, tem2, tem3;
  array_sum::GlobalSum nc1, nc2, nc3, em1, em2, em3;

  Kokkos::parallel_reduce("EFldSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, 
  array_sum::GlobalSum &nc1_, array_sum::GlobalSum &nc2_, array_sum::GlobalSum &nc3_, 
  array_sum::GlobalSum &em1_, array_sum::GlobalSum &em2_, array_sum::GlobalSum &em3_) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

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
    if (rad < rzoom) {
      int idx1 = (int) (8.0*(1.0+x1v/rzoom));
      int idx2 = (int) (8.0*(1.0+x2v/rzoom));
      int idx3 = (int) (8.0*(1.0+x3v/rzoom));
      // print postion and idx for debugging using printf
      // printf("m = %d, k = %d, j = %d, i = %d, idx1 = %d, idx2 = %d, idx3 = %d\n", m, k, j, i, idx1, idx2, idx3);

      nc1_.the_array[idx1] += 1.0;
      nc2_.the_array[idx2] += 1.0;
      nc3_.the_array[idx3] += 1.0;
      em1_.the_array[idx1] += e1(m,k,j,i);
      em2_.the_array[idx2] += e2(m,k,j,i);
      em3_.the_array[idx3] += e3(m,k,j,i);
    }

  }, Kokkos::Sum<array_sum::GlobalSum>(nc1),
     Kokkos::Sum<array_sum::GlobalSum>(nc2),
     Kokkos::Sum<array_sum::GlobalSum>(nc3),
     Kokkos::Sum<array_sum::GlobalSum>(em1),
     Kokkos::Sum<array_sum::GlobalSum>(em2),
     Kokkos::Sum<array_sum::GlobalSum>(em3));

  // Normalize
  par_for("emf-norm", DevExeSpace(), 0, NREDUCTION_VARIABLES-1,
  KOKKOS_LAMBDA(int i) {
    printf("idx: i = %d, nc1 = %f, nc2 = %f, nc3 = %f\n", i, nc1.the_array[i], nc2.the_array[i], nc3.the_array[i]);
    printf("sum: i = %d, em1 = %f, em2 = %f, em3 = %f\n", i, em1.the_array[i], em2.the_array[i], em3.the_array[i]);
    printf("mean i = %d, em1 = %f, em2 = %f, em3 = %f\n", i, em1.the_array[i]/nc1.the_array[i], em2.the_array[i]/nc2.the_array[i], em3.the_array[i]/nc3.the_array[i]);
  });

  par_for("fix-emf", DevExeSpace(), 0, nmb1, ks-ng, ke+ng, js-ng, je+ng ,is-ng, ie+ng,
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
    if (rad < rzoom) {
      int idx1 = (int) (8.0*(1.0+x1v/rzoom));
      int idx2 = (int) (8.0*(1.0+x2v/rzoom));
      int idx3 = (int) (8.0*(1.0+x3v/rzoom));
      Real e1m = em1.the_array[idx1]/nc1.the_array[idx1];
      Real e2m = em2.the_array[idx2]/nc2.the_array[idx2];
      Real e3m = em3.the_array[idx3]/nc3.the_array[idx3];
      e1(m,k,j,i) = e1m;
      e2(m,k,j,i) = e2m;
      e3(m,k,j,i) = e3m;
      e1(m,k+1,j,i) = e1m;
      e1(m,k,j+1,i) = e1m;
      e1(m,k+1,j+1,i) = e1m;
      e2(m,k+1,j,i) = e2m;
      e2(m,k,j,i+1) = e2m;
      e2(m,k+1,j,i+1) = e2m;
      e3(m,k,j+1,i) = e3m;
      e3(m,k,j,i+1) = e3m;
      e3(m,k,j+1,i+1) = e3m;
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::ApplyEField()
//! \brief Fix E field on the zoomed grid

void Zoom::ApplyEField(DvceEdgeFld4D<Real> emf) {
  if (zamr.zone == 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb1 = pmy_pack->nmb_thispack-1;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  Real rzoom = zamr.radius;

  int zid = 8*(zamr.zone-1);
  par_for("fix-emf", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1 ,is-1, ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real x1f = LeftEdgeX  (i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
    Real x2f = LeftEdgeX  (j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
    Real x3f = LeftEdgeX  (k-ks, nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
    
    int hnx1 = nx1/2; int hnx2 = nx2/2;  int hnx3 = nx3/2;
    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    // TODO: check if this is correct
    int ci = i - hnx1 * x1l;
    int cj = j - hnx2 * x2l;
    int ck = k - hnx3 * x3l;
    Real fac = 1.0; //(rzoom-rad)/rzoom;
    // ef2(m,k,j,i) += fac*e2(zm,ck,cj,ci);
    // ef3(m,k,j,i) += fac*e3(zm,ck,cj,ci);
    if (sqrt(SQR(x1v)+SQR(x2f)+SQR(x3f)) < rzoom) {
      ef1(m,k,j,i) += fac*e1(zm,ck,cj,ci);
    }
    if (sqrt(SQR(x1f)+SQR(x2v)+SQR(x3f)) < rzoom) {
      ef2(m,k,j,i) += fac*e2(zm,ck,cj,ci);
    }
    if (sqrt(SQR(x1f)+SQR(x2f)+SQR(x3v)) < rzoom) {
      ef3(m,k,j,i) += fac*e3(zm,ck,cj,ci);
    }
  });

  return;
}
} // namespace zoom
