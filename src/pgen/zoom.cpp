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
#include "eos/ideal_c2p_hyd.hpp"
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
    ndiag(-1),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    efld("efld",1,1,1,1),
    emf0("emf0",1,1,1,1),
    delta_efld("delta_efld",1,1,1,1),
    max_emf0("max_emf0",1,1),
    zoom_fluxes("zoom_fluxes",1,1,1)
  {
  is_set = pin->GetOrAddBoolean("zoom","is_set",false);
  read_rst = pin->GetOrAddBoolean("zoom","read_rst",true);
  write_rst = pin->GetOrAddBoolean("zoom","write_rst",true);
  zoom_bcs = pin->GetOrAddBoolean("zoom","zoom_bcs",true);
  zoom_ref = pin->GetOrAddBoolean("zoom","zoom_ref",true);
  zoom_dt = pin->GetOrAddBoolean("zoom","zoom_dt",false);
  fix_efield = pin->GetOrAddBoolean("zoom","fix_efield",false);
  dump_diag  = pin->GetOrAddBoolean("zoom","dump_diag",false);
  ndiag = pin->GetOrAddInteger("zoom","ndiag",-1);
  calc_cons_change = pin->GetOrAddBoolean("zoom","calc_cons_change",false);
  zint.t_run_fac = pin->GetOrAddReal("zoom","t_run_fac",1.0);
  zint.t_run_pow = pin->GetOrAddReal("zoom","t_run_pow",0.0);
  zint.t_run_max = pin->GetOrAddReal("zoom","t_run_max",FLT_MAX);
  zint.t_run_fac_zone_0 = pin->GetOrAddReal("zoom","t_run_fac_zone_0",zint.t_run_fac);
  zint.t_run_fac_zone_1 = pin->GetOrAddReal("zoom","t_run_fac_zone_1",zint.t_run_fac);
  zint.t_run_fac_zone_2 = pin->GetOrAddReal("zoom","t_run_fac_zone_2",zint.t_run_fac);
  zint.t_run_fac_zone_3 = pin->GetOrAddReal("zoom","t_run_fac_zone_3",zint.t_run_fac);
  zint.t_run_fac_zone_4 = pin->GetOrAddReal("zoom","t_run_fac_zone_4",zint.t_run_fac);
  zint.t_run_fac_zone_5 = pin->GetOrAddReal("zoom","t_run_fac_zone_5",zint.t_run_fac);
  zint.t_run_fac_zone_6 = pin->GetOrAddReal("zoom","t_run_fac_zone_6",zint.t_run_fac);
  zint.t_run_fac_zone_max = pin->GetOrAddReal("zoom","t_run_fac_zone_max",zint.t_run_fac);

  // TODO(@mhguo): may set the parameters so that the initial level equals the max level
  // TODO(@mhguo): currently we need to check whether zamr.level is correct by hand
  auto pmesh = pmy_pack->pmesh;
  zamr.nlevels = pin->GetOrAddInteger("zoom","nlevels",4);
  zamr.max_level = pmesh->max_level;
  zamr.min_level = zamr.max_level - zamr.nlevels + 1;
  zamr.level = pin->GetOrAddInteger("zoom","level",zamr.max_level);
  zamr.zone = zamr.max_level - zamr.level;
  zamr.direction = pin->GetOrAddInteger("zoom","direction",-1);
  zamr.just_zoomed = false;
  zamr.first_emf = false;
  zamr.dump_rst = true;
  zchg.dvol = 0.0;
  zchg.dmass = 0.0;
  zchg.dengy = 0.0;
  zrun.id = 0;
  zrun.next_time = 0.0;
  SetInterval();
  
  nleaf = 2;
  if (pmesh->two_d) nleaf = 4;
  if (pmesh->three_d) nleaf = 8;
  mzoom = nleaf*zamr.nlevels;
  nvars = pin->GetOrAddInteger("zoom","nvars",5);
  // TODO(@mhguo): move to a new struct?
  r_in = pin->GetReal("zoom","r_in");
  d_zoom = pin->GetOrAddReal("zoom","d_zoom",(FLT_MIN));
  p_zoom = pin->GetOrAddReal("zoom","p_zoom",(FLT_MIN));
  nflux = 3;
  emf_flag = 0;
  emf_f0 = 1.0;
  emf_f1 = 0.0;
  emf_fmax = 1.0;
  if (fix_efield) {
    emf_flag = pin->GetInteger("zoom","emf_flag");
    emf_f1 = pin->GetReal("zoom","emf_f1");
    emf_fmax = pin->GetReal("zoom","emf_fmax");
  }

  // allocate memory for primitive variables
  auto &indcs = pmesh->mb_indcs;
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

  // allocate electric fields just after zoom
  Kokkos::realloc(emf0.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(emf0.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(emf0.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  // allocate delta electric fields
  Kokkos::realloc(delta_efld.x1e, mzoom, n_ccells3+1, n_ccells2+1, n_ccells1);
  Kokkos::realloc(delta_efld.x2e, mzoom, n_ccells3+1, n_ccells2, n_ccells1+1);
  Kokkos::realloc(delta_efld.x3e, mzoom, n_ccells3, n_ccells2+1, n_ccells1+1);

  Kokkos::realloc(max_emf0, mzoom, 3);
  for (int i = 0; i < mzoom; i++) {
    for (int j = 0; j < 3; j++) {
      max_emf0(i,j) = 0.0;
    }
  }

  Kokkos::realloc(zoom_fluxes, 2, zamr.nlevels, nflux);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < zamr.nlevels; j++) {
      for (int k = 0; k < nflux; k++) {
        zoom_fluxes(i,j,k) = 0.0;
      }
    }
  }
  // only do this when needed since it is slow
  if (emf_flag >= 3) {
    // construct spherical grids
    int anglevel = 10;
    int ninterp = 1;
    auto &grids = spherical_grids;
    for (int i = 0; i < zamr.nlevels; i++) {
      Real rzoom = 0.9*static_cast<Real>(1<<i); // 2^i*0.9 to avoid active zone
      grids.push_back(std::make_unique<SphericalGrid>(pmy_pack, anglevel, rzoom, ninterp));
    }
  }

  Initialize();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::Initialize()
//! \brief Initialize Zoom variables

void Zoom::Initialize()
{
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 0;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 0;
  int nc1 = indcs.cnx1 + 2*ng;
  int nc2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*ng) : 0;
  int nc3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*ng) : 0;

  auto &u0_ = u0;
  auto &w0_ = w0;
  auto &cu0 = coarse_u0;
  auto &cw0 = coarse_w0;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e01 = emf0.x1e;
  auto e02 = emf0.x2e;
  auto e03 = emf0.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  bool is_mhd = (pmy_pack->pmhd != nullptr);
  auto peos = (is_mhd)? pmy_pack->pmhd->peos : pmy_pack->phydro->peos;
  Real gm1 = peos->eos_data.gamma - 1.0;

  Real dzoom = this->d_zoom;
  Real pzoom = this->p_zoom;

  par_for("zoom_init", DevExeSpace(),0,mzoom-1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    w0_(m,IDN,k,j,i) = dzoom;
    w0_(m,IM1,k,j,i) = 0.0;
    w0_(m,IM2,k,j,i) = 0.0;
    w0_(m,IM3,k,j,i) = 0.0;
    w0_(m,IEN,k,j,i) = pzoom/gm1;
  });

  par_for("zoom_init_c",DevExeSpace(),0,mzoom-1,0,nc3-1,0,nc2-1,0,nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    cw0(m,IDN,k,j,i) = dzoom;
    cw0(m,IM1,k,j,i) = 0.0;
    cw0(m,IM2,k,j,i) = 0.0;
    cw0(m,IM3,k,j,i) = 0.0;
    cw0(m,IEN,k,j,i) = pzoom/gm1;
  });

  // In MHD, we don't use conserved variables so no need to convert
  if (!is_mhd) {
    peos->PrimToCons(w0_,u0_,0,n3-1,0,n2-1,0,n1-1);
    peos->PrimToCons(cw0,cu0,0,nc3-1,0,nc2-1,0,nc1-1);
  }

  par_for("zoom_init_e1",DevExeSpace(),0,mzoom-1,0,nc3,0,nc2,0,nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e1(m,k,j,i) = 0.0;
    e01(m,k,j,i) = 0.0;
    de1(m,k,j,i) = 0.0;
  });
  par_for("zoom_init_e2",DevExeSpace(),0,mzoom-1,0,nc3,0,nc2-1,0,nc1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e2(m,k,j,i) = 0.0;
    e02(m,k,j,i) = 0.0;
    de2(m,k,j,i) = 0.0;
  });
  par_for("zoom_init_e3",DevExeSpace(),0,mzoom-1,0,nc3-1,0,nc2,0,nc1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    e3(m,k,j,i) = 0.0;
    e03(m,k,j,i) = 0.0;
    de3(m,k,j,i) = 0.0;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::PrintInfo()
//! \brief Print Zoom information

void Zoom::PrintInfo()
{
  if (global_variable::my_rank == 0) {
    std::cout << "============== Zoom Information ==============" << std::endl;
    // print basic parameters
    std::cout << "Basic: is_set = " << is_set << " read_rst = " << read_rst
              << " write_rst = " << write_rst << " ndiag = " << ndiag << std::endl;
    std::cout << "Funcs: zoom_bcs = " << zoom_bcs << " zoom_ref = " << zoom_ref 
              << " zoom_dt = " << zoom_dt << " fix_efield = " << fix_efield
              << " emf_flag = " << emf_flag << std::endl;
    // print model parameters
    std::cout << "Model: mzoom = " << mzoom << " nvars = " << nvars
              << " r_in = " << r_in << " d_zoom = " << d_zoom
              << " p_zoom = " << p_zoom << " emf_f0 = " << emf_f0
              << " emf_f1 = " << emf_f1 << std::endl;
    // print interval parameters
    std::cout << "Interval: t_run_fac = " << zint.t_run_fac
              << " t_run_pow = " << zint.t_run_pow
              << " t_run_max = " << zint.t_run_max
              << std::endl;
    std::cout << " t_run_fac_zone_0 = " << zint.t_run_fac_zone_0
              << " t_run_fac_zone_max = " << zint.t_run_fac_zone_max << std::endl;
    std::cout << " tfz_1 = " << zint.t_run_fac_zone_1
              << " tfz_2 = " << zint.t_run_fac_zone_2
              << " tfz_3 = " << zint.t_run_fac_zone_3
              << " tfz_4 = " << zint.t_run_fac_zone_4
              << " tfz_5 = " << zint.t_run_fac_zone_5
              << " tfz_6 = " << zint.t_run_fac_zone_6 << std::endl;
    // print level structure
    std::cout << "Level: max_level = " << zamr.max_level
              << " min_level = " << zamr.min_level
              << " level = " << zamr.level << " zone = " << zamr.zone
              << " direction = " << zamr.direction << std::endl;
    // print runtime information
    std::cout << "Time: runtime = " << zamr.runtime << " next time = "
              << zrun.next_time << std::endl;
    // throw warning
    if (zamr.zone != 0) {
      std::cout << "### WARNING! in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Zoom zone is not zero, this is not expected" << std::endl;
    }
    std::cout << "==============================================" << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::BoundaryConditions()
//! \brief User-defined boundary conditions

void Zoom::BoundaryConditions()
{
  if (!zoom_bcs) return;
  // put here because BoundaryConditions() is called in InitBoundaryValuesAndPrimitives(),
  // just after RedistAndRefineMeshBlocks() in AdaptiveMeshRefinement()
  if (zamr.just_zoomed && zamr.direction > 0 && zamr.level != zamr.min_level) {
    ApplyVariables();
    zamr.just_zoomed = false;
    if (global_variable::my_rank == 0) {
      std::cout << "Zoom: Apply variables after zooming" << std::endl;
    }
  }
  zchg.dvol = 0.0; zchg.dmass = 0.0; zchg.dengy = 0.0;
  if (zamr.zone == 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;

  auto &size = pmy_pack->pmb->mb_size;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb = pmy_pack->nmb_thispack;
  bool is_gr = pmy_pack->pcoord->is_general_relativistic;

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

  Real rzoom = zamr.radius;
  int zid = nleaf*(zamr.zone-1);
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  // calculate change in conserved variables
  if (calc_cons_change) {
    const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
    const int nkji = nx3*nx2*nx1;
    const int nji  = nx2*nx1;
    Real dvol = 0.0, mass = 0.0, engy = 0.0, dmass = 0.0, dengy = 0.0;
    Kokkos::parallel_reduce("cons_change",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &dv, Real &m0, Real &e0, Real &dm, Real &de) {
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
        Real dx1 = size.d_view(m).dx1;
        Real dx2 = size.d_view(m).dx2;
        Real dx3 = size.d_view(m).dx3;
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        int ci = i - cnx1 * x1l;
        int cj = j - cnx2 * x2l;
        int ck = k - cnx3 * x3l;
        if (is_mhd) {
          w0_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
          w0_(m,IM1,k,j,i) = cw0(zm,IM1,ck,cj,ci);
          w0_(m,IM2,k,j,i) = cw0(zm,IM2,ck,cj,ci);
          w0_(m,IM3,k,j,i) = cw0(zm,IM3,ck,cj,ci);
          w0_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);

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
          if (is_gr) {
            Real glower[4][4], gupper[4][4];
            ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
            SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
          } else {
            SingleP2C_IdealMHD(w, u);
          }

          // store conserved quantities in 3D array
          // u0_(m,IDN,k,j,i) = u.d;
          // u0_(m,IM1,k,j,i) = u.mx;
          // u0_(m,IM2,k,j,i) = u.my;
          // u0_(m,IM3,k,j,i) = u.mz;
          // u0_(m,IEN,k,j,i) = u.e;
          dv += dx1*dx2*dx3;
          m0 += dx1*dx2*dx3*u.d;
          e0 += dx1*dx2*dx3*u.e;
          dm += dx1*dx2*dx3*(u.d-u0_(m,IDN,k,j,i));
          de += dx1*dx2*dx3*(u.e-u0_(m,IEN,k,j,i));
        } else {
          // u0_(m,IDN,k,j,i) = cu0(zm,IDN,ck,cj,ci);
          // u0_(m,IM1,k,j,i) = cu0(zm,IM1,ck,cj,ci);
          // u0_(m,IM2,k,j,i) = cu0(zm,IM2,ck,cj,ci);
          // u0_(m,IM3,k,j,i) = cu0(zm,IM3,ck,cj,ci);
          // u0_(m,IEN,k,j,i) = cu0(zm,IEN,ck,cj,ci);
          dv += dx1*dx2*dx3;
          m0 += dx1*dx2*dx3*cu0(zm,IDN,ck,cj,ci);
          e0 += dx1*dx2*dx3*cu0(zm,IEN,ck,cj,ci);
          dm += dx1*dx2*dx3*(cu0(zm,IDN,ck,cj,ci)-u0_(m,IDN,k,j,i));
          de += dx1*dx2*dx3*(cu0(zm,IEN,ck,cj,ci)-u0_(m,IEN,k,j,i));
        }
      }
    }, Kokkos::Sum<Real>(dvol), Kokkos::Sum<Real>(mass), Kokkos::Sum<Real>(engy),
       Kokkos::Sum<Real>(dmass), Kokkos::Sum<Real>(dengy));
#if MPI_PARALLEL_ENABLED
    Real m_sum[5] = {dvol,mass,engy,dmass,dengy};
    Real gm_sum[5];
    MPI_Allreduce(m_sum, gm_sum, 5, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    dvol = gm_sum[0]; mass = gm_sum[1]; engy = gm_sum[2]; dmass = gm_sum[3]; dengy = gm_sum[4];
#endif
    zchg.dvol = dvol;
    zchg.dmass = dmass;
    zchg.dengy = dengy;
    // if (global_variable::my_rank == 0) {
    //   std::cout << "Zoom: Conserved quantities change: dvol = " << dvol
    //             << " mass = " << mass << " engy = " << engy
    //             << " dmass = " << dmass << " dengy = " << dengy << std::endl;
    // }
  }
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

    if (rad < rzoom) {
      bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
      bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
      int leaf_id = 1*x1r + 2*x2r + 4*x3r;
      int zm = zid + leaf_id;
      int ci = i - cnx1 * x1l;
      int cj = j - cnx2 * x2l;
      int ck = k - cnx3 * x3l;
      if (is_mhd) {
        w0_(m,IDN,k,j,i) = cw0(zm,IDN,ck,cj,ci);
        w0_(m,IM1,k,j,i) = cw0(zm,IM1,ck,cj,ci);
        w0_(m,IM2,k,j,i) = cw0(zm,IM2,ck,cj,ci);
        w0_(m,IM3,k,j,i) = cw0(zm,IM3,ck,cj,ci);
        w0_(m,IEN,k,j,i) = cw0(zm,IEN,ck,cj,ci);

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
        if (is_gr) {
          Real glower[4][4], gupper[4][4];
          ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
          SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);
        } else {
          SingleP2C_IdealMHD(w, u);
        }

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

//----------------------------------------------------------------------------------------
//! \fn void Zoom::AMR()
//! \brief Main function for Zoom Adaptive Mesh Refinement

void Zoom::AMR() {
  if (!zoom_ref) return;
  zamr.just_zoomed = false;
  zamr.dump_rst = (zamr.zone == 0);
  if (pmy_pack->pmesh->time >= zrun.next_time) {
    if (global_variable::my_rank == 0) {
      std::cout << "Zoom AMR: old level = " << zamr.level << std::endl;
    }
    if (zoom_bcs && zamr.direction < 0) {
      UpdateVariables();
      // TODO(@mhguo): only store the data needed on this rank instead of holding all
      SyncVariables();
      UpdateGhostVariables();
    }
    if (fix_efield && zamr.direction < 0) {
      if (emf_flag >=1) {
        zamr.first_emf = true;
      }
      if (emf_flag >= 3) {
        SphericalFlux(0,zamr.zone+1); // store fluxes
      }
    }
    RefineCondition();
    zamr.level += zamr.direction;
    zamr.direction = (zamr.level==zamr.max_level) ? -1 : zamr.direction;
    zamr.direction = (zamr.level==zamr.min_level) ? 1 : zamr.direction;
    zamr.zone = zamr.max_level - zamr.level;
    SetInterval();
    if (global_variable::my_rank == 0) {
      std::cout << "Zoom AMR: new level = " << zamr.level
                << " zone = " << zamr.zone << std::endl;
      std::cout << "Zoom AMR: runtime = " << zamr.runtime 
                << " next time = " << zrun.next_time << std::endl;
    }
    zamr.just_zoomed = true;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::SetInterval()
//! \brief Set the time interval for the next zoom

void Zoom::SetInterval() {
  zamr.radius = std::pow(2.0,static_cast<Real>(zamr.zone));
  Real timescale = pow(zamr.radius,zint.t_run_pow);
  zamr.runtime = zint.t_run_fac*timescale;
  if (zamr.level==zamr.max_level) {zamr.runtime = zint.t_run_fac_zone_0*timescale;}
  if (zamr.zone == 1) {zamr.runtime = zint.t_run_fac_zone_1*timescale;}
  if (zamr.zone == 2) {zamr.runtime = zint.t_run_fac_zone_2*timescale;}
  if (zamr.zone == 3) {zamr.runtime = zint.t_run_fac_zone_3*timescale;}
  if (zamr.zone == 4) {zamr.runtime = zint.t_run_fac_zone_4*timescale;}
  if (zamr.zone == 5) {zamr.runtime = zint.t_run_fac_zone_5*timescale;}
  if (zamr.zone == 6) {zamr.runtime = zint.t_run_fac_zone_6*timescale;}
  if (zamr.level==zamr.min_level) {zamr.runtime = zint.t_run_fac_zone_max*timescale;}
  if (zamr.runtime > zint.t_run_max) {zamr.runtime = zint.t_run_max;}
  zrun.id++;
  zrun.next_time = pmy_pack->pmesh->time + zamr.runtime;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::DumpData()
//! \brief dump zoom data to file

// TODO: dumping on a single rank now, should consider parallel dumping
void Zoom::DumpData() {
  if (global_variable::my_rank == 0) {
    std::cout << "Zoom: Dumping data" << std::endl;
    auto pm = pmy_pack->pmesh;
    auto &indcs = pm->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;

    std::string fname;
    fname.assign("Zoom");
    // add pmesh ncycles
    fname.append(".");
    fname.append(std::to_string(pm->ncycle));
    fname.append(".dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "wb")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Error output file could not be opened" <<std::endl;
      std::exit(EXIT_FAILURE);
    }
    int datasize = sizeof(Real);
    // xyz? bcc?
    IOWrapperSizeT cnt = mzoom*nvars*(n_ccells3)*(n_ccells2)*(n_ccells1);
    std::fwrite(coarse_w0.data(),datasize,cnt,pfile);
    auto mbptr = efld.x1e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2+1)*(n_ccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x2e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = efld.x3e;
    cnt = mzoom*(n_ccells3)*(n_ccells2+1)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x1e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2+1)*(n_ccells1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x2e;
    cnt = mzoom*(n_ccells3+1)*(n_ccells2)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    mbptr = emf0.x3e;
    cnt = mzoom*(n_ccells3)*(n_ccells2+1)*(n_ccells1+1);
    std::fwrite(mbptr.data(),datasize,cnt,pfile);
    std::fclose(pfile);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::RefineCondition()
//! \brief User-defined refinement condition(s)

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
      if (levels_thisrank.d_view(m) == old_level) {
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
//! \brief Update variables before zooming

void Zoom::UpdateVariables() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
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
  int zid = nleaf*zamr.zone;
  int nlf = nleaf;
  Real rzoom = zamr.radius;
  int nvar = nvars;
  Real rin = r_in;
  // TODO(@mhguo): it looks 0.8*rzoom works, but ideally should use edge center
  Real rzfac = 0.8; // r < rzfac*rzoom

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
      if (rad_min < rin) {
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        auto des_slice = Kokkos::subview(u0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        auto src_slice = Kokkos::subview(u, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                          Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        des_slice = Kokkos::subview(w0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        src_slice = Kokkos::subview(w, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
        Kokkos::deep_copy(des_slice, src_slice);
        par_for("zoom-update",DevExeSpace(), 0,nvar-1, cks,cke, cjs,cje, cis,cie,
        KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
          int finei = 2*i - cis;  // correct if cis = is
          int finej = 2*j - cjs;  // correct if cjs = js
          int finek = 2*k - cks;  // correct if cks = ks
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
        });
        UpdateHydroVariables(zm, m);
        if (pmy_pack->pmhd != nullptr && fix_efield) {
          DvceEdgeFld4D<Real> emf = pmy_pack->pmhd->efld;
          auto e1 = efld.x1e;
          auto e2 = efld.x2e;
          auto e3 = efld.x3e;
          auto ef1 = emf.x1e;
          auto ef2 = emf.x2e;
          auto ef3 = emf.x3e;
          // update coarse electric fields
          par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            int finei = 2*i - cis;  // correct when cis=is
            int finej = 2*j - cjs;  // correct when cjs=js
            int finek = 2*k - cks;  // correct when cks=ks
            e1(zm,k,j,i) = 0.5*(ef1(m,finek,finej,finei) + ef1(m,finek,finej,finei+1));
            e2(zm,k,j,i) = 0.5*(ef2(m,finek,finej,finei) + ef2(m,finek,finej+1,finei));
            e3(zm,k,j,i) = 0.5*(ef3(m,finek,finej,finei) + ef3(m,finek+1,finej,finei));

            // TODO(@mhguo): it looks 0.8*rzoom works, but ideally should use edge center
            Real x1v = CellCenterX(i-cis, cnx1, x1min, x1max);
            Real x2v = CellCenterX(j-cjs, cnx2, x2min, x2max);
            Real x3v = CellCenterX(k-cks, cnx3, x3min, x3max);
            Real x1f = LeftEdgeX  (i-cis, cnx1, x1min, x1max);
            Real x2f = LeftEdgeX  (j-cjs, cnx2, x2min, x2max);
            Real x3f = LeftEdgeX  (k-cks, cnx3, x3min, x3max);
            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            Real rade1 = sqrt(SQR(x1v)+SQR(x2f)+SQR(x3f));
            Real rade2 = sqrt(SQR(x1f)+SQR(x2v)+SQR(x3f));
            Real rade3 = sqrt(SQR(x1f)+SQR(x2f)+SQR(x3v));
            if (zid>0) {
              int zmp = zm-nlf;
              int prei = finei - cnx1 * x1l;
              int prej = finej - cnx2 * x2l;
              int prek = finek - cnx3 * x3l;
              if (rade1 < rzfac*rzoom) {
                e1(zm,k,j,i) = 0.5*(e1(zmp,prek,prej,prei) + e1(zmp,prek,prej,prei+1));
              }
              if (rade2 < rzfac*rzoom) {
                e2(zm,k,j,i) = 0.5*(e2(zmp,prek,prej,prei) + e2(zmp,prek,prej+1,prei));
              }
              if (rade3 < rzfac*rzoom) {
                e3(zm,k,j,i) = 0.5*(e3(zmp,prek,prej,prei) + e3(zmp,prek+1,prej,prei));
              }
            }
            // TODO(@mhguo): think to what extent we should zero out the electric field
            if (rade1 < 0.5*rzfac*rin) {e1(zm,k,j,i) = 0.0;}
            if (rade2 < 0.5*rzfac*rin) {e2(zm,k,j,i) = 0.0;}
            if (rade3 < 0.5*rzfac*rin) {e3(zm,k,j,i) = 0.0;}
          });
        }
        std::cout << "Zoom: Update variables for zoom meshblock " << zm << std::endl;
      }
    }
  }
  // if (zid != nleaf*(zamr.zone+1)) {
  //   std::cerr << "Error: Zoom::UpdateVariables() failed: zid = " << zid <<
  //                " zone = " << zamr.zone << " level = " << zamr.level << std::endl;
  //   std::exit(1);
  // }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::UpdateHydroVariables()
//! \brief Update hydro variables using conserved hydro variables

void Zoom::UpdateHydroVariables(int zm, int m) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is;
  int &js = indcs.js;
  int &ks = indcs.ks;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  DvceArray5D<Real> u0_, w0_;
  bool is_gr = pmy_pack->pcoord->is_general_relativistic;
  auto peos = (pmy_pack->pmhd != nullptr)? pmy_pack->pmhd->peos : pmy_pack->phydro->peos;
  auto eos = peos->eos_data;
  if (pmy_pack->phydro != nullptr) {
    u0_ = pmy_pack->phydro->u0;
    w0_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    u0_ = pmy_pack->pmhd->u0;
    w0_ = pmy_pack->pmhd->w0;
  }
  auto cw = coarse_w0;
  Real &x1min = size.h_view(m).x1min;
  Real &x1max = size.h_view(m).x1max;
  Real &x2min = size.h_view(m).x2min;
  Real &x2max = size.h_view(m).x2max;
  Real &x3min = size.h_view(m).x3min;
  Real &x3max = size.h_view(m).x3max;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  par_for("zoom-update-cwu",DevExeSpace(), cks,cke, cjs,cje, cis,cie,
  KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
    int fi = 2*ci - cis;  // correct when cis=is
    int fj = 2*cj - cjs;  // correct when cjs=js
    int fk = 2*ck - cks;  // correct when cks=ks
    cw(zm,IDN,ck,cj,ci) = 0.0;
    cw(zm,IM1,ck,cj,ci) = 0.0;
    cw(zm,IM2,ck,cj,ci) = 0.0;
    cw(zm,IM3,ck,cj,ci) = 0.0;
    cw(zm,IEN,ck,cj,ci) = 0.0;
    Real glower[4][4], gupper[4][4];
    // Step 1: compute coarse-grained hydro conserved variables
    for (int ii=0; ii<2; ++ii) {
      for (int jj=0; jj<2; ++jj) {
        for (int kk=0; kk<2; ++kk) {
          // Load single state of primitive variables
          HydPrim1D w;
          w.d  = w0_(m,IDN,fk+kk,fj+jj,fi+ii);
          w.vx = w0_(m,IVX,fk+kk,fj+jj,fi+ii);
          w.vy = w0_(m,IVY,fk+kk,fj+jj,fi+ii);
          w.vz = w0_(m,IVZ,fk+kk,fj+jj,fi+ii);
          w.e  = w0_(m,IEN,fk+kk,fj+jj,fi+ii);

          // call p2c function
          HydCons1D u;
          if (is_gr) {
            Real x1v = CellCenterX(fi+ii-is, nx1, x1min, x1max);
            Real x2v = CellCenterX(fj+jj-js, nx2, x2min, x2max);
            Real x3v = CellCenterX(fk+kk-ks, nx3, x3min, x3max);
            ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);
            SingleP2C_IdealGRHyd(glower, gupper, w, eos.gamma, u);
          } else {
            SingleP2C_IdealHyd(w, u);
          }

          // store conserved quantities using cw
          cw(zm,IDN,ck,cj,ci) += 0.125*u.d;
          cw(zm,IM1,ck,cj,ci) += 0.125*u.mx;
          cw(zm,IM2,ck,cj,ci) += 0.125*u.my;
          cw(zm,IM3,ck,cj,ci) += 0.125*u.mz;
          cw(zm,IEN,ck,cj,ci) += 0.125*u.e;
        }
      }
    }
    // Step 2: convert coarse-grained hydro conserved variables to primitive variables
    // Shall we add excision?
    // load single state conserved variables
    HydCons1D u;
    u.d  = cw(zm,IDN,ck,cj,ci);
    u.mx = cw(zm,IM1,ck,cj,ci);
    u.my = cw(zm,IM2,ck,cj,ci);
    u.mz = cw(zm,IM3,ck,cj,ci);
    u.e  = cw(zm,IEN,ck,cj,ci);

    HydPrim1D w;
    if (is_gr) {
      // Extract components of metric
      Real x1v = CellCenterX(ci-cis, cnx1, x1min, x1max);
      Real x2v = CellCenterX(cj-cjs, cnx2, x2min, x2max);
      Real x3v = CellCenterX(ck-cks, cnx3, x3min, x3max);
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      HydCons1D u_sr;
      Real s2;
      TransformToSRHyd(u,glower,gupper,s2,u_sr);
      bool dfloor_used=false, efloor_used=false;
      bool c2p_failure=false;
      int iter_used=0;
      SingleC2P_IdealSRHyd(u_sr, eos, s2, w,
                        dfloor_used, efloor_used, c2p_failure, iter_used);
      // apply velocity ceiling if necessary
      Real tmp = glower[1][1]*SQR(w.vx)
                + glower[2][2]*SQR(w.vy)
                + glower[3][3]*SQR(w.vz)
                + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
                + 2.0*glower[2][3]*w.vy*w.vz;
      Real lor = sqrt(1.0+tmp);
      if (lor > eos.gamma_max) {
        Real factor = sqrt((SQR(eos.gamma_max)-1.0)/(SQR(lor)-1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    } else {
      bool dfloor_used=false, efloor_used=false, tfloor_used=false;
      SingleC2P_IdealHyd(u, eos, w, dfloor_used, efloor_used, tfloor_used);
    }
    cw(zm,IDN,ck,cj,ci) = w.d;
    cw(zm,IVX,ck,cj,ci) = w.vx;
    cw(zm,IVY,ck,cj,ci) = w.vy;
    cw(zm,IVZ,ck,cj,ci) = w.vz;
    cw(zm,IEN,ck,cj,ci) = w.e;
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::SyncVariables()
//! \brief Syncronize variables between different ranks

void Zoom::SyncVariables() {
#if MPI_PARALLEL_ENABLED
  // broadcast zoom data
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  int u0_slice_size = nvars * ncells1 * ncells2 * ncells3;
  int w0_slice_size = nvars * ncells1 * ncells2 * ncells3;
  int cu_slice_size = nvars * n_ccells1 * n_ccells2 * n_ccells3;
  int cw_slice_size = nvars * n_ccells1 * n_ccells2 * n_ccells3;
  int zid = nleaf*zamr.zone;
  for (int leaf=0; leaf<nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    int zm_rank = 0;
    for (int m=0; m<pmy_pack->pmesh->nmb_total; ++m) {
      auto lloc = pmy_pack->pmesh->lloc_eachmb[m];
      if (lloc.level == zamr.level) {
        if ((lloc.lx1 == pow(2,zamr.level-1)+x1r-1) &&
            (lloc.lx2 == pow(2,zamr.level-1)+x2r-1) &&
            (lloc.lx3 == pow(2,zamr.level-1)+x3r-1)) {
          zm_rank = pmy_pack->pmesh->rank_eachmb[m];
          // print basic information
          // std::cout << "Zoom: Syncing variables for zoom meshblock " << zm
          //           << " from rank " << zm_rank << std::endl;
        }
      }
    }
    // It looks device to device communication is not supported, so copy to host first
    Kokkos::realloc(harr_5d, 1, nvars, ncells3, ncells2, ncells1);
    auto u0_slice = Kokkos::subview(u0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, u0_slice);
    MPI_Bcast(harr_5d.data(), u0_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(u0_slice, harr_5d);

    auto w0_slice = Kokkos::subview(w0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, w0_slice);
    MPI_Bcast(harr_5d.data(), w0_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(w0_slice, harr_5d);

    Kokkos::realloc(harr_5d, 1, nvars, n_ccells3, n_ccells2, n_ccells1);
    auto cu_slice = Kokkos::subview(coarse_u0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, cu_slice);
    MPI_Bcast(harr_5d.data(), cu_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(cu_slice, harr_5d);

    auto cw_slice = Kokkos::subview(coarse_w0, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    Kokkos::deep_copy(harr_5d, cw_slice);
    MPI_Bcast(harr_5d.data(), cw_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(cw_slice, harr_5d);
  }
  SyncZoomEField(efld,nleaf*zamr.zone);
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::UpdateGhostVariables()
//! \brief Update variables in ghost cells between different meshblocks

// TODO(@mhguo): add emf?
void Zoom::UpdateGhostVariables() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  auto u = u0, w = w0;
  auto cu = coarse_u0, cw = coarse_w0;
  int nvar = nvars;
  int zid = nleaf*zamr.zone;
  for (int leaf=0; leaf<nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    for (int sleaf=0; sleaf<nleaf; ++sleaf) { // source leaf index
      if (leaf == sleaf) continue;
      int sm = zid + sleaf;
      int sx1r = (sleaf%2 == 1); int sx2r = (sleaf%4 > 1); int sx3r = (sleaf > 3);
      int si = (x1r == sx1r)? 0 : (x1r? nx1 : -nx1);
      int sj = (x2r == sx2r)? 0 : (x2r? nx2 : -nx2);
      int sk = (x3r == sx3r)? 0 : (x3r? nx3 : -nx3);
      int il = (x1r == sx1r)? is : (x1r? (is - ng) : (ie + 1));
      int iu = (x1r == sx1r)? ie : (x1r? (is - 1) : (ie + ng));
      int jl = (x2r == sx2r)? js : (x2r? (js - ng) : (je + 1));
      int ju = (x2r == sx2r)? je : (x2r? (js - 1) : (je + ng));
      int kl = (x3r == sx3r)? ks : (x3r? (ks - ng) : (ke + 1));
      int ku = (x3r == sx3r)? ke : (x3r? (ks - 1) : (ke + ng));
      par_for("zoom-comm",DevExeSpace(), 0, nvar-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        u(zm,n,k,j,i) = u(sm,n,k+sk,j+sj,i+si);
        w(zm,n,k,j,i) = w(sm,n,k+sk,j+sj,i+si);
      });
      si = (x1r == sx1r)? 0 : (x1r? cnx1 : -cnx1);
      sj = (x2r == sx2r)? 0 : (x2r? cnx2 : -cnx2);
      sk = (x3r == sx3r)? 0 : (x3r? cnx3 : -cnx3);
      il = (x1r == sx1r)? cis : (x1r? (cis - ng) : (cie + 1));
      iu = (x1r == sx1r)? cie : (x1r? (cis - 1) : (cie + ng));
      jl = (x2r == sx2r)? cjs : (x2r? (cjs - ng) : (cje + 1));
      ju = (x2r == sx2r)? cje : (x2r? (cjs - 1) : (cje + ng));
      kl = (x3r == sx3r)? cks : (x3r? (cks - ng) : (cke + 1));
      ku = (x3r == sx3r)? cke : (x3r? (cks - 1) : (cke + ng));
      par_for("zoom-comm-c",DevExeSpace(), 0, nvar-1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        cu(zm,n,k,j,i) = cu(sm,n,k+sk,j+sj,i+si);
        cw(zm,n,k,j,i) = cw(sm,n,k+sk,j+sj,i+si);
      });
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::ApplyVariables()
//! \brief Apply finer level variables to coarser level

void Zoom::ApplyVariables() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];
  Real gamma = 0.0;
  DvceArray5D<Real> u_, w_;
  if (pmy_pack->phydro != nullptr) {
    gamma = pmy_pack->phydro->peos->eos_data.gamma;
    u_ = pmy_pack->phydro->u0;
    w_ = pmy_pack->phydro->w0;
  } else if (pmy_pack->pmhd != nullptr) {
    gamma = pmy_pack->pmhd->peos->eos_data.gamma;
    u_ = pmy_pack->pmhd->u0;
    w_ = pmy_pack->pmhd->w0;
  }
  Real rzoom = zamr.radius;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto u0_ = u0, w0_ = w0;
  int zid = nleaf*zamr.zone;
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
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        if (pmy_pack->phydro != nullptr) { // TODO: this seems wrong, may use 2*r_zoom
          auto src_slice = Kokkos::subview(u0_, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          auto des_slice = Kokkos::subview(u_, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          Kokkos::deep_copy(des_slice, src_slice);
          src_slice = Kokkos::subview(w0_, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          des_slice = Kokkos::subview(w_, Kokkos::make_pair(m,m+1), Kokkos::ALL,
                                      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
          Kokkos::deep_copy(des_slice, src_slice);
        } else if (pmy_pack->pmhd != nullptr) {
          auto b = pmy_pack->pmhd->b0;
          par_for("zoom_apply", DevExeSpace(),ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
          KOKKOS_LAMBDA(int k, int j, int i) {
            Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
            Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
            Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
            Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));
            if (rad < 2*rzoom) { // apply to 2*rzoom since rzoom is already updated
              w_(m,IDN,k,j,i) = w0_(zm,IDN,k,j,i);
              w_(m,IM1,k,j,i) = w0_(zm,IM1,k,j,i);
              w_(m,IM2,k,j,i) = w0_(zm,IM2,k,j,i);
              w_(m,IM3,k,j,i) = w0_(zm,IM3,k,j,i);
              w_(m,IEN,k,j,i) = w0_(zm,IEN,k,j,i);
              Real glower[4][4], gupper[4][4];
              ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

              // Load single state of primitive variables
              MHDPrim1D w;
              w.d  = w_(m,IDN,k,j,i);
              w.vx = w_(m,IVX,k,j,i);
              w.vy = w_(m,IVY,k,j,i);
              w.vz = w_(m,IVZ,k,j,i);
              w.e  = w_(m,IEN,k,j,i);

              // load cell-centered fields into primitive state
              // use simple linear average of face-centered fields as bcc is not updated
              w.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
              w.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
              w.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));

              // call p2c function
              HydCons1D u;
              SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

              // store conserved quantities in 3D array
              u_(m,IDN,k,j,i) = u.d;
              u_(m,IM1,k,j,i) = u.mx;
              u_(m,IM2,k,j,i) = u.my;
              u_(m,IM3,k,j,i) = u.mz;
              u_(m,IEN,k,j,i) = u.e;
            }
          });
        }
        std::cout << "Zoom: Apply variables for zoom meshblock " << zm << std::endl;
      }
    }
  }
  // if (zid != nleaf*(zamr.zone+1)) {
  //   std::cerr << "Error: Zoom::ApplyVariables() failed: zid = " << zid <<
  //                " zone = " << zamr.zone << " level = " << zamr.level << std::endl;
  //   std::exit(1);
  // }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::FixEField()
//! \brief Modify E field on the zoomed grid

// TODO(@mhguo): may change FixEField to a more general name
void Zoom::FixEField(DvceEdgeFld4D<Real> emf) {
  if (emf_flag == 0) {
    MeanEField(emf);
  } else if (emf_flag == 1) {
    AddEField(emf);
  } else if (emf_flag == 2) {
    AddDeltaEField(emf);
  } else if (emf_flag == 3) {
    AddDeltaEField(emf);
  } else {
    std::cerr << "Error: Zoom::FixEField() failed: emf_flag = " << emf_flag << std::endl;
    std::exit(1);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::MeanEField()
//! \brief Fix E field on the zoomed grid using mean value

void Zoom::MeanEField(DvceEdgeFld4D<Real> emf) {
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

  // TODO(@mhguo): do MPI_Allreduce here, otherwise not working for multiple ranks

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
//! \fn void Zoom::AddEField()
//! \brief Add E field from small scale

// TODO(@mhguo): check the corner case in ghost zones
void Zoom::AddEField(DvceEdgeFld4D<Real> emf) {
  if (zamr.zone == 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb1 = pmy_pack->nmb_thispack-1;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  Real rzoom = zamr.radius;

  int zid = nleaf*(zamr.zone-1);
  Real fac = emf_f1; //(rzoom-rad)/rzoom;
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

    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    int ci = i - cnx1 * x1l;
    int cj = j - cnx2 * x2l;
    int ck = k - cnx3 * x3l;
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

//----------------------------------------------------------------------------------------
//! \fn void Zoom::AddDeltaEField()
//! \brief Add delta E field from small scale or adaptive E field

// TODO(@mhguo): check the corner case in ghost zones
void Zoom::AddDeltaEField(DvceEdgeFld4D<Real> emf) {
  if (zamr.zone == 0) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb1 = pmy_pack->nmb_thispack-1;
  if (zamr.first_emf) {
    UpdateDeltaEField(emf);
    SyncZoomEField(emf0,nleaf*(zamr.zone-1));
    SyncZoomEField(delta_efld,nleaf*(zamr.zone-1));
    SetMaxEField();
    if (dump_diag) {
      DumpData();
    }
    // TODO(@mhguo): print basic information or not
    zamr.first_emf = false;
  }
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  Real rzoom = zamr.radius;

  int zid = nleaf*(zamr.zone-1);
  Real &f0 = emf_f0; //(rad-rzoom)/rzoom;
  Real &f1 = emf_f1; //(rzoom-rad)/rzoom;
  // TODO: add adaptive E field
  if (emf_flag == 3) { // 3 = adaptive
    SphericalFlux(1,zamr.zone);
    Real poy_pow_0=zoom_fluxes(0,zamr.zone,2);
    Real poy_pow=zoom_fluxes(1,zamr.zone,2);
    // if (poy_pow > poy_pow_0) {
    //   efld_fac *= 0.9;
    // } else {
    //   efld_fac_a *= 1.1;
    // }
    // plus/minus value?
    // if (poy_pow > poy_pow_0) {
    //   f1 = std::min(f1+0.01, 1.0);
    // } else {
    //   f1 = std::max(f1-0.01, -1.0);
    // }
    // print basic flux information
    // TODO(@mhguo): remove this or set interval
    // if (global_variable::my_rank == 0) {
    //   std::cout << " Zoom Fluxes: mdot_0 = " << zoom_fluxes(0,zamr.zone,0)
    //             << " mdot = " << zoom_fluxes(1,zamr.zone,0)
    //             << " ehdot_0 = " << zoom_fluxes(0,zamr.zone,1)
    //             << " ehdot = " << zoom_fluxes(1,zamr.zone,1)
    //             << " poy_pow_0 = " << poy_pow_0
    //             << " poy_pow = " << poy_pow << " emf_f1 = " << f1 << std::endl;
    // }
  }
  Real emax1 = emf_fmax*max_emf0(zamr.zone-1,0);
  Real emax2 = emf_fmax*max_emf0(zamr.zone-1,1);
  Real emax3 = emf_fmax*max_emf0(zamr.zone-1,2);
  par_for("apply-emf", DevExeSpace(), 0, nmb1, ks-1, ke+1, js-1, je+1 ,is-1, ie+1,
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

    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    int ci = i - cnx1 * x1l;
    int cj = j - cnx2 * x2l;
    int ck = k - cnx3 * x3l;
    if (sqrt(SQR(x1v)+SQR(x2f)+SQR(x3f)) < rzoom) {
      // ef1(m,k,j,i) = f0*ef1(m,k,j,i) + f1*de1(zm,ck,cj,ci);
      // limit de1 to be between -emax1 and emax1
      ef1(m,k,j,i) = f0*ef1(m,k,j,i) + f1*fmax(-emax1, fmin(emax1, de1(zm,ck,cj,ci)));
    }
    if (sqrt(SQR(x1f)+SQR(x2v)+SQR(x3f)) < rzoom) {
      // ef2(m,k,j,i) = f0*ef2(m,k,j,i) + f1*de2(zm,ck,cj,ci);
      // limit de2 to be between -emax2 and emax2
      ef2(m,k,j,i) = f0*ef2(m,k,j,i) + f1*fmax(-emax2, fmin(emax2, de2(zm,ck,cj,ci)));
    }
    if (sqrt(SQR(x1f)+SQR(x2f)+SQR(x3v)) < rzoom) {
      // ef3(m,k,j,i) = f0*ef3(m,k,j,i) + f1*de3(zm,ck,cj,ci);
      // limit de3 to be between -emax3 and emax3
      ef3(m,k,j,i) = f0*ef3(m,k,j,i) + f1*fmax(-emax3, fmin(emax3, de3(zm,ck,cj,ci)));
    }
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::UpdateDeltaEField()
//! \brief Update delta E field on the zoomed grid

void Zoom::UpdateDeltaEField(DvceEdgeFld4D<Real> emf) {
  if (global_variable::my_rank == 0) {
    std::cout << "Zoom: UpdateDeltaEField" << std::endl;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int &cis = indcs.cis;  int &cie  = indcs.cie;
  int &cjs = indcs.cjs;  int &cje  = indcs.cje;
  int &cks = indcs.cks;  int &cke  = indcs.cke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;
  int nmb = pmy_pack->nmb_thispack;
  int mbs = pmy_pack->pmesh->gids_eachrank[global_variable::my_rank];
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto e01 = emf0.x1e;
  auto e02 = emf0.x2e;
  auto e03 = emf0.x3e;
  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  auto ef1 = emf.x1e;
  auto ef2 = emf.x2e;
  auto ef3 = emf.x3e;
  int zid = nleaf*(zamr.zone-1);
  Real rin = r_in;
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
      if (rad_min < rin) {
        bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
        bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
        int leaf_id = 1*x1r + 2*x2r + 4*x3r;
        int zm = zid + leaf_id;
        // update delta electric fields
        par_for("zoom-update-efld",DevExeSpace(), cks,cke+1, cjs,cje+1, cis,cie+1,
        KOKKOS_LAMBDA(const int ck, const int cj, const int ci) {
          int fi = ci + cnx1 * x1l; // correct if cis = is
          int fj = cj + cnx2 * x2l; // correct if cjs = js
          int fk = ck + cnx3 * x3l; // correct if cks = ks
          e01(zm,ck,cj,ci) = ef1(m,fk,fj,fi);
          e02(zm,ck,cj,ci) = ef2(m,fk,fj,fi);
          e03(zm,ck,cj,ci) = ef3(m,fk,fj,fi);
          de1(zm,ck,cj,ci) = e1(zm,ck,cj,ci) - ef1(m,fk,fj,fi);
          de2(zm,ck,cj,ci) = e2(zm,ck,cj,ci) - ef2(m,fk,fj,fi);
          de3(zm,ck,cj,ci) = e3(zm,ck,cj,ci) - ef3(m,fk,fj,fi);
        });
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::SyncZoomEField()
//! \brief Syncronize variables between different ranks

void Zoom::SyncZoomEField(DvceEdgeFld4D<Real> emf, int zid) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    std::cout << "Zoom: SyncZoomEField" << std::endl;
  }
  // broadcast zoom data
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
  int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
  int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
  int e1_slice_size = (n_ccells3+1) * (n_ccells2+1) * n_ccells1;
  int e2_slice_size = (n_ccells3+1) * n_ccells2 * (n_ccells1+1);
  int e3_slice_size = n_ccells3 * (n_ccells2+1) * (n_ccells1+1);

  // int zid = nleaf*(zamr.zone-1);
  for (int leaf=0; leaf<nleaf; ++leaf) {
    // determine which rank is the "root" rank
    int zm = zid + leaf;
    int x1r = (leaf%2 == 1); int x2r = (leaf%4 > 1); int x3r = (leaf > 3);
    int zm_rank = 0;
    for (int m=0; m<pmy_pack->pmesh->nmb_total; ++m) {
      auto lloc = pmy_pack->pmesh->lloc_eachmb[m];
      // TODO(@mhguo): not working for half domain
      if (lloc.level == zamr.level) {
        if ((lloc.lx1 == pow(2,zamr.level-1)+x1r-1) &&
            (lloc.lx2 == pow(2,zamr.level-1)+x2r-1) &&
            (lloc.lx3 == pow(2,zamr.level-1)+x3r-1)) {
          zm_rank = pmy_pack->pmesh->rank_eachmb[m];
          // print basic information
          // std::cout << "Zoom: Syncing delta efield for zoom meshblock " << zm
          //           << " from rank " << zm_rank << std::endl;
        }
      }
    }
    // It looks device to device communication is not supported, so copy to host first
    Kokkos::realloc(harr_4d, 1, n_ccells3+1, n_ccells2+1, n_ccells1);
    auto e1_slice = Kokkos::subview(emf.x1e, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e1_slice);
    MPI_Bcast(harr_4d.data(), e1_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e1_slice, harr_4d);

    Kokkos::realloc(harr_4d, 1, n_ccells3+1, n_ccells2, n_ccells1+1);
    auto e2_slice = Kokkos::subview(emf.x2e, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e2_slice);
    MPI_Bcast(harr_4d.data(), e2_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e2_slice, harr_4d);

    Kokkos::realloc(harr_4d, 1, n_ccells3, n_ccells2+1, n_ccells1+1);
    auto e3_slice = Kokkos::subview(emf.x3e, Kokkos::make_pair(zm,zm+1), Kokkos::ALL,
                                    Kokkos::ALL, Kokkos::ALL);
    Kokkos::deep_copy(harr_4d, e3_slice);
    MPI_Bcast(harr_4d.data(), e3_slice_size, MPI_ATHENA_REAL, zm_rank, MPI_COMM_WORLD);
    Kokkos::deep_copy(e3_slice, harr_4d);
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::SetMaxEField()
//! \brief Set maximum E field on the zoomed grid

void Zoom::SetMaxEField() {
  if (zamr.zone == 0) return;
  auto pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, cnx1p1 = indcs.cnx1+1;
  int js = indcs.js, cnx2p1 = indcs.cnx2+1;
  int ks = indcs.ks, cnx3p1 = indcs.cnx3+1;
  const int zid = nleaf*(zamr.zone-1);
  const int nmkji = nleaf*cnx3p1*cnx2p1*cnx1p1;
  const int nkji = cnx3p1*cnx2p1*cnx1p1;
  const int nji  = cnx2p1*cnx1p1;
  const int ni = cnx1p1;
  auto e01 = emf0.x1e;
  auto e02 = emf0.x2e;
  auto e03 = emf0.x3e;
  Real emax1 = 0.0;
  Real emax2 = 0.0;
  Real emax3 = 0.0;
  Kokkos::parallel_reduce("max_emf0_1",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_e1, Real &max_e2, Real &max_e3) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + is;
    k += ks;
    j += js;
    m += zid;
    max_e1 = fmax(max_e1, fabs(e01(m,k,j,i)));
    max_e2 = fmax(max_e2, fabs(e02(m,k,j,i)));
    max_e3 = fmax(max_e3, fabs(e03(m,k,j,i)));
  }, Kokkos::Max<Real>(emax1), Kokkos::Max<Real>(emax2),Kokkos::Max<Real>(emax3));
  max_emf0(zamr.zone-1,0) = emax1;
  max_emf0(zamr.zone-1,1) = emax2;
  max_emf0(zamr.zone-1,2) = emax3;
  if (global_variable::my_rank == 0) {
    std::cout << "Zoom: MaxEField: max_emf0 = " << emax1 << " " << emax2 << " " << emax3
              << std::endl;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::SphericalFlux()
//! \brief Compute fluxes on the zoomed grid

// TODO (@mhguo): consider: 1. whether we really need this, 2: where to put this
void Zoom::SphericalFlux(int aim_id, int grid_id) {
  int &g = grid_id;
  auto pmbp = pmy_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

  // capture variables
  auto &w0_ = pmbp->pmhd->w0;
  auto &bcc_ = pmbp->pmhd->bcc0;
  int nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;

  // extract grids
  auto &grids = spherical_grids;
  // interpolate cell-centered magnetic fields and store
  // NOTE(@pdmullen): We later reuse the interp_vals array to interpolate primitives.
  // Therefore, we must stow interpolated field components.
  grids[g]->InterpolateToSphere(3, bcc_);
  DualArray2D<Real> interpolated_bcc;
  Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
  Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
  interpolated_bcc.template modify<DevExeSpace>();
  interpolated_bcc.template sync<HostMemSpace>();

  // interpolate primitives
  grids[g]->InterpolateToSphere(nvars, w0_);

  // compute poynting flux power
  int &nflx = nflux;
  std::vector<Real> zfluxes(nflx);
  for (int n=0; n<nflx; ++n) {
    zfluxes[n] = 0.0;
  }
  for (int n=0; n<grids[g]->nangles; ++n) {
    // extract coordinate data at this angle
    Real r = grids[g]->radius;
    Real theta = grids[g]->polar_pos.h_view(n,0);
    Real x1 = grids[g]->interp_coord.h_view(n,0);
    Real x2 = grids[g]->interp_coord.h_view(n,1);
    Real x3 = grids[g]->interp_coord.h_view(n,2);
    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

    // extract interpolated primitives
    Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);  
    Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
    Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
    Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
    Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);

    // extract interpolated field components
    Real &int_bx = interpolated_bcc.h_view(n,IBX);
    Real &int_by = interpolated_bcc.h_view(n,IBY);
    Real &int_bz = interpolated_bcc.h_view(n,IBZ);

    // Compute interpolated u^\mu in CKS
    Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
             2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
             2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
    Real alpha = sqrt(-1.0/gupper[0][0]);
    Real gamma = sqrt(1.0 + q);
    Real u0 = gamma/alpha;
    Real u1 = int_vx - alpha * gamma * gupper[0][1];
    Real u2 = int_vy - alpha * gamma * gupper[0][2];
    Real u3 = int_vz - alpha * gamma * gupper[0][3];

    // Lower vector indices
    Real u_0 = glower[0][0]*u0 + glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;  
    Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
    Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
    Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

    // Calculate 4-magnetic field (returns zero if not MHD)
    Real b0 = u_1*int_bx + u_2*int_by + u_3*int_bz;
    Real b1 = (int_bx + b0 * u1) / u0;
    Real b2 = (int_by + b0 * u2) / u0;
    Real b3 = (int_bz + b0 * u3) / u0;

    // compute b_\mu in CKS and b_sq (returns zero if not MHD)
    Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
    Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
    Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
    Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
    Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

    // Transform CKS 4-velocity and 4-magnetic field to spherical KS
    Real a2 = SQR(spin);
    Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
    Real r2 = SQR(r);
    Real sth = sin(theta);
    Real drdx = r*x1/(2.0*r2 - rad2 + a2);
    Real drdy = r*x2/(2.0*r2 - rad2 + a2);
    Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
    Real dphdx = (-x2/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdx);
    Real dphdy = ( x1/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdy);
    Real dphdz = (spin/(r2 + a2)*drdz);
    // r,phi component of 4-velocity in spherical KS
    Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
    Real uph = dphdx*u1 + dphdy*u2 + dphdz*u3;
    // r,phi component of 4-magnetic field in spherical KS
    Real br  = drdx *b1 + drdy *b2 + drdz *b3;
    Real bph = dphdx*b1 + dphdy*b2 + dphdz*b3;

    // integration params
    Real &domega = grids[g]->solid_angles.h_view(n);
    Real sqrtmdet = (r2+SQR(spin*cos(theta)));

    // flags
    Real on = (int_dn != 0.0)? 1.0 : 0.0; // check if angle is on this rank

    // compute mass flux
    Real m_flx = int_dn*ur;
    // compute hydro energy flux
    Real t1_0_h = (int_dn + gamma*int_ie)*ur*u_0;
    Real ehyd_flx = -t1_0_h - m_flx;
    // compute poynting flux
    Real t1_0_m = (b_sq)*ur*u_0 - br*b_0;
    Real poy_flx = -t1_0_m;

    // accumulate mass accretion rate
    zfluxes[0] += m_flx*sqrtmdet*domega*on;
    // accumulate hydro energy power
    zfluxes[1] += ehyd_flx*sqrtmdet*domega*on;
    // accumulate poynting flux power
    zfluxes[2] += poy_flx*sqrtmdet*domega*on;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, zfluxes.data(), nflx,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  for (int n=0; n<nflx; ++n) {
    // zoom_fluxes[aim_id][g][n] = zfluxes[n];
    zoom_fluxes(aim_id,g,n) = zfluxes[n];
  }

  // TODO(@mhguo): remove this or set interval
  // if (global_variable::my_rank == 0) {
  //     std::cout << " Zoom Spherical Fluxes: id=" << aim_id << " zone=" << g
  //               << " mdot = " << zoom_fluxes(aim_id,zamr.zone,0)
  //               << " ehdot = " << zoom_fluxes(aim_id,zamr.zone,1)
  //               << " emdot = " << zoom_fluxes(aim_id,zamr.zone,2)
  //               << std::endl;
  // }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::NewTimeStep()
//! \brief New time step for zoom

Real Zoom::NewTimeStep(Mesh* pm) {
  Real dt = pm->dt/pm->cfl_no;
  if (!zoom_dt) return dt;
  bool &is_gr = pmy_pack->pcoord->is_general_relativistic;
  bool is_mhd = (pmy_pack->pmhd != nullptr);
  dt = (is_gr)? GRTimeStep(pm) : dt; // replace dt with GRTimeStep
  // TODO(@mhguo): 1. EMFTimeStep is too small, 2. we may use v=c instead
  // Real dt_emf = dt;
  // if (emf_dt && is_mhd) {
  //   dt_emf = EMFTimeStep(pm);
  //   if (ndiag > 0 && (pm->ncycle % ndiag == 0) && (zamr.zone > 0)) {
  //     if (dt_emf < dt) {
  //       std::cout << "Zoom: dt_emf = " << dt_emf << " dt = " << dt
  //                 << " on rank " << global_variable::my_rank << std::endl;
  //     }
  //   }
  // }
  // dt = fmin(dt_emf, dt); // get minimum of EMFTimeStep and dt
  return dt;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::GRTimeStep()
//! \brief New time step for GR zoom, only for GR since others are already handled

Real Zoom::GRTimeStep(Mesh* pm) {
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  bool is_hydro = (pmy_pack->phydro != nullptr);
  bool is_mhd = (pmy_pack->pmhd != nullptr);

  if (is_hydro) {
    auto &w0_ = pmy_pack->phydro->w0;
    auto &eos = pmy_pack->phydro->peos->eos_data;

    // find smallest dx/(v +/- Cs) in each direction for hydrodynamic problems
    Kokkos::parallel_reduce("ZHydroNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      // Use the GR sound speed to compute the time step
      // References to left primitives
      Real &wd = w0_(m,IDN,k,j,i);
      Real &ux = w0_(m,IVX,k,j,i);
      Real &uy = w0_(m,IVY,k,j,i);
      Real &uz = w0_(m,IVZ,k,j,i);

      // FIXME ERM: Ideal fluid for now
      Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

      // Extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Calculate 4-velocity (contravariant compt)
      Real q = glower[IVX][IVX] * SQR(ux) + glower[IVY][IVY] * SQR(uy) +
               glower[IVZ][IVZ] * SQR(uz) + 2.0*glower[IVX][IVY] * ux * uy +
           2.0*glower[IVX][IVZ] * ux * uz + 2.0*glower[IVY][IVZ] * uy * uz;

      Real alpha = std::sqrt(-1.0/gupper[0][0]);
      Real gamma = sqrt(1.0 + q);
      Real uu[4];
      uu[0] = gamma / alpha;
      uu[IVX] = ux - alpha * gamma * gupper[0][IVX];
      uu[IVY] = uy - alpha * gamma * gupper[0][IVY];
      uu[IVZ] = uz - alpha * gamma * gupper[0][IVZ];

      // Calculate wavespeeds
      Real lm, lp;
      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVX], gupper[0][0],
                                  gupper[0][IVX], gupper[IVX][IVX], lp, lm);
      max_dv1 = fmax(fabs(lm), lp);

      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVY], gupper[0][0],
                                  gupper[0][IVY], gupper[IVY][IVY], lp, lm);
      max_dv2 = fmax(fabs(lm), lp);

      eos.IdealGRHydroSoundSpeeds(wd, p, uu[0], uu[IVZ], gupper[0][0],
                                  gupper[0][IVZ], gupper[IVZ][IVZ], lp, lm);
      max_dv3 = fmax(fabs(lm), lp);

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  } else if (is_mhd) {
    auto &w0_ = pmy_pack->pmhd->w0;
    auto &eos = pmy_pack->pmhd->peos->eos_data;
    auto &bcc0_ = pmy_pack->pmhd->bcc0;

    // find smallest dx/(v +/- Cf) in each direction for mhd problems
    Kokkos::parallel_reduce("ZMHDNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
      // compute m,k,j,i indices of thread and call function
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/nx1;
      int i = (idx - m*nkji - k*nji - j*nx1) + is;
      k += ks;
      j += js;
      Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;

      // Use the GR fast magnetosonic speed to compute the time step
      // References to left primitives
      Real &wd = w0_(m,IDN,k,j,i);
      Real &ux = w0_(m,IVX,k,j,i);
      Real &uy = w0_(m,IVY,k,j,i);
      Real &uz = w0_(m,IVZ,k,j,i);
      Real &bcc1 = bcc0_(m,IBX,k,j,i);
      Real &bcc2 = bcc0_(m,IBY,k,j,i);
      Real &bcc3 = bcc0_(m,IBZ,k,j,i);

      // FIXME ERM: Ideal fluid for now
      Real p = eos.IdealGasPressure(w0_(m,IEN,k,j,i));

      // Extract components of metric
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

      // Calculate 4-velocity (contravariant compt)
      Real q = glower[IVX][IVX] * SQR(ux) + glower[IVY][IVY] * SQR(uy) +
               glower[IVZ][IVZ] * SQR(uz) + 2.0*glower[IVX][IVY] * ux * uy +
           2.0*glower[IVX][IVZ] * ux * uz + 2.0*glower[IVY][IVZ] * uy * uz;

      Real alpha = std::sqrt(-1.0/gupper[0][0]);
      Real gamma = sqrt(1.0 + q);
      Real uu[4];
      uu[0] = gamma / alpha;
      uu[IVX] = ux - alpha * gamma * gupper[0][IVX];
      uu[IVY] = uy - alpha * gamma * gupper[0][IVY];
      uu[IVZ] = uz - alpha * gamma * gupper[0][IVZ];

      // lower vector indices (covariant compt)
      Real ul[4];
      ul[0]   = glower[0][0]  *uu[0]   + glower[0][IVX]*uu[IVX] +
                glower[0][IVY]*uu[IVY] + glower[0][IVZ]*uu[IVZ];

      ul[IVX] = glower[IVX][0]  *uu[0]   + glower[IVX][IVX]*uu[IVX] +
                glower[IVX][IVY]*uu[IVY] + glower[IVX][IVZ]*uu[IVZ];

      ul[IVY] = glower[IVY][0]  *uu[0]   + glower[IVY][IVX]*uu[IVX] +
                glower[IVY][IVY]*uu[IVY] + glower[IVY][IVZ]*uu[IVZ];

      ul[IVZ] = glower[IVZ][0]  *uu[0]   + glower[IVZ][IVX]*uu[IVX] +
                glower[IVZ][IVY]*uu[IVY] + glower[IVZ][IVZ]*uu[IVZ];


      // Calculate 4-magnetic field in right state
      Real bu[4];
      bu[0]   = ul[IVX]*bcc1 + ul[IVY]*bcc2 + ul[IVZ]*bcc3;
      bu[IVX] = (bcc1 + bu[0] * uu[IVX]) / uu[0];
      bu[IVY] = (bcc2 + bu[0] * uu[IVY]) / uu[0];
      bu[IVZ] = (bcc3 + bu[0] * uu[IVZ]) / uu[0];

      // lower vector indices (covariant compt)
      Real bl[4];
      bl[0]   = glower[0][0]  *bu[0]   + glower[0][IVX]*bu[IVX] +
                glower[0][IVY]*bu[IVY] + glower[0][IVZ]*bu[IVZ];

      bl[IVX] = glower[IVX][0]  *bu[0]   + glower[IVX][IVX]*bu[IVX] +
                glower[IVX][IVY]*bu[IVY] + glower[IVX][IVZ]*bu[IVZ];

      bl[IVY] = glower[IVY][0]  *bu[0]   + glower[IVY][IVX]*bu[IVX] +
                glower[IVY][IVY]*bu[IVY] + glower[IVY][IVZ]*bu[IVZ];

      bl[IVZ] = glower[IVZ][0]  *bu[0]   + glower[IVZ][IVX]*bu[IVX] +
                glower[IVZ][IVY]*bu[IVY] + glower[IVZ][IVZ]*bu[IVZ];

      Real b_sq = bl[0]*bu[0] + bl[IVX]*bu[IVX] + bl[IVY]*bu[IVY] +bl[IVZ]*bu[IVZ];

      // Calculate wavespeeds
      Real lm, lp;
      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVX], b_sq, gupper[0][0],
                               gupper[0][IVX], gupper[IVX][IVX], lp, lm);
      max_dv1 = fmax(fabs(lm), lp);

      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVY], b_sq, gupper[0][0],
                               gupper[0][IVY], gupper[IVY][IVY], lp, lm);
      max_dv2 = fmax(fabs(lm), lp);

      eos.IdealGRMHDFastSpeeds(wd, p, uu[0], uu[IVZ], b_sq, gupper[0][0],
                               gupper[0][IVZ], gupper[IVZ][IVZ], lp, lm);
      max_dv3 = fmax(fabs(lm), lp);

      min_dt1 = fmin((size.d_view(m).dx1/max_dv1), min_dt1);
      min_dt2 = fmin((size.d_view(m).dx2/max_dv2), min_dt2);
      min_dt3 = fmin((size.d_view(m).dx3/max_dv3), min_dt3);
    }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));
  }

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  Real dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void Zoom::EMFTimeStep()
//! \brief New time step for emf in zoom

// TODO(@mhguo): not working now, need to update
Real Zoom::EMFTimeStep(Mesh* pm) {
  if (zamr.zone == 0) return std::numeric_limits<float>::max();
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int cnx1 = indcs.cnx1, cnx2 = indcs.cnx2, cnx3 = indcs.cnx3;

  Real dt1 = std::numeric_limits<float>::max();
  Real dt2 = std::numeric_limits<float>::max();
  Real dt3 = std::numeric_limits<float>::max();

  // capture class variables for kernel
  auto &size = pmy_pack->pmb->mb_size;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  auto &eos = pmy_pack->pmhd->peos->eos_data;
  auto &bcc0_ = pmy_pack->pmhd->bcc0;

  auto de1 = delta_efld.x1e;
  auto de2 = delta_efld.x2e;
  auto de3 = delta_efld.x3e;
  Real rzoom = zamr.radius;

  int zid = nleaf*(zamr.zone-1);
  Real &f0 = emf_f0; //(rad-rzoom)/rzoom;
  Real &f1 = emf_f1; //(rzoom-rad)/rzoom;

  // find smallest dx*|B/E| in each direction for mhd problems
  Kokkos::parallel_reduce("ZEMFNudt",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt1, Real &min_dt2, Real &min_dt3) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    // Real max_dv1 = 0.0, max_dv2 = 0.0, max_dv3 = 0.0;
    Real max_de1 = 0.0, max_de2 = 0.0, max_de3 = 0.0;

    // Use the GR fast magnetosonic speed to compute the time step
    // References to left primitives
    Real &bcc1 = bcc0_(m,IBX,k,j,i);
    Real &bcc2 = bcc0_(m,IBY,k,j,i);
    Real &bcc3 = bcc0_(m,IBZ,k,j,i);

    // Extract components of metric
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

    bool x1r = (x1max > 0.0); bool x2r = (x2max > 0.0); bool x3r = (x3max > 0.0);
    bool x1l = (x1min < 0.0); bool x2l = (x2min < 0.0); bool x3l = (x3min < 0.0);
    int leaf_id = 1*x1r + 2*x2r + 4*x3r;
    int zm = zid + leaf_id;
    int ci = i - cnx1 * x1l;
    int cj = j - cnx2 * x2l;
    int ck = k - cnx3 * x3l;
    // should be face centered or edge centered, but use cell centered for now
    if (rad < rzoom) {
      max_de1 = fmax(fabs(de1(zm,ck,cj,ci)), fmax(fabs(de1(zm,ck+1,cj,ci)),
                fmax(fabs(de1(zm,ck,cj+1,ci)), fabs(de1(zm,ck+1,cj+1,ci)))));
      max_de2 = fmax(fabs(de2(zm,ck,cj,ci)), fmax(fabs(de2(zm,ck+1,cj,ci)),
                fmax(fabs(de2(zm,ck,cj,ci+1)), fabs(de2(zm,ck+1,cj,ci+1)))));
      max_de3 = fmax(fabs(de3(zm,ck,cj,ci)), fmax(fabs(de3(zm,ck,cj+1,ci)),
                fmax(fabs(de3(zm,ck,cj,ci+1)), fabs(de3(zm,ck,cj+1,ci+1)))));
    }
    Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
    min_dt1 = fmin(fabs(bcc1)/(max_de2/dx3+max_de3/dx2), min_dt1);
    min_dt2 = fmin(fabs(bcc2)/(max_de3/dx1+max_de1/dx3), min_dt2);
    min_dt3 = fmin(fabs(bcc3)/(max_de1/dx2+max_de2/dx1), min_dt3);
  }, Kokkos::Min<Real>(dt1), Kokkos::Min<Real>(dt2),Kokkos::Min<Real>(dt3));

  // compute minimum of dt1/dt2/dt3 for 1D/2D/3D problems
  Real dtnew = dt1;
  if (pmy_pack->pmesh->multi_d) { dtnew = std::min(dtnew, dt2); }
  if (pmy_pack->pmesh->three_d) { dtnew = std::min(dtnew, dt3); }

  return dtnew;
}

} // namespace zoom
