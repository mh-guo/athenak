//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_bondi.cpp
//! \brief Problem generator for spherically symmetric black hole accretion.

#include <cmath>   // abs(), NAN, pow(), sqrt()
#include <cstring> // strcmp()
#include <iostream>
#include <sstream>
#include <string> // string
#include <cstdio> // fclose

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "pgen/turb_init.hpp"
#include "pgen/turb_mhd.hpp"
#include "pgen/zoom.hpp"

#include <Kokkos_Random.hpp>

namespace {

KOKKOS_INLINE_FUNCTION
static Real Acceleration(const Real r) {
  return -1.0/SQR(r);
}

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3);

struct bondi_pgen {
  Real spin;                // black hole spin
  Real dexcise, pexcise;    // excision parameters
  int  ic_type;             // initial condition type
  Real n_adi, k_adi, gm;    // hydro EOS parameters
  Real r_crit;              // sonic point radius
  Real c1, c2;              // useful constants
  Real temp_min, temp_max;  // bounds for temperature root find
  Real temp_inf, c_s_inf;   // asymptotic temperature and sound speed
  Real rho_inf, pgas_inf;   // asymptotic density and pressure
  Real r_bondi;             // Bondi radius 2GM/c_s^2
  Real r_sink;              // sink radius
  Real d_sink, p_sink;      // sink parameters
  bool reset_ic = false;    // reset initial conditions after run
};

  bondi_pgen bondi;

// prototypes for user-defined BCs and error functions
void FixedBondiInflow(Mesh *pm);
void AddUserSrcs(Mesh *pm, const Real bdt);
void BondiFluxes(HistoryData *pdata, Mesh *pm);
void ZoomAMR(MeshBlockPack* pmbp) {pmbp->pzoom->AMR();}
Real ZoomNewTimeStep(Mesh* pm) {return pm->pmb_pack->pzoom->NewTimeStep(pm);}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief set initial conditions for Bondi accretion test
//  Compile with '-D PROBLEM=bondi' to enroll as user-specific problem generator
//    reference: Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto peos = (is_mhd) ? pmbp->pmhd->peos : pmbp->phydro->peos;
  auto &coord = pmbp->pcoord->coord_data;

  // set user-defined BCs and error function pointers
  // pgen_final_func = BondiErrors;
  user_bcs_func = FixedBondiInflow;
  user_srcs_func = AddUserSrcs;
  user_hist_func = BondiFluxes;
  if (pmbp->pzoom != nullptr && pmbp->pzoom->is_set) {
    pmbp->pzoom->PrintInfo();
    user_ref_func = ZoomAMR;
    if (pmbp->pzoom->zoom_dt) user_dt_func = ZoomNewTimeStep;
  }

  // Read problem-specific parameters from input file
  // global parameters
  bondi.k_adi = pin->GetReal("problem", "k_adi");
  bondi.r_crit = pin->GetReal("problem", "r_crit");

  // Get ideal gas EOS data
  bondi.gm = peos->eos_data.gamma;
  Real gm1 = bondi.gm - 1.0;

  // Parameters
  bondi.temp_min = 1.0e-7;  // lesser temperature root must be greater than this
  bondi.temp_max = 1.0e0;   // greater temperature root must be less than this

  // Get spin of black hole
  bondi.spin = pmbp->pcoord->coord_data.bh_spin;

  // Get excision parameters
  bondi.dexcise = pmbp->pcoord->coord_data.dexcise;
  bondi.pexcise = pmbp->pcoord->coord_data.pexcise;

  // Get initial condition type
  int ic_type = bondi.ic_type = pin->GetOrAddInteger("problem", "ic_type", 0);

  // Get ratio of specific heats
  bondi.n_adi = 1.0/(bondi.gm - 1.0);

  // Prepare various constants for determining primitives
  Real u_crit_sq = 1.0/(2.0*bondi.r_crit);                           // (HSW 71)
  Real u_crit = -sqrt(u_crit_sq);
  Real t_crit = (bondi.n_adi/(bondi.n_adi+1.0)
                 * u_crit_sq/(1.0-(bondi.n_adi+3.0)*u_crit_sq));     // (HSW 74)
  bondi.c1 = pow(t_crit, bondi.n_adi) * u_crit * SQR(bondi.r_crit);  // (HSW 68)
  bondi.c2 = (SQR(1.0 + (bondi.n_adi+1.0) * t_crit)
              * (1.0 - 3.0/(2.0*bondi.r_crit)));                     // (HSW 69)
  bondi.temp_inf = (sqrt(bondi.c2)-1.0)/(1.0+bondi.n_adi);           // (HSW 69)
  bondi.c_s_inf = sqrt(bondi.gm * bondi.temp_inf);
  bondi.r_bondi = 2.0/SQR(bondi.c_s_inf);
  bondi.r_sink = pin->GetReal("problem", "r_sink");
  bondi.d_sink = pin->GetReal("problem", "d_sink");
  bondi.p_sink = pin->GetReal("problem", "p_sink");

  if (ic_type > 0) {
    // Prepare various constants for determining primitives
    bondi.temp_inf = pin->GetReal("problem", "temp_inf");
    bondi.c_s_inf = sqrt(bondi.gm * bondi.temp_inf);
    bondi.rho_inf = pow(bondi.temp_inf/bondi.k_adi, bondi.n_adi);
    bondi.rho_inf = pin->GetOrAddReal("problem", "dens_inf", bondi.rho_inf);
    bondi.pgas_inf = bondi.rho_inf * bondi.temp_inf;
    bondi.r_bondi = 2.0/SQR(bondi.c_s_inf);
    // bondi.r_crit = (5.0-3.0*bondi.gm)/4.0;
    // bondi.c1 = 0.25*pow(2.0/(5.0-3.0*bondi.gm), (5.0-3.0*bondi.gm)*0.5*bondi.n_adi);
    // bondi.c2 = -bondi.n_adi; // useless in Newtonian case
  }

  if (global_variable::my_rank == 0) {
    std::cout << " Bondi radius = " << bondi.r_bondi << std::endl;
    std::cout << " Critical radius = " << bondi.r_crit << std::endl;
    std::cout << " c1 = " << bondi.c1 << std::endl;
  }

  // Spherical Grid for user-defined history
  auto &grids = spherical_grids;
  const Real rflux = 1.0;
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, rflux, 1));
  grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, 1.5*std::pow(2.0,0.5), 1));
  int hist_nr = pin->GetOrAddInteger("problem", "hist_nr", 4);
  Real rmin = pin->GetOrAddReal("problem", "hist_rmin", 3.0);
  Real rmax = pin->GetOrAddReal("problem", "hist_rmax", 0.75*pmy_mesh_->mesh_size.x1max);
  for (int i=0; i<hist_nr-2; i++) {
    Real r_i = std::pow(rmax/rmin,static_cast<Real>(i)/static_cast<Real>(hist_nr-3))*rmin;
    grids.push_back(std::make_unique<SphericalGrid>(pmbp, 10, r_i, 1));
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Spherical grids for user-defined history:" << std::endl;
    std::cout << "  rmin = " << rmin << " rmax = " << rmax << std::endl;
    for (auto &grid : grids) {
      std::cout << "  r = " << grid->radius << std::endl;
    }
  }
  if (restart) return;

  // capture variables for the kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto bondi_ = bondi;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  // Initialize primitive values (HYDRO ONLY)
  // local parameters
  Real pert_amp = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("pgen_bondi", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho, pgas, uu1, uu2, uu3;
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    // TODO: add flat IC
    ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
    // Calculate perturbation
    auto rand_gen = rand_pool64.get_state(); // get random number state this thread
    Real perturbation = 2.0*pert_amp*(rand_gen.frand() - 0.5);
    rand_pool64.free_state(rand_gen);        // free state for use by other threads
    w0_(m,IDN,k,j,i) = rho;
    w0_(m,IEN,k,j,i) = pgas/gm1 * (1.0 + perturbation);
    w0_(m,IM1,k,j,i) = uu1;
    w0_(m,IM2,k,j,i) = uu2;
    w0_(m,IM3,k,j,i) = uu3;
  });

  // Add magnetic field
  Real b_ini = pin->GetOrAddReal("problem", "b_ini", 0.0);
  if (is_mhd && b_ini>0.0) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    par_for("pgen_bondi_bfield", DevExeSpace(), 0,(nmb-1),0,n3m1,0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0_.x1f(m,k,j,i) = 0.0;
      if (i==n1m1) b0_.x1f(m,k,j,i+1) = 0.0;
      b0_.x2f(m,k,j,i) = 0.0;
      if (j==n2m1) b0_.x2f(m,k,j+1,i) = 0.0;
      b0_.x3f(m,k,j,i) = b_ini;
      if (k==n3m1) b0_.x3f(m,k+1,j,i) = b_ini;
      bcc0_(m,IBX,k,j,i) = 0.0;
      bcc0_(m,IBY,k,j,i) = 0.0;
      bcc0_(m,IBZ,k,j,i) = b_ini;
    });
  }

  // Convert primitives to conserved
  auto &u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto &u1_ = is_mhd ? pmbp->pmhd->u1 : pmbp->phydro->u1;
  if (bondi.reset_ic) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u1_, 0, n1m1, 0, n2m1, 0, n3m1);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u1_, 0, n1m1, 0, n2m1, 0, n3m1);
    }
    return;
  }

  // Convert primitives to conserved
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->PrimToCons(w0_, u0_, 0, n1m1, 0, n2m1, 0, n3m1);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, 0, n1m1, 0, n2m1, 0, n3m1);
  }
  // Add turbulence
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 9, "turb_init") == 0) {
      TurbulenceInit *pturb;
      pturb = new TurbulenceInit(it->block_name,pmbp, pin);
      pturb->InitializeModes(1);
      pturb->AddForcing(1);
      delete pturb;
    }
    if (it->block_name.compare(0, 8, "turb_mhd") == 0) {
      TurbulenceMhd *pturb;
      pturb = new TurbulenceMhd(it->block_name,pmbp, pin);
      pturb->InitializeModes(1);
      pturb->AddForcing(1);
      delete pturb;
    }
  }
  // Convert conserved to primitives
  if (pmbp->phydro != nullptr) {
    pmbp->phydro->peos->ConsToPrim(u0_, w0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
    pmbp->phydro->CopyCons(nullptr,1);
  } else if (pmbp->pmhd != nullptr) {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_, b0_, w0_, bcc0_, false, 0, n1m1, 0, n2m1, 0, n3m1);
    pmbp->pmhd->CopyCons(nullptr,1);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates

KOKKOS_INLINE_FUNCTION
static void ComputePrimitiveSingle(Real x1v, Real x2v, Real x3v, CoordData coord,
                                   struct bondi_pgen pgen,
                                   Real& rho, Real& pgas,
                                   Real& uu1, Real& uu2, Real& uu3) {
  if (pgen.ic_type>0) {
    rho = pgen.rho_inf;
    pgas = pgen.pgas_inf;
    uu1 = 0.0;
    uu2 = 0.0;
    uu3 = 0.0;
    return;
  }
}

//----------------------------------------------------------------------------------------
//! \fn FixedBondiInflow
//  \brief Sets boundary condition on surfaces of computational domain
// Note quantities at boundaryies are held fixed to initial condition values

void FixedBondiInflow(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &coord = pmbp->pcoord->coord_data;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int n1m1 = n1 - 1, n2m1 = n2 - 1, n3m1 = n3 - 1;
  int &is = indcs.is;  int &ie  = indcs.ie;
  int &js = indcs.js;  int &je  = indcs.je;
  int &ks = indcs.ks;  int &ke  = indcs.ke;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  int nmb = pmbp->nmb_thispack;

  auto bondi_ = bondi;
  bool is_mhd = (pmbp->pmhd != nullptr);
  auto u0_ = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  auto w0_ = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;

  Real gm1 = bondi.gm - 1.0;
  Real rsink = bondi.r_sink;
  Real dsink = bondi.d_sink;
  Real esink = bondi.p_sink/gm1;
  par_for("initial_radial", DevExeSpace(),0,nmb-1,ks-ng,ke+ng,js-ng,je+ng,is-ng,ie+ng,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rad = sqrt(SQR(x1v)+SQR(x2v)+SQR(x3v));

    if (rad < rsink) {
      u0_(m,IDN,k,j,i) = dsink;
      u0_(m,IEN,k,j,i) = esink;
      u0_(m,IM1,k,j,i) = 0.0;
      u0_(m,IM2,k,j,i) = 0.0;
      u0_(m,IM3,k,j,i) = 0.0;
      w0_(m,IDN,k,j,i) = dsink;
      w0_(m,IEN,k,j,i) = esink;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVX,k,j,i) = 0.0;
      w0_(m,IVX,k,j,i) = 0.0;
    }
  });

  // Primitive boundary conditions
  // X1-Boundary
  // Set X1-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x1", DevExeSpace(),0,(nmb-1),0,n3m1,0,n2m1,
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,is-i-1) = b0.x1f(m,k,j,is);
          b0.x2f(m,k,j,is-i-1) = b0.x2f(m,k,j,is);
          if (j == n2-1) {b0.x2f(m,k,j+1,is-i-1) = b0.x2f(m,k,j+1,is);}
          b0.x3f(m,k,j,is-i-1) = b0.x3f(m,k,j,is);
          if (k == n3-1) {b0.x3f(m,k+1,j,is-i-1) = b0.x3f(m,k+1,j,is);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
        for (int i=0; i<ng; ++i) {
          b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
          b0.x2f(m,k,j,ie+i+1) = b0.x2f(m,k,j,ie);
          if (j == n2-1) {b0.x2f(m,k,j+1,ie+i+1) = b0.x2f(m,k,j+1,ie);}
          b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
          if (k == n3-1) {b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie);}
        }
      }
    });
  }
  // TODO (@mhguo): check whether it should be is or is-1, also in gr_torus problem
  // ConsToPrim over all X1 ghost zones *and* at the innermost/outermost X1-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,is-ng,is,0,n2m1,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,ie,ie+ng,0,n2m1,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,is-ng,is,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,ie,ie+ng,0,n2m1,0,n3m1);
  }
  // Set X1-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("fixed_x1", DevExeSpace(),0,(nmb-1),0,n3m1,0,n2m1,0,(ng-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x1 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x1 boundary
    x1v = CellCenterX((ie+i+1)-is, indcs.nx1, x1min, x1max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_, rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,(ie+i+1)) = rho;
      w0_(m,IEN,k,j,(ie+i+1)) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,(ie+i+1)) = uu1;
      w0_(m,IM2,k,j,(ie+i+1)) = uu2;
      w0_(m,IM3,k,j,(ie+i+1)) = uu3;
    }
  });
  // PrimToCons on X1 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,is-ng,is-1,0,n2m1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,ie+1,ie+ng,0,n2m1,0,n3m1);
  }

  // X2-Boundary
  // Set X2-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x2", DevExeSpace(),0,(nmb-1),0,n3m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int k, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,js-j-1,i) = b0.x1f(m,k,js,i);
          if (i == n1-1) {b0.x1f(m,k,js-j-1,i+1) = b0.x1f(m,k,js,i+1);}
          b0.x2f(m,k,js-j-1,i) = b0.x2f(m,k,js,i);
          b0.x3f(m,k,js-j-1,i) = b0.x3f(m,k,js,i);
          if (k == n3-1) {b0.x3f(m,k+1,js-j-1,i) = b0.x3f(m,k+1,js,i);}
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
        for (int j=0; j<ng; ++j) {
          b0.x1f(m,k,je+j+1,i) = b0.x1f(m,k,je,i);
          if (i == n1-1) {b0.x1f(m,k,je+j+1,i+1) = b0.x1f(m,k,je,i+1);}
          b0.x2f(m,k,je+j+2,i) = b0.x2f(m,k,je+1,i);
          b0.x3f(m,k,je+j+1,i) = b0.x3f(m,k,je,i);
          if (k == n3-1) {b0.x3f(m,k+1,je+j+1,i) = b0.x3f(m,k+1,je,i);}
        }
      }
    });
  }
  // ConsToPrim over all X2 ghost zones *and* at the innermost/outermost X2-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,js-ng,js,0,n3m1);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,je,je+ng,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,js-ng,js,0,n3m1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,je,je+ng,0,n3m1);
  }
  // Set X2-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("fixed_x2", DevExeSpace(),0,(nmb-1),0,n3m1,0,(ng-1),0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x2 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x2 boundary
    x2v = CellCenterX((je+j+1)-js, indcs.nx2, x2min, x2max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,(je+j+1),i) = rho;
      w0_(m,IEN,k,(je+j+1),i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,(je+j+1),i) = uu1;
      w0_(m,IM2,k,(je+j+1),i) = uu2;
      w0_(m,IM3,k,(je+j+1),i) = uu3;
    }
  });
  // PrimToCons on X2 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,js-ng,js-1,0,n3m1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,je+1,je+ng,0,n3m1);
  }

  // X3-Boundary
  // Set X3-BCs on b0 if Meshblock face is at the edge of computational domain
  if (is_mhd) {
    auto &b0 = pmbp->pmhd->b0;
    par_for("noinflow_field_x3", DevExeSpace(),0,(nmb-1),0,n2m1,0,n1m1,
    KOKKOS_LAMBDA(int m, int j, int i) {
      if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ks-k-1,j,i) = b0.x1f(m,ks,j,i);
          if (i == n1-1) {b0.x1f(m,ks-k-1,j,i+1) = b0.x1f(m,ks,j,i+1);}
          b0.x2f(m,ks-k-1,j,i) = b0.x2f(m,ks,j,i);
          if (j == n2-1) {b0.x2f(m,ks-k-1,j+1,i) = b0.x2f(m,ks,j+1,i);}
          b0.x3f(m,ks-k-1,j,i) = b0.x3f(m,ks,j,i);
        }
      }
      if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
        for (int k=0; k<ng; ++k) {
          b0.x1f(m,ke+k+1,j,i) = b0.x1f(m,ke,j,i);
          if (i == n1-1) {b0.x1f(m,ke+k+1,j,i+1) = b0.x1f(m,ke,j,i+1);}
          b0.x2f(m,ke+k+1,j,i) = b0.x2f(m,ke,j,i);
          if (j == n2-1) {b0.x2f(m,ke+k+1,j+1,i) = b0.x2f(m,ke,j+1,i);}
          b0.x3f(m,ke+k+2,j,i) = b0.x3f(m,ke+1,j,i);
        }
      }
    });
  }
  // ConsToPrim over all X3 ghost zones *and* at the innermost/outermost X3-active zones
  // of Meshblocks, even if Meshblock face is not at the edge of computational domain
  if (!is_mhd) {
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ks-ng,ks);
    pmbp->phydro->peos->ConsToPrim(u0_,w0_,false,0,n1m1,0,n2m1,ke,ke+ng);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    auto &b0_ = pmbp->pmhd->b0;
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->ConsToPrim(u0_,b0_,w0_,bcc0_,false,0,n1m1,0,n2m1,ke+1,ke+ng);
  }
  // Set X3-BCs on w0 if Meshblock face is at the edge of computational domain
  par_for("fixed_ix3", DevExeSpace(),0,(nmb-1),0,(ng-1),0,n2m1,0,n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // inner x3 boundary
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real rho, pgas, uu1, uu2, uu3;
    if (mb_bcs.d_view(m,BoundaryFace::inner_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,k,j,i) = rho;
      w0_(m,IEN,k,j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,k,j,i) = uu1;
      w0_(m,IM2,k,j,i) = uu2;
      w0_(m,IM3,k,j,i) = uu3;
    }

    // outer x3 boundary
    x3v = CellCenterX((ke+k+1)-ks, indcs.nx3, x3min, x3max);

    if (mb_bcs.d_view(m,BoundaryFace::outer_x3) == BoundaryFlag::user) {
      ComputePrimitiveSingle(x1v,x2v,x3v,coord,bondi_,rho,pgas,uu1,uu2,uu3);
      w0_(m,IDN,(ke+k+1),j,i) = rho;
      w0_(m,IEN,(ke+k+1),j,i) = pgas/(bondi_.gm - 1.0);
      w0_(m,IM1,(ke+k+1),j,i) = uu1;
      w0_(m,IM2,(ke+k+1),j,i) = uu2;
      w0_(m,IM3,(ke+k+1),j,i) = uu3;
    }
  });
  // PrimToCons on X3 physical boundary ghost zones
  if (!is_mhd) {
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->phydro->peos->PrimToCons(w0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  } else {
    auto &bcc0_ = pmbp->pmhd->bcc0;
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ks-ng,ks-1);
    pmbp->pmhd->peos->PrimToCons(w0_,bcc0_,u0_,0,n1m1,0,n2m1,ke+1,ke+ng);
  }

  if (pm->pmb_pack->pzoom != nullptr && pm->pmb_pack->pzoom->is_set) {
    pm->pmb_pack->pzoom->BoundaryConditions();
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void AddUserSrcs()
//! \brief Add User Source Terms
// NOTE source terms must all be computed using primitive (w0) and NOT conserved (u0) vars
void AddUserSrcs(Mesh *pm, const Real bdt) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  int is_mhd = (pmbp->pmhd != nullptr);
  DvceArray5D<Real> &u0 = is_mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  DvceArray5D<Real> &w0 = is_mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  // capture variables for the kernel
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  int nmb1 = pmbp->nmb_thispack - 1;
  Real rin = bondi.r_sink;
  par_for("accel", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    Real accel = Acceleration(rad);
    if (rad < rin) {
      // Potential = -GM/(r^2+r_a^2*exp(-r^2/r_a^2))^0.5
      Real r_a = 0.5*rin;
      Real fac = exp(-rad*rad/r_a/r_a);
      accel *= rad*rad*rad*(1-fac)/pow(rad*rad+r_a*r_a*fac,1.5);
    }
    Real dmomr = bdt*w0(m,IDN,k,j,i)*accel;
    Real denergy = bdt*w0(m,IDN,k,j,i)*accel/rad*
                  (w0(m,IVX,k,j,i)*x1v+w0(m,IVY,k,j,i)*x2v+w0(m,IVZ,k,j,i)*x3v);
    u0(m,IM1,k,j,i) += dmomr*x1v/rad;
    u0(m,IM2,k,j,i) += dmomr*x2v/rad;
    u0(m,IM3,k,j,i) += dmomr*x3v/rad;
    u0(m,IEN,k,j,i) += denergy;
  });
  return;
}

//----------------------------------------------------------------------------------------
// Function for computing accretion fluxes through constant spherical KS radius surfaces

void BondiFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // set nvars, adiabatic index, primitive array w0, and field array bcc0 if is_mhd
  int nvars; Real gamma; bool is_mhd = false;
  DvceArray5D<Real> w0_, bcc0_;
  if (pmbp->phydro != nullptr) {
    nvars = pmbp->phydro->nhydro + pmbp->phydro->nscalars;
    gamma = pmbp->phydro->peos->eos_data.gamma;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    is_mhd = true;
    nvars = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
    gamma = pmbp->pmhd->peos->eos_data.gamma;
    w0_ = pmbp->pmhd->w0;
    bcc0_ = pmbp->pmhd->bcc0;
  }

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  // int nflux = (is_mhd) ? 4 : 3;
  const int nflux = 30;

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  pdata->nhist = nradii*nflux;
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  // no more than 7 characters per label
  std::string data_label[nflux] = {"r", "out", "m", "mout", "mdot", "mdotout",
      "edot", "edotout", "ekdot", "eidot", "emdot", "epdot", "pdot", "pdotout",
      "lxdot", "lydot", "lzdot", "lx", "ly", "lz", "phi",
      "eint", "v^2", "vr", "vph", "b^2", "br", "bph", "bdotv", "Behyd",
  };
  int gi0 = 0;
  if (pmbp->pzoom != nullptr && pmbp->pzoom->is_set) {
    gi0 = 1;
    pdata->nhist += 1;
    pdata->label[0] = "zone";
    pdata->hdata[0] = (global_variable::my_rank == 0)? pmbp->pzoom->zamr.zone : 0.0;
  }
  for (int g=0; g<nradii; ++g) {
    std::string gstr = std::to_string(g);
    for (int i=0; i<nflux; ++i) {
      pdata->label[gi0+nflux*g+i] = data_label[i] + "_" + gstr;
    }
  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  for (int g=0; g<nradii; ++g) {
    // zero fluxes at this radius
    for (int i=0; i<nflux; ++i) {
      pdata->hdata[gi0+nflux*g+i] = 0.0;
    }

    // interpolate primitives (and cell-centered magnetic fields iff mhd)
    if (is_mhd) {
      grids[g]->InterpolateToSphere(3, bcc0_);
      Kokkos::realloc(interpolated_bcc, grids[g]->nangles, 3);
      Kokkos::deep_copy(interpolated_bcc, grids[g]->interp_vals);
      interpolated_bcc.template modify<DevExeSpace>();
      interpolated_bcc.template sync<HostMemSpace>();
    }
    grids[g]->InterpolateToSphere(nvars, w0_);

    // compute fluxes
    for (int n=0; n<grids[g]->nangles; ++n) {
      // extract coordinate data at this angle
      Real r = grids[g]->radius;
      Real theta = grids[g]->polar_pos.h_view(n,0);
      Real phi = grids[g]->polar_pos.h_view(n,1);
      Real x1 = grids[g]->interp_coord.h_view(n,0);
      Real x2 = grids[g]->interp_coord.h_view(n,1);
      Real x3 = grids[g]->interp_coord.h_view(n,2);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real &int_ie = grids[g]->interp_vals.h_view(n,IEN);

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      Real v_sq = SQR(int_vx)+SQR(int_vy)+SQR(int_vz);
      Real b_sq = SQR(int_bx) + SQR(int_by) + SQR(int_bz);

      // Transform CKS 4-velocity and 4-magnetic field to spherical KS
      Real rad2 = SQR(x1)+SQR(x2)+SQR(x3);
      Real r2 = SQR(r);
      Real sth = sin(theta);
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = x1/r;
      Real drdy = x2/r;
      Real drdz = x3/r;
      // r component of velocity
      Real vr  = drdx *int_vx + drdy *int_vy + drdz *int_vz;
      // phi component of velocity
      Real vph = (x1*int_vy - x2*int_vx)/sqrt(SQR(x1)+SQR(x2));
      // r component of magnetic field (returns zero if not MHD)
      Real br  = drdx *int_bx + drdy *int_by + drdz *int_bz;
      // phi component of magnetic field (returns zero if not MHD)
      Real bph = (x1*int_by - x2*int_bx)/sqrt(SQR(x1)+SQR(x2));
      // phi component of 4-velocity
      // Real u_ph = (-r*sph)*sth*u_1 + (r*cph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      // Real b_ph = (-r*sph)*sth*b_1 + (r*cph)*sth*b_2;
      Real bdotv = int_bx*int_vx + int_by*int_vy + int_bz*int_vz;
      Real e_kin = 0.5*int_dn*v_sq;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      // Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // flags
      Real on = (int_dn != 0.0)? 1.0 : 0.0; // check if angle is on this rank
      Real is_out = (vr>0.0)? 1.0 : 0.0;

      // compute mass flux
      Real mflx = int_dn*vr;

      // compute energy flux
      Real potential = -1.0/r; // gravitational potential
      Real eflx_kin = e_kin*vr;
      Real eflx_int = gamma*int_ie*vr;
      Real eflx_mag = (is_mhd)? b_sq*vr-br*(bdotv) : 0.0;
      Real eflx_pot = int_dn*potential*vr; // gravitational potential energy flux
      Real eflx_hyd = eflx_kin + eflx_int + eflx_pot;
      Real eflx = eflx_hyd + eflx_mag;
      // Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      // compute momentum flux
      Real pflx = int_dn*vr*vr;
      // compute angular momentum flux
      // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
      Real lx = int_dn*(x2*int_vz-x3*int_vy);
      Real ly = int_dn*(x3*int_vx-x1*int_vz);
      Real lz = int_dn*(x1*int_vy-x2*int_vx);
      Real lxflx = lx*vr;
      Real lyflx = ly*vr;
      Real lzflx = lz*vr;
      Real bflx = (is_mhd)? 0.5*fabs(br): 0.0;
      // Real t1_2 = 0.0;
      // Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      // Real phi_flx = (is_mhd) ? 0.5*fabs(br*u0 - b0*ur) : 0.0;
      // Real t1_0_hyd = (int_dn + gamma*int_ie)*ur*u_0;
      // Real bernl_hyd = (on)? -(1.0 + gamma*int_ie/int_dn)*u_0-1.0 : 0.0;
      Real bernl_hyd = (on)? (0.5*v_sq+gamma*int_ie/int_dn+potential) : 0.0;

      Real flux_data[nflux] = {r, is_out, int_dn, int_dn*is_out, mflx, mflx*is_out,
        eflx, eflx*is_out, eflx_kin, eflx_int, eflx_mag, eflx_pot, pflx, pflx*is_out,
        lxflx, lyflx, lzflx, lx, ly, lz, bflx,
        int_ie, v_sq, vr, vph, b_sq, br, bph, bdotv, bernl_hyd,
      };

      pdata->hdata[gi0+nflux*g+0] = (global_variable::my_rank == 0)? flux_data[0] : 0.0;
      for (int i=1; i<nflux; ++i) {
        pdata->hdata[gi0+nflux*g+i] += flux_data[i]*r2*domega*on;
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}

} // namespace
