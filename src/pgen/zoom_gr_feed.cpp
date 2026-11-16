//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom_gr_feed.cpp
//! \brief Cold uniform reservoir feeding a BH in Cartesian Kerr-Schild coordinates.
//! Hydro/MHD with optional radiation. No imposed rotation or mass injection rate.
//! External gas is held at rho0,T0 only on inflowing boundary faces; winds can leave.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_opacities.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "units/units.hpp"
#include "pgen/pgen.hpp"
#include "pgen/turb_seed.hpp"

namespace {
Real feed_rho, feed_eint;

KOKKOS_INLINE_FUNCTION
Real FeedNormalVelocity(Real x, Real y, Real z, CoordData coord, int axis,
                        Real ux, Real uy, Real uz) {
  Real gl[4][4], gu[4][4];
  ComputeMetricAndInverse(x,y,z,coord.is_minkowski,coord.bh_spin,gl,gu);
  Real q = gl[1][1]*ux*ux + gl[2][2]*uy*uy + gl[3][3]*uz*uz
         + 2.0*(gl[1][2]*ux*uy + gl[1][3]*ux*uz + gl[2][3]*uy*uz);
  Real alpha = sqrt(-1.0/gu[0][0]);
  return ((axis == 1) ? ux : ((axis == 2) ? uy : uz))
         - alpha*sqrt(1.0+q)*gu[0][axis];
}
} // namespace

void FeedBoundary(Mesh *pm);
void FeedFluxes(HistoryData *pdata, Mesh *pm);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic || pmbp->pdyngr != nullptr ||
      pmbp->punit == nullptr || (pmbp->phydro == nullptr && pmbp->pmhd == nullptr) ||
      (pmbp->phydro != nullptr && pmbp->pmhd != nullptr) || !pmy_mesh_->three_d) {
    std::cout << "### FATAL ERROR: zoom_gr_feed requires 3D fixed GR, units, and "
              << "either hydro or MHD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool mhd = (pmbp->pmhd != nullptr);
  auto *peos = mhd ? pmbp->pmhd->peos : pmbp->phydro->peos;
  if (!peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR: zoom_gr_feed requires an ideal-gas EOS" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gm1 = peos->eos_data.gamma - 1.0;
  feed_rho = pin->GetOrAddReal("problem", "dens_norm", 1.0);
  const Real temp_k = pin->GetOrAddReal("problem", "temperature", 3000.0);
  const Real temp = temp_k/pmbp->punit->temperature_cgs();
  if (!(feed_rho > 0.0 && temp_k > 0.0 && gm1 > 0.0)) {
    std::cout << "### FATAL ERROR: positive density, temperature and gamma>1 required"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  feed_eint = feed_rho*temp/gm1;
  user_bcs_func = FeedBoundary;
  user_hist_func = FeedFluxes;
  const int nlevels = pin->GetOrAddInteger("problem", "hist_nlevels", 3);
  const int nintp = pin->GetOrAddInteger("problem", "hist_nintp", 2);
  const Real rmin = pin->GetOrAddReal("problem", "hist_rmin", 3.0);
  const Real rmax = pin->GetOrAddReal("problem", "hist_rmax",
                                    0.75*pmy_mesh_->mesh_size.x1max);
  const Real dr = pin->GetOrAddReal("problem", "hist_dr", 1.0);
  if (!(rmin > 0.0 && rmax >= rmin && dr > 0.0)) {
    std::cout << "### FATAL ERROR: invalid history radii/spacing" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Optional torus-style inner surfaces; the horizon sample can touch excision.
  if (pin->GetOrAddBoolean("problem", "hist_inner_surfaces", false)) {
    const Real spin = pmbp->pcoord->coord_data.bh_spin;
    const Real rh = 1.0 + std::sqrt(1.0 - SQR(spin));
    const Real rout = 1.5*std::sqrt(2.0);
    if (!(rmin > rout)) {
      std::cout << "### FATAL ERROR: inner history surfaces require hist_rmin > "
                << rout << std::endl;
      std::exit(EXIT_FAILURE);
    }
    spherical_grids.push_back(std::make_unique<SphericalGrid>(
        pmbp,nlevels,rh,nintp));
    spherical_grids.push_back(std::make_unique<SphericalGrid>(
        pmbp,nlevels,rout,nintp));
  }
  const int nr = static_cast<int>(std::log2(rmax/rmin)/dr)+1;
  for (int n=0; n<nr; ++n) {
    spherical_grids.push_back(std::make_unique<SphericalGrid>(
        pmbp,nlevels,rmin*std::pow(2.0,n*dr),nintp));
  }
  if (restart) return;

  const auto indcs = pmy_mesh_->mb_indcs;
  const auto coord = pmbp->pcoord->coord_data;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = indcs.nx1+2*indcs.ng;
  const int n2 = indcs.nx2+2*indcs.ng;
  const int n3 = indcs.nx3+2*indcs.ng;
  auto w = mhd ? pmbp->pmhd->w0 : pmbp->phydro->w0;
  auto u = mhd ? pmbp->pmhd->u0 : pmbp->phydro->u0;
  const Real rho0 = feed_rho, eint0 = feed_eint;
  auto mask = pmbp->pcoord->excision_floor;
  Kokkos::deep_copy(w,0.0);
  par_for("feed_ic",DevExeSpace(),0,nmb-1,0,n3-1,0,n2-1,0,n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const bool excised = coord.bh_excise && mask(m,k,j,i);
    w(m,IDN,k,j,i) = excised ? coord.dexcise : rho0;
    w(m,IEN,k,j,i) = excised ? coord.pexcise/gm1 : eint0;
  });
  // Explicitly zero all face and cell B, including faces not covered by seed windows.
  if (mhd) {
    Kokkos::deep_copy(pmbp->pmhd->b0.x1f,0.0);
    Kokkos::deep_copy(pmbp->pmhd->b0.x2f,0.0);
    Kokkos::deep_copy(pmbp->pmhd->b0.x3f,0.0);
    Kokkos::deep_copy(pmbp->pmhd->bcc0,0.0);
  }
  if (pmbp->prad != nullptr) Kokkos::deep_copy(pmbp->prad->i0,0.0);
  for (auto it=pin->block.begin(); it!=pin->block.end(); ++it) {
    if (it->block_name.compare(0,9,"turb_seed") == 0) {
      // Zero radiation naturally gives gas-pressure normalization.
      TurbSeed seed(it->block_name,pmbp,pin);
      seed.Apply();
    }
  }
  if (mhd) {
    peos->PrimToCons(w,pmbp->pmhd->bcc0,u,0,n1-1,0,n2-1,0,n3-1);
  } else {
    peos->PrimToCons(w,u,0,n1-1,0,n2-1,0,n3-1);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "Cold reservoir: rho=" << rho0 << " ("
              << rho0*pmbp->punit->density_cgs() << " g/cm^3), T=" << temp_k
              << " K, initial radiation=0" << std::endl;
  }
}

// The same operations apply to all six box faces. Indices below are ordered x,y,z;
// array accesses reverse them to k,j,i. Only ghost cells/faces are overwritten.
void FeedBoundary(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  const auto ind = pm->mb_indcs;
  const int nc[3] = {ind.nx1+2*ind.ng, ind.nx2+2*ind.ng, ind.nx3+2*ind.ng};
  const int first[3] = {ind.is, ind.js, ind.ks};
  const int last[3] = {ind.ie, ind.je, ind.ke};
  const int nmb = pack->nmb_thispack;
  const bool mhd = (pack->pmhd != nullptr);
  auto *eos = mhd ? pack->pmhd->peos : pack->phydro->peos;
  auto w = mhd ? pack->pmhd->w0 : pack->phydro->w0;
  auto u = mhd ? pack->pmhd->u0 : pack->phydro->u0;
  const int nvar = w.extent_int(1);
  auto bcs = pack->pmb->mb_bcs;
  auto size = pack->pmb->mb_size;
  const auto coord = pack->pcoord->coord_data;
  const Real rho0 = feed_rho, eint0 = feed_eint;

  for (int axis=0; axis<3; ++axis) {
    const int t1 = (axis+1)%3, t2 = (axis+2)%3;
    for (int side=0; side<2; ++side) {
      const int face = 2*axis+side;
      const int sign = (side == 0) ? -1 : 1;
      const int edge = (side == 0) ? first[axis] : last[axis];

      // Extrapolate face B as in NoInflowTorus. Do not overwrite the normal
      // field on the physical boundary itself; CT evolves that face.
      if (mhd) {
        for (int component=0; component<3; ++component) {
          auto field = (component == 0) ? pack->pmhd->b0.x1f :
                       ((component == 1) ? pack->pmhd->b0.x2f : pack->pmhd->b0.x3f);
          const int src_edge = edge + ((side == 1 && component == axis) ? 1 : 0);
          const int nt1 = nc[t1]+(component == t1 ? 1 : 0);
          const int nt2 = nc[t2]+(component == t2 ? 1 : 0);
          par_for("feed_field",DevExeSpace(),0,nmb-1,0,nt2-1,0,nt1-1,
          KOKKOS_LAMBDA(int m, int b, int a) {
            if (bcs.d_view(m,face) != BoundaryFlag::user) return;
            int src[3];
            src[axis] = src_edge; src[t1] = a; src[t2] = b;
            for (int g=1; g<=ind.ng; ++g) {
              int dst[3] = {src[0],src[1],src[2]};
              dst[axis] += sign*g;
              field(m,dst[2],dst[1],dst[0]) = field(m,src[2],src[1],src[0]);
            }
          });
        }
      }

      // Refresh primitives and bcc on the ghost slab plus its adjacent active layer.
      int lo[3] = {0,0,0}, hi[3] = {nc[0]-1,nc[1]-1,nc[2]-1};
      lo[axis] = (side == 0) ? edge-ind.ng : edge;
      hi[axis] = (side == 0) ? edge : edge+ind.ng;
      if (mhd) {
        eos->ConsToPrim(u,pack->pmhd->b0,w,pack->pmhd->bcc0,false,
                       lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
      } else {
        eos->ConsToPrim(u,w,false,lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
      }
      par_for("feed_fluid",DevExeSpace(),0,nmb-1,0,nc[t2]-1,0,nc[t1]-1,
      KOKKOS_LAMBDA(int m, int b, int a) {
        if (bcs.d_view(m,face) != BoundaryFlag::user) return;
        int src[3];
        src[axis] = edge; src[t1] = a; src[t2] = b;
        Real x = CellCenterX(src[0]-ind.is,ind.nx1,size.d_view(m).x1min,
                            size.d_view(m).x1max);
        Real y = CellCenterX(src[1]-ind.js,ind.nx2,size.d_view(m).x2min,
                            size.d_view(m).x2max);
        Real z = CellCenterX(src[2]-ind.ks,ind.nx3,size.d_view(m).x3min,
                            size.d_view(m).x3max);
        Real un = FeedNormalVelocity(x,y,z,coord,axis+1,
            w(m,IVX,src[2],src[1],src[0]),w(m,IVY,src[2],src[1],src[0]),
            w(m,IVZ,src[2],src[1],src[0]));
        const bool out = sign*un > 0.0;
        for (int g=1; g<=ind.ng; ++g) {
          int dst[3] = {src[0],src[1],src[2]};
          dst[axis] += sign*g;
          for (int n=0; n<nvar; ++n) {
            w(m,n,dst[2],dst[1],dst[0]) = out ? w(m,n,src[2],src[1],src[0]) :
                ((n == IDN) ? rho0 : ((n == IEN) ? eint0 : 0.0));
          }
        }
      });

      // Radiation direction is independent of the fluid flow sign.
      if (pack->prad != nullptr) {
        auto intensity = pack->prad->i0;
        auto tet = pack->prad->tet_c;
        auto nh = pack->prad->nh_c;
        const int nang = pack->prad->prgeo->nangles;
        par_for("feed_radiation",DevExeSpace(),0,nmb-1,0,nang-1,0,nc[t2]-1,0,nc[t1]-1,
        KOKKOS_LAMBDA(int m, int n, int b, int a) {
          if (bcs.d_view(m,face) != BoundaryFlag::user) return;
          int src[3];
          src[axis] = edge; src[t1] = a; src[t2] = b;
          for (int g=1; g<=ind.ng; ++g) {
            int dst[3] = {src[0],src[1],src[2]};
            dst[axis] += sign*g;
            Real ndir = 0.0;
            for (int d=0; d<4; ++d) {
              ndir += tet(m,d,axis+1,dst[2],dst[1],dst[0])*nh.d_view(n,d);
            }
            intensity(m,n,dst[2],dst[1],dst[0]) = (sign*ndir > 0.0) ?
                intensity(m,n,src[2],src[1],src[0]) : 0.0;
          }
        });
      }
      // Convert only the ghost slab back to conserved variables.
      if (side == 0) hi[axis] = edge-1;
      else lo[axis] = edge+1;
      if (mhd) {
        eos->PrimToCons(w,pack->pmhd->bcc0,u,lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
      } else {
        eos->PrimToCons(w,u,lo[0],hi[0],lo[1],hi[1],lo[2],hi[2]);
      }
    }
  }
}

// Spherical-shell diagnostics, following the existing torus column conventions.
void FeedFluxes(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  // extract BH parameters
  bool &flat = pmbp->pcoord->coord_data.is_minkowski;
  Real &spin = pmbp->pcoord->coord_data.bh_spin;

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

  // Calculate conversion for P to e if using DynGRMHD.
  Real to_ien = 1.;
  if (pmbp->pdyngr != nullptr) {
    to_ien = 1.0 / (gamma - 1.);
  }

  // History row: nbase gas/MHD values, then optional 6 radiation values.
  static constexpr int nbase = 29;
  static constexpr int nradmom = 10;  // upper-triangle R^{mu nu}
  static constexpr int nraddiag = nradmom + 3;  // plus cell-centered opacities

  // extract grids, number of radii, number of fluxes, and history appending index
  auto &grids = pm->pgen->spherical_grids;
  int nradii = grids.size();
  const bool is_radiation_enabled = (pmbp->prad != nullptr);
  const int nflux = nbase + (is_radiation_enabled ? 6 : 0);
  int nang1 = 0;
  if (is_radiation_enabled) { nang1 = pmbp->prad->prgeo->nangles - 1; }

  // set number of and names of history variables for hydro or mhd
  //  (1) mass accretion rate
  //  (2) energy flux
  //  (3) angular momentum flux
  //  (4) magnetic flux (iff MHD)
  // Radiation: integrals of E_com, R^r_t, and F^r u_t + u^r F_t;
  // then solid-angle means of sigma_a,s,p (inverse code length).
  // Outward Killing-energy powers are minus the stored mixed-index fluxes.
  pdata->nhist = nradii*nflux + (pm->pzoom != nullptr ? 1 : 0);
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "User history function specified pdata->nhist larger than"
              << " NHISTORY_VARIABLES" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Short physical-quantity labels; the radius index is appended below.
  std::vector<std::string> data_label = {
    "r", "out", "m", "mout", "mdot", "mdotout", "edot", "edotout", "lx", "ly", "lz",
    "lzout", "phi", "eint", "b^2", "alpha", "lor", "u0", "u_0", "ur", "uph", "b0",
    "b_0", "br", "bph", "edothyd", "edho", "edotadv", "edao"
  };
  if (is_radiation_enabled) {
    data_label.push_back("r00_ff");
    data_label.push_back("redot");
    data_label.push_back("redot_ff");
    data_label.push_back("sigma_a");
    data_label.push_back("sigma_s");
    data_label.push_back("sigma_p");
  }
  int gi0 = 0;
  if (pm->pzoom != nullptr) {
    gi0 = 1;
    pdata->label[0] = "zone";
    pdata->hdata[0] = (global_variable::my_rank == 0)? pm->pzoom->zstate.zone : 0.0;
  }
  for (int g=0; g<nradii; ++g) {
    std::string gstr = std::to_string(g);
    for (int i=0; i<nflux; ++i) {
      pdata->label[gi0+nflux*g+i] = data_label[i] + "_" + gstr;
    }
  }

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  DvceArray5D<Real> rad_moments;
  if (is_radiation_enabled) {
    Kokkos::realloc(rad_moments, nmb, nraddiag, n3, n2, n1);
    auto rad_moments_ = rad_moments;
    auto i0_ = pmbp->prad->i0;
    auto tet_c_ = pmbp->prad->tet_c;
    auto tetcov_c_ = pmbp->prad->tetcov_c;
    auto nh_c_ = pmbp->prad->nh_c;
    auto solid_angles_ = pmbp->prad->prgeo->solid_angles;
    par_for("accretion_rad_moments", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real n0 = tet_c_(m,0,0,k,j,i);
      Real moments[nradmom] = {};
      // Transform each direction once, then accumulate all ten tensor components.
      for (int ord=0; ord<=nang1; ++ord) {
        Real nc[4] = {}, n_0 = 0.0;
        for (int d=0; d<4; ++d) {
          for (int mu=0; mu<4; ++mu) {
            nc[mu] += tet_c_(m,d,mu,k,j,i)*nh_c_.d_view(ord,d);
          }
          n_0 += tetcov_c_(m,d,0,k,j,i)*nh_c_.d_view(ord,d);
        }
        if (n0*n_0 == 0.0) continue;
        const Real weight = i0_(m,ord,k,j,i)/(n0*n_0)*solid_angles_.d_view(ord);
        for (int mu=0, n12=0; mu<4; ++mu) {
          for (int nu=mu; nu<4; ++nu, ++n12) {
            moments[n12] += nc[mu]*nc[nu]*weight;
          }
        }
      }
      for (int n=0; n<nradmom; ++n) rad_moments_(m,n,k,j,i) = moments[n];
      // No coupling coefficients are defined when rad_source is disabled.
      for (int n=nradmom; n<nraddiag; ++n) rad_moments_(m,n,k,j,i) = 0.0;
    });
    if (pmbp->prad->rad_source) {
      // Evaluate the same cell-centered sigma as rad_fluid_sigma, including ghosts
      // needed by the sphere interpolation stencil.
      auto &size = pmbp->pmb->mb_size;
      const int is = indcs.is, js = indcs.js, ks = indcs.ks;
      auto excision_flux_ = pmbp->pcoord->excision_flux;
      auto prad = pm->pmb_pack->prad;
      Real density_scale_ = 1.0, temperature_scale_ = 1.0, length_scale_ = 1.0;
      Real mean_mol_weight_ = 1.0;
      Real rosseland_coef_ = 1.0, planck_minus_rosseland_coef_ = 0.0;
      if (prad->are_units_enabled) {
        density_scale_ = pm->pmb_pack->punit->density_cgs();
        temperature_scale_ = pm->pmb_pack->punit->temperature_cgs();
        length_scale_ = pm->pmb_pack->punit->length_cgs();
        mean_mol_weight_ = pm->pmb_pack->punit->mu();
        rosseland_coef_ = pm->pmb_pack->punit->rosseland_coef_cgs;
        planck_minus_rosseland_coef_ = pm->pmb_pack->punit->planck_minus_rosseland_coef_cgs;
      }

      Real gm1, dfloor;
      if (pm->pmb_pack->phydro != nullptr) {
        gm1 = pm->pmb_pack->phydro->peos->eos_data.gamma - 1.0;
        dfloor = pm->pmb_pack->phydro->peos->eos_data.dfloor;
      } else {
        gm1 = pm->pmb_pack->pmhd->peos->eos_data.gamma - 1.0;
        dfloor = pm->pmb_pack->pmhd->peos->eos_data.dfloor;
      }

      const bool is_mhd_pack = is_mhd;
      bool correct_radsrc_opacity_ = prad->correct_radsrc_opacity;
      Real dfloor_op_ = prad->dfloor_opacity;
      Real dtrunc_max_ = prad->dens_trunc_max;
      Real tau_trunc_ = prad->tau_truncation;
      Real sigmoid_res_ = prad->sigmoid_residual;
      Real kappa_s_ = prad->kappa_s;
      bool table_opacity_ = prad->table_opacity;
      bool power_opacity_ = prad->power_opacity;
      bool op_table_use_r_ = prad->op_table_use_r;
      Real k_elec_opacity_ = prad->k_elec_opacity;
      auto ross_rho_ = prad->ross_rho;
      auto ross_t_ = prad->ross_t;
      auto planck_rho_ = prad->planck_rho;
      auto planck_t_ = prad->planck_t;
      auto ross_table_ = prad->ross_table;
      auto planck_table_ = prad->planck_table;
      Real kappa_a_ = prad->kappa_a;
      Real kappa_p_ = prad->kappa_p;

      par_for("feed_history_sigmas", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
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

        Real glower[4][4], gupper[4][4];
        ComputeMetricAndInverse(x1v,x2v,x3v,flat,spin,glower,gupper);

        Real &wdn = w0_(m,IDN,k,j,i);
        Real &wvx = w0_(m,IVX,k,j,i);
        Real &wvy = w0_(m,IVY,k,j,i);
        Real &wvz = w0_(m,IVZ,k,j,i);
        Real &wen = w0_(m,IEN,k,j,i);

        Real bccx = 0.0, bccy = 0.0, bccz = 0.0;
        if (is_mhd_pack) {
          bccx = bcc0_(m,IBX,k,j,i);
          bccy = bcc0_(m,IBY,k,j,i);
          bccz = bcc0_(m,IBZ,k,j,i);
        }

        Real sigma_a = 0.0, sigma_s = 0.0, sigma_p = 0.0;
        radiation::RadiationFluidCellSigmas(
            wdn, wvx, wvy, wvz, wen*to_ien, gm1, glower, gupper,
            excision_flux_, m, k, j, i,
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3,
            correct_radsrc_opacity_, is_mhd_pack,
            bccx, bccy, bccz,
            dfloor, dfloor_op_, dtrunc_max_, tau_trunc_, sigmoid_res_, kappa_s_,
            table_opacity_, power_opacity_,
            density_scale_, temperature_scale_, length_scale_,
            op_table_use_r_,
            ross_rho_, ross_t_, planck_rho_, planck_t_, ross_table_, planck_table_,
            k_elec_opacity_,
            mean_mol_weight_, rosseland_coef_, planck_minus_rosseland_coef_,
            kappa_a_, kappa_p_,
            sigma_a, sigma_s, sigma_p);

        rad_moments_(m,nradmom,k,j,i) = sigma_a;
        rad_moments_(m,nradmom+1,k,j,i) = sigma_s;
        rad_moments_(m,nradmom+2,k,j,i) = sigma_p;
      });
    }

  }

  // go through angles at each radii:
  DualArray2D<Real> interpolated_bcc;  // needed for MHD
  DualArray2D<Real> interpolated_rad_moments;
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
    if (is_radiation_enabled) {
      grids[g]->InterpolateToSphere(nraddiag, rad_moments);
      Kokkos::realloc(interpolated_rad_moments, grids[g]->nangles, nraddiag);
      Kokkos::deep_copy(interpolated_rad_moments, grids[g]->interp_vals);
      interpolated_rad_moments.template modify<DevExeSpace>();
      interpolated_rad_moments.template sync<HostMemSpace>();
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
      Real glower[4][4], gupper[4][4];
      ComputeMetricAndInverse(x1,x2,x3,flat,spin,glower,gupper);

      // extract interpolated primitives
      Real &int_dn = grids[g]->interp_vals.h_view(n,IDN);
      Real &int_vx = grids[g]->interp_vals.h_view(n,IVX);
      Real &int_vy = grids[g]->interp_vals.h_view(n,IVY);
      Real &int_vz = grids[g]->interp_vals.h_view(n,IVZ);
      Real int_ie = grids[g]->interp_vals.h_view(n,IEN)*to_ien;

      // extract interpolated field components (iff is_mhd)
      Real int_bx = 0.0, int_by = 0.0, int_bz = 0.0;
      if (is_mhd) {
        int_bx = interpolated_bcc.h_view(n,IBX);
        int_by = interpolated_bcc.h_view(n,IBY);
        int_bz = interpolated_bcc.h_view(n,IBZ);
      }

      // Compute interpolated u^\mu in CKS
      Real q = glower[1][1]*int_vx*int_vx + 2.0*glower[1][2]*int_vx*int_vy +
               2.0*glower[1][3]*int_vx*int_vz + glower[2][2]*int_vy*int_vy +
               2.0*glower[2][3]*int_vy*int_vz + glower[3][3]*int_vz*int_vz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor/alpha;
      Real u1 = int_vx - alpha * lor * gupper[0][1];
      Real u2 = int_vy - alpha * lor * gupper[0][2];
      Real u3 = int_vz - alpha * lor * gupper[0][3];

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
      Real sph = sin(phi);
      Real cph = cos(phi);
      Real drdx = r*x1/(2.0*r2 - rad2 + a2);
      Real drdy = r*x2/(2.0*r2 - rad2 + a2);
      Real drdz = (r*x3 + a2*x3/r)/(2.0*r2-rad2+a2);
      Real dphdx = (-x2/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdx);
      Real dphdy = ( x1/(SQR(x1)+SQR(x2)) + (spin/(r2 + a2))*drdy);
      Real dphdz = (spin/(r2 + a2)*drdz);
      // contravariant r component of 4-velocity
      Real ur  = drdx *u1 + drdy *u2 + drdz *u3;
      // contravariant r component of 4-magnetic field (returns zero if not MHD)
      Real br  = drdx *b1 + drdy *b2 + drdz *b3;
      // phi component of 4-velocity in spherical KS
      Real uph = dphdx*u1 + dphdy*u2 + dphdz*u3;
      // phi component of 4-magnetic field in spherical KS
      Real bph = dphdx*b1 + dphdy*b2 + dphdz*b3;
      // covariant phi component of 4-velocity
      Real u_ph = (-r*sph-spin*cph)*sth*u_1 + (r*cph-spin*sph)*sth*u_2;
      // covariant phi component of 4-magnetic field (returns zero if not MHD)
      Real b_ph = (-r*sph-spin*cph)*sth*b_1 + (r*cph-spin*sph)*sth*b_2;

      // integration params
      Real &domega = grids[g]->solid_angles.h_view(n);
      Real sqrtmdet = (r2+SQR(spin*cos(theta)));

      // flags
      // assuming you always have density on this rank
      Real on = (int_dn != 0.0)? 1.0 : 0.0; // check if angle is on this rank
      Real is_out = (ur>0.0)? 1.0 : 0.0;

      // compute mass flux
      Real m_flx = int_dn*ur;

      // compute energy flux
      Real t1_0 = (int_dn + gamma*int_ie + b_sq)*ur*u_0 - br*b_0;
      // compute angular momentum flux
      // TODO(@mhguo): write a correct function to compute x,y angular momentum flux
      Real t1_1 = 0.0;
      Real t1_2 = 0.0;
      Real t1_3 = (int_dn + gamma*int_ie + b_sq)*ur*u_ph - br*b_ph;
      Real phi_flx = (is_mhd) ? 0.5*fabs(br*u0 - b0*ur) : 0.0;
      Real t1_0_hyd = (int_dn + gamma*int_ie)*ur*u_0;
      Real bernl_hyd = (on)? -(1.0 + gamma*int_ie/int_dn)*u_0-1.0 : 0.0;

      // Coordinate-frame radiation moments are computed cell-centered on device,
      // interpolated to the sphere, then contracted with the surface radial direction.
      Real r00_ff = 0.0, redot = 0.0, redot_ff = 0.0;
      if (is_radiation_enabled) {
        Real rup[4][4];
        rup[0][0] = interpolated_rad_moments.h_view(n,0);
        rup[0][1] = interpolated_rad_moments.h_view(n,1);
        rup[0][2] = interpolated_rad_moments.h_view(n,2);
        rup[0][3] = interpolated_rad_moments.h_view(n,3);
        rup[1][1] = interpolated_rad_moments.h_view(n,4);
        rup[1][2] = interpolated_rad_moments.h_view(n,5);
        rup[1][3] = interpolated_rad_moments.h_view(n,6);
        rup[2][2] = interpolated_rad_moments.h_view(n,7);
        rup[2][3] = interpolated_rad_moments.h_view(n,8);
        rup[3][3] = interpolated_rad_moments.h_view(n,9);
        rup[1][0] = rup[0][1];
        rup[2][0] = rup[0][2];
        rup[3][0] = rup[0][3];
        rup[2][1] = rup[1][2];
        rup[3][1] = rup[1][3];
        rup[3][2] = rup[2][3];

        // r^r_0: same radial alignment as gas t1_0 (u^r via dr/dx^i u^i), not Cartesian u^1.
        // r^{r nu} = (dr/dx^i) r^{i nu} for i=1,2,3; static CKS uses r=r(x,y,z) with dr/dt=0.
        // r^r_0 = g_{0 nu} r^{r nu} (nu contravariant on r); sum over nu.
        redot = 0.0;
        for (int nu=0; nu<4; ++nu) {
          const Real rrnu = drdx*rup[1][nu] + drdy*rup[2][nu] + drdz*rup[3][nu];
          redot += glower[0][nu]*rrnu;
        }
        // Exact fluid-frame projection, signature (-+++):
        // E = R^{mu nu} u_mu u_nu; F^mu = -R^{mu nu}u_nu - E u^mu.
        // R^r_t = (E u^r u_t + P^r_t) + (F^r u_t + u^r F_t).
        const Real ucov[4] = {u_0, u_1, u_2, u_3};
        const Real ucon[4] = {u0, u1, u2, u3};
        Real ru[4] = {}, fcon[4];
        for (int mu=0; mu<4; ++mu) {
          for (int nu=0; nu<4; ++nu) ru[mu] += rup[mu][nu]*ucov[nu];
          r00_ff += ucov[mu]*ru[mu];
        }
        Real f_t = 0.0;
        for (int mu=0; mu<4; ++mu) {
          fcon[mu] = -ru[mu] - r00_ff*ucon[mu];
          f_t += glower[0][mu]*fcon[mu];
        }
        const Real fr = drdx*fcon[1] + drdy*fcon[2] + drdz*fcon[3];
        redot_ff = fr*u_0 + ur*f_t;
      }

      // gas/MHD: same order as data_label (0 .. nbase-1)
      const Real base_flux[nbase] = {
        r, is_out, int_dn, int_dn*is_out, m_flx, m_flx*is_out,
        t1_0, t1_0*is_out, t1_1, t1_2, t1_3, t1_3*is_out, phi_flx,
        int_ie, b_sq, alpha, lor, u0, u_0, ur, uph, b0, b_0, br, bph,
        t1_0_hyd, t1_0_hyd*is_out, bernl_hyd, bernl_hyd*is_out
      };

      pdata->hdata[gi0+nflux*g+0] = (global_variable::my_rank == 0) ? base_flux[0] : 0.0;
      for (int i=1; i<nbase; ++i) {
        pdata->hdata[gi0+nflux*g+i] += base_flux[i]*sqrtmdet*domega*on;
      }
      if (is_radiation_enabled) {
        pdata->hdata[gi0+nflux*g+nbase+0] += r00_ff*sqrtmdet*domega*on;
        pdata->hdata[gi0+nflux*g+nbase+1] += redot*sqrtmdet*domega*on;
        pdata->hdata[gi0+nflux*g+nbase+2] += redot_ff*sqrtmdet*domega*on;
        for (int q=0; q<3; ++q) {
          pdata->hdata[gi0+nflux*g+nbase+3+q] +=
              interpolated_rad_moments.h_view(n,nradmom+q)*domega*on/(4.0*M_PI);
        }
      }
    }
  }

  // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }

  return;
}
