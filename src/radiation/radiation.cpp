//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.cpp
//! \brief implementation of Radiation class constructor and assorted other functions

#include <float.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "globals.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/coordinates.hpp"
#include "geodesic-grid/geodesic_grid.hpp"
#include "units/units.hpp"
#include "radiation/radiation.hpp"
#include "radiation/radiation_opacities.hpp"

namespace radiation {
namespace {
// opacity_table_diag: one parallel_for calls device TableOpacity (d_view) after h->d sync
void TableOpacityDeviceProbe(Real dens, Real temp, Real density_scale, Real temperature_scale,
                             Real length_scale, bool op_table_use_r, Real k_elec,
                             const DualArray1D<Real> &ross_rho,
                             const DualArray1D<Real> &ross_t,
                             const DualArray1D<Real> &planck_rho,
                             const DualArray1D<Real> &planck_t,
                             const DualArray2D<Real> &ross_table,
                             const DualArray2D<Real> &planck_table,
                             Real &sa, Real &ss, Real &sp) {
  Kokkos::View<Real[3], DevExeSpace> d_sig("opacity_diag_sig");
  Kokkos::parallel_for(
    "OpacityTable diag TableOpacity", Kokkos::RangePolicy<DevExeSpace>(0, 1),
    KOKKOS_LAMBDA(int) {
      Real a = 0.0, s = 0.0, p = 0.0;
      TableOpacity(dens, density_scale, temp, temperature_scale, length_scale, op_table_use_r,
                    ross_rho, ross_t, planck_rho, planck_t, ross_table, planck_table, k_elec,
                    a, s, p);
      d_sig(0) = a;
      d_sig(1) = s;
      d_sig(2) = p;
    });
  Kokkos::fence();
  auto h_sig = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_sig);
  sa = h_sig(0);
  ss = h_sig(1);
  sp = h_sig(2);
}
} // namespace
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
    nh_c("nh_c",1,1),
    nh_f("nh_f",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    na("na",1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("iflx",1,1,1,1,1),
    divfa("divfa",1,1,1,1,1),
    pmy_pack(ppack),
    ross_rho("ross_rho",1),
    ross_t("ross_t",1),
    planck_rho("planck_rho",1),
    planck_t("planck_t",1),
    ross_table("ross_table",1,1),
    planck_table("planck_table",1,1) {
  // Check for general relativity
  if (!(pmy_pack->pcoord->is_general_relativistic)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation requires general relativity" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Check for hydrodynamics, mhd, and units
  is_hydro_enabled = pin->DoesBlockExist("hydro");
  is_mhd_enabled = pin->DoesBlockExist("mhd");
  if (is_hydro_enabled && is_mhd_enabled) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
      << std::endl << "Radiation does not support two fluid calculations, yet "
      << "both <hydro> and <mhd> blocks exist in input file" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  are_units_enabled = pin->DoesBlockExist("units");

  // Check flags and parameters for ad hoc fixes
  correct_radsrc_velocity = pin->GetOrAddBoolean("radiation","correct_radsrc_velocity",false);
  correct_radsrc_opacity  = pin->GetOrAddBoolean("radiation","correct_radsrc_opacity",false);
  dfloor_opacity = pin->GetOrAddReal("radiation","dfloor_opacity",1e-100);
  dens_trunc_max = pin->GetOrAddReal("radiation","dens_trunc_max",1e100);
  tau_truncation = pin->GetOrAddReal("radiation","tau_truncation",1e-100);
  sigmoid_residual = pin->GetOrAddReal("radiation","sigmoid_residual",1e-2);
  sigmoid_residual = fmin(sigmoid_residual, 1./3); // sigmoid residual must be less than 1./3

  // Enable radiation source term (radiation+(M)HD) by default if hydro or mhd enabled
  // Otherwise, disable radiation source term.  The former can be overriden by
  // specification in the input file.
  if (is_hydro_enabled || is_mhd_enabled) {
    rad_source = pin->GetOrAddBoolean("radiation","rad_source",true);
  } else {
    rad_source = false;
  }

  table_opacity = pin->GetOrAddBoolean("radiation","table_opacity",false);
  ross_table_len_x = 0;
  ross_table_len_y = 0;
  planck_table_len_x = 0;
  planck_table_len_y = 0;
  op_table_use_r = false;
  k_elec_opacity = 0.0;

  // Set radiation coupling parameters including scattering and absorption opacities,
  // radiation constant, and source term behavior.
  if (rad_source) {
    kappa_s = pin->GetReal("radiation","kappa_s");
    power_opacity = pin->GetOrAddBoolean("radiation","power_opacity",false);
    if (power_opacity && table_opacity) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<radiation>/table_opacity and power_opacity cannot both be true"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (table_opacity) {
      op_table_use_r = pin->GetOrAddBoolean("radiation","table_use_r",false);
      k_elec_opacity = pin->GetOrAddReal("radiation","k_elec_opacity",0.2);
      ross_table_len_x = pin->GetOrAddInteger("radiation","ross_table_x",0);
      ross_table_len_y = pin->GetOrAddInteger("radiation","ross_table_y",0);
      planck_table_len_x = pin->GetOrAddInteger("radiation","planck_table_x",0);
      planck_table_len_y = pin->GetOrAddInteger("radiation","planck_table_y",0);
      if (ross_table_len_x * ross_table_len_y * planck_table_len_x * planck_table_len_y == 0
          || ross_table_len_x != planck_table_len_x || ross_table_len_y != planck_table_len_y) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Opacity table: require positive ross_table_x/y and planck_table_x/y"
          << " with matching Rosseland and Planck dimensions." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      kappa_a = 0.0;
      kappa_p = 0.0;
    } else if (!(power_opacity)) {
      kappa_a = pin->GetReal("radiation","kappa_a");
      kappa_p = pin->GetReal("radiation","kappa_p");
    }
    is_compton_enabled = pin->GetOrAddBoolean("radiation","compton",false);
    if (is_compton_enabled && !(are_units_enabled)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Compton requires enabling units" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (are_units_enabled) {
      arad = (pmy_pack->punit->rad_constant_cgs*
              SQR(SQR(pmy_pack->punit->temperature_cgs()))/
              pmy_pack->punit->pressure_cgs());
    } else {
      arad = pin->GetReal("radiation","arad");
    }
    affect_fluid = pin->GetOrAddBoolean("radiation","affect_fluid",true);
  }

  // Optional near-horizon radiation energy ceiling. This is intended as a
  // localized safety limiter and is disabled by default.
  erad_ceiling = pin->GetOrAddBoolean("radiation","erad_ceiling",false);
  erad_rho_max = pin->GetOrAddReal("radiation","erad_rho_max",(FLT_MAX));
  erad_max_alpha = pin->GetOrAddReal("radiation","erad_max_alpha",0.8);

  // Check for fluid evolution
  fixed_fluid = pin->GetOrAddBoolean("radiation","fixed_fluid",false);

  // Source terms (if needed)
  if (pin->DoesBlockExist("rad_srcterms")) {
    psrc = new SourceTerms("rad_srcterms", ppack, pin);
  }

  // Setup angular mesh and radiation geometry data
  int nlevel = pin->GetInteger("radiation", "nlevel");
  rotate_geo = pin->GetOrAddBoolean("radiation","rotate_geo",true);
  angular_fluxes = pin->GetOrAddBoolean("radiation","angular_fluxes",true);
  n_0_floor = pin->GetOrAddReal("radiation","n_0_floor",0.1);
  prgeo = new GeodesicGrid(nlevel, rotate_geo, angular_fluxes);

  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(nh_c,prgeo->nangles,4);
  Kokkos::realloc(nh_f,prgeo->nangles,6,4);
  Kokkos::realloc(tet_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tetcov_c,nmb,4,4,ncells3,ncells2,ncells1);
  Kokkos::realloc(tet_d1_x1f,nmb,4,ncells3,ncells2,ncells1+1);
  Kokkos::realloc(tet_d2_x2f,nmb,4,ncells3,ncells2+1,ncells1);
  Kokkos::realloc(tet_d3_x3f,nmb,4,ncells3+1,ncells2,ncells1);
  if (angular_fluxes) {Kokkos::realloc(na,nmb,prgeo->nangles,ncells3,ncells2,ncells1,6);}
  if (is_hydro_enabled || is_mhd_enabled) {
    Kokkos::realloc(norm_to_tet,nmb,4,4,ncells3,ncells2,ncells1);
  }
  }
  SetOrthonormalTetrad();

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for intensities
  {
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(i0,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int nccells1 = indcs.cnx1 + 2*(indcs.ng);
    int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_i0,nmb,prgeo->nangles,nccells3,nccells2,nccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) variables
  pbval_i = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_i->InitializeBuffers(prgeo->nangles);

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // select reconstruction method (default PLM)
    {std::string xorder = pin->GetOrAddString("radiation","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("teno") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      } else if (xorder.compare("teno") == 0) {
        recon_method = ReconstructionMethod::teno;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<radiation> recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    }

    // allocate second registers, fluxes, masks
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(i1,      nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x1f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x2f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    Kokkos::realloc(iflx.x3f,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    if (angular_fluxes) {
      Kokkos::realloc(divfa,nmb,prgeo->nangles,ncells3,ncells2,ncells1);
    }
  }

  if (rad_source && table_opacity) {
    Kokkos::realloc(ross_rho, ross_table_len_x);
    Kokkos::realloc(ross_t, ross_table_len_y);
    Kokkos::realloc(ross_table, ross_table_len_y, ross_table_len_x);
    Kokkos::realloc(planck_rho, planck_table_len_x);
    Kokkos::realloc(planck_t, planck_table_len_y);
    Kokkos::realloc(planck_table, planck_table_len_y, planck_table_len_x);
    LoadOpacityTables(pin);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Radiation::LoadOpacityTables(ParameterInput *pin)
//! \brief Rank 0 reads ASCII opacity tables; MPI_Bcast to all ranks; sync to device.

void Radiation::LoadOpacityTables(ParameterInput *pin) {
  bool opacity_table_diag = pin->GetOrAddBoolean("radiation","opacity_table_diag",false);

  std::string fross = pin->GetOrAddString("radiation","opacity_ross",
                                          "./aveopacity.txt");
  std::string fplanck = pin->GetOrAddString("radiation","opacity_planck",
                                            "./PlanckOpacity.txt");
  std::string flogt = pin->GetOrAddString("radiation","opacity_logt",
                                          "./logT.txt");
  std::string flogr = pin->GetOrAddString("radiation","opacity_logrho",
                                          "./logRhoT.txt");

  int nx = ross_table_len_x;
  int ny = ross_table_len_y;

  if (global_variable::my_rank == 0) {
    FILE *fkappa = std::fopen(fross.c_str(), "r");
    FILE *fpt = std::fopen(flogt.c_str(), "r");
    FILE *fpr = std::fopen(flogr.c_str(), "r");
    FILE *fpk = std::fopen(fplanck.c_str(), "r");
    if (fkappa == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Opacity table: could not open Rosseland table file \"" << fross
        << "\": " << std::strerror(errno) << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (fpt == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Opacity table: could not open log-T axis file \"" << flogt
        << "\": " << std::strerror(errno) << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (fpr == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Opacity table: could not open log-rho axis file \"" << flogr
        << "\": " << std::strerror(errno) << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (fpk == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Opacity table: could not open Planck table file \"" << fplanck
        << "\": " << std::strerror(errno) << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nx; ++i) {
      if (std::fscanf(fpr, "%lf", &(ross_rho.h_view(i))) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Opacity table: error reading log rho axis file." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    for (int i = 0; i < ny; ++i) {
      if (std::fscanf(fpt, "%lf", &(ross_t.h_view(i))) != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Opacity table: error reading log T axis file." << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        if (std::fscanf(fkappa, "%lf", &(ross_table.h_view(j,i))) != 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Opacity table: error reading Rosseland table." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        if (std::fscanf(fpk, "%lf", &(planck_table.h_view(j,i))) != 1) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl << "Opacity table: error reading Planck table." << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
    std::fclose(fkappa);
    std::fclose(fpt);
    std::fclose(fpr);
    std::fclose(fpk);
  }

#if MPI_PARALLEL_ENABLED
  MPI_Bcast(ross_rho.h_view.data(), nx, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(ross_t.h_view.data(), ny, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(ross_table.h_view.data(), nx*ny, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  MPI_Bcast(planck_table.h_view.data(), nx*ny, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  for (int i = 0; i < nx; ++i) {
    planck_rho.h_view(i) = ross_rho.h_view(i);
  }
  for (int i = 0; i < ny; ++i) {
    planck_t.h_view(i) = ross_t.h_view(i);
  }

  if (opacity_table_diag && global_variable::my_rank == 0) {
    Real density_scale = 1.0;
    Real temperature_scale = 1.0;
    Real length_scale = 1.0;
    if (are_units_enabled) {
      density_scale = pmy_pack->punit->density_cgs();
      temperature_scale = pmy_pack->punit->temperature_cgs();
      length_scale = pmy_pack->punit->length_cgs();
    }
    auto rho_h = ross_rho.h_view;
    auto t_h = ross_t.h_view;
    auto rt_h = ross_table.h_view;
    auto pt_h = planck_table.h_view;

    bool mono_rho = true;
    for (int i = 1; i < nx; ++i) {
      if (rho_h(i) <= rho_h(i-1)) {
        mono_rho = false;
        break;
      }
    }
    bool mono_t = true;
    for (int j = 1; j < ny; ++j) {
      if (t_h(j) <= t_h(j-1)) {
        mono_t = false;
        break;
      }
    }
    Real kr_min = rt_h(0,0), kr_max = rt_h(0,0);
    Real kp_min = pt_h(0,0), kp_max = pt_h(0,0);
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        kr_min = std::min(kr_min, rt_h(j,i));
        kr_max = std::max(kr_max, rt_h(j,i));
        kp_min = std::min(kp_min, pt_h(j,i));
        kp_max = std::max(kp_max, pt_h(j,i));
      }
    }

    std::cout << std::endl << "[radiation] opacity_table_diag: loaded opacity tables"
              << std::endl << "  files: ross=\"" << fross << "\" planck=\"" << fplanck << "\""
              << std::endl << "         logrho=\"" << flogr << "\" logT=\"" << flogt << "\""
              << std::endl << "  grid: nx=" << nx << " ny=" << ny
              << "  table_use_r=" << (op_table_use_r ? "true" : "false")
              << "  k_elec_opacity=" << k_elec_opacity
              << std::endl << "  axis log10(rho) [" << rho_h(0) << ", " << rho_h(nx-1) << "]"
              << " strictly increasing: " << (mono_rho ? "yes" : "NO (lookup assumes sorted)")
              << std::endl << "  axis log10(T)   [" << t_h(0) << ", " << t_h(ny-1) << "]"
              << " strictly increasing: " << (mono_t ? "yes" : "NO (lookup assumes sorted)")
              << std::endl << "  Rosseland kappa in table: min=" << kr_min << " max=" << kr_max
              << std::endl << "  Planck kappa in table:    min=" << kp_min << " max=" << kp_max
              << std::endl << "  table corners Rosseland (j,i)=(0,0),(ny-1,nx-1): "
              << rt_h(0,0) << ", " << rt_h(ny-1,nx-1)
              << std::endl << "  table corners Planck:    "
              << pt_h(0,0) << ", " << pt_h(ny-1,nx-1)
              << std::endl << "  units for TableOpacity probe: density_scale=" << density_scale
              << " temperature_scale=" << temperature_scale
              << " length_scale=" << length_scale << std::endl;
  }

  ross_rho.template modify<HostMemSpace>();
  ross_rho.template sync<DevExeSpace>();
  ross_t.template modify<HostMemSpace>();
  ross_t.template sync<DevExeSpace>();
  planck_rho.template modify<HostMemSpace>();
  planck_rho.template sync<DevExeSpace>();
  planck_t.template modify<HostMemSpace>();
  planck_t.template sync<DevExeSpace>();
  ross_table.template modify<HostMemSpace>();
  ross_table.template sync<DevExeSpace>();
  planck_table.template modify<HostMemSpace>();
  planck_table.template sync<DevExeSpace>();

  if (opacity_table_diag && global_variable::my_rank == 0) {
    Real density_scale = 1.0;
    Real temperature_scale = 1.0;
    Real length_scale = 1.0;
    if (are_units_enabled) {
      density_scale = pmy_pack->punit->density_cgs();
      temperature_scale = pmy_pack->punit->temperature_cgs();
      length_scale = pmy_pack->punit->length_cgs();
    }
    auto rho_h = ross_rho.h_view;
    auto t_h = ross_t.h_view;
    Real log_x_mid = 0.5*(rho_h(0) + rho_h(nx-1));
    Real log_t_mid = 0.5*(t_h(0) + t_h(ny-1));
    Real temp_mid = std::pow(10.0, log_t_mid) / temperature_scale;
    Real dens_mid = 0.0;
    if (op_table_use_r) {
      dens_mid = std::pow(10.0, log_x_mid + 3.0*log_t_mid - 18.0) / density_scale;
    } else {
      dens_mid = std::pow(10.0, log_x_mid) / density_scale;
    }
    Real sa = 0.0, ss = 0.0, sp = 0.0;
    TableOpacityDeviceProbe(dens_mid, temp_mid, density_scale, temperature_scale, length_scale,
                            op_table_use_r, k_elec_opacity,
                            ross_rho, ross_t, planck_rho, planck_t, ross_table, planck_table,
                            sa, ss, sp);
    std::cout << std::scientific << std::setprecision(8)
              << "  probe (log-space center, device TableOpacity): dens=" << dens_mid
              << " temp=" << temp_mid
              << " -> sigma_a=" << sa << " sigma_s=" << ss << " sigma_p=" << sp << std::endl;

    Real probe_rho = pin->GetOrAddReal("radiation","opacity_diag_rho",0.0);
    Real probe_temp = pin->GetOrAddReal("radiation","opacity_diag_temp",0.0);
    if (probe_rho > 0.0 && probe_temp > 0.0) {
      TableOpacityDeviceProbe(probe_rho, probe_temp, density_scale, temperature_scale, length_scale,
                              op_table_use_r, k_elec_opacity,
                              ross_rho, ross_t, planck_rho, planck_t, ross_table, planck_table,
                              sa, ss, sp);
      std::cout << std::scientific << std::setprecision(8)
                << "  probe (opacity_diag_rho/temp, device TableOpacity): dens=" << probe_rho
                << " temp=" << probe_temp
                << " -> sigma_a=" << sa << " sigma_s=" << ss << " sigma_p=" << sp << std::endl;
    }
    std::cout << std::defaultfloat;
  }
}

//----------------------------------------------------------------------------------------
// destructor

Radiation::~Radiation() {
  delete pbval_i;
  delete prgeo;
  if (psrc != nullptr) {delete psrc;}
}

} // namespace radiation
