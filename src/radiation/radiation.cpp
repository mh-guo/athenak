//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.cpp
//! \brief implementation of Radiation class constructor and assorted other functions

#include <float.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
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

namespace radiation {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Radiation::Radiation(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    i0("i0",1,1,1,1,1),
    i1("i1",1,1,1,1,1),
    iflx("iflx",1,1,1,1,1),
    divfa("divfa",1,1,1,1,1),
    nh_c("nh_c",1,1),
    nh_f("nh_f",1,1,1),
    tet_c("tet_c",1,1,1,1,1,1),
    tetcov_c("tetcov_c",1,1,1,1,1,1),
    tet_d1_x1f("tet_d1_x1f",1,1,1,1,1),
    tet_d2_x2f("tet_d2_x2f",1,1,1,1,1),
    tet_d3_x3f("tet_d3_x3f",1,1,1,1,1),
    na("na",1,1,1,1,1,1),
    norm_to_tet("norm_to_tet",1,1,1,1,1,1),
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
}

//----------------------------------------------------------------------------------------
// destructor

Radiation::~Radiation() {
  delete pbval_i;
  delete prgeo;
  if (psrc != nullptr) {delete psrc;}
}

} // namespace radiation
