//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_regrid.cpp
//! \brief restart regrid helpers

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/prolongation.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/adm.hpp"
#include "z4c/z4c.hpp"
#include "radiation/radiation.hpp"
#include "srcterms/turb_driver.hpp"
#include "restart_regrid.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

void RestartRegridFail(const char *msg) {
  std::cout << "### FATAL ERROR in restart regrid" << std::endl
            << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

void ReadCellCentered(IOWrapper &resfile, Mesh *pm, HostArray5D<Real> &host,
                      const int nvar, const int nout3, const int nout2,
                      const int nout1, const IOWrapperSizeT data_size,
                      const IOWrapperSizeT section_offset,
                      const bool single_file_per_rank, const char *label) {
  int nmb = pm->pmb_pack->nmb_thispack;
  Kokkos::realloc(host, nmb, nvar, nout3, nout2, nout1);

  int noutmbs_max = pm->nmb_eachrank[0];
  int noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<global_variable::nranks; ++i) {
    noutmbs_max = std::max(noutmbs_max, pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min, pm->nmb_eachrank[i]);
  }

  IOWrapperSizeT myoffset = section_offset;
  if (!single_file_per_rank) {
    myoffset += data_size*pm->gids_eachrank[global_variable::my_rank];
  }
  for (int m=0; m<noutmbs_max; ++m) {
    if (m < noutmbs_min) {
      auto mbptr = Kokkos::subview(host, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at_all(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
          != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC " << label
                  << " data not read correctly from rst file, restart file is broken."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      myoffset += data_size;
    } else if (m < pm->nmb_thisrank) {
      auto mbptr = Kokkos::subview(host, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                                   Kokkos::ALL);
      int mbcnt = mbptr.size();
      if (resfile.Read_Reals_at(mbptr.data(), mbcnt, myoffset, single_file_per_rank)
          != mbcnt) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "CC " << label
                  << " data not read correctly from rst file, restart file is broken."
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      myoffset += data_size;
    }
  }
}

void ReadFaceCentered(IOWrapper &resfile, Mesh *pm, HostFaceFld4D<Real> &host,
                      const int nout3, const int nout2, const int nout1,
                      const IOWrapperSizeT data_size,
                      const IOWrapperSizeT section_offset,
                      const bool single_file_per_rank) {
  int nmb = pm->pmb_pack->nmb_thispack;
  Kokkos::realloc(host.x1f, nmb, nout3, nout2, nout1+1);
  Kokkos::realloc(host.x2f, nmb, nout3, nout2+1, nout1);
  Kokkos::realloc(host.x3f, nmb, nout3+1, nout2, nout1);

  int noutmbs_max = pm->nmb_eachrank[0];
  int noutmbs_min = pm->nmb_eachrank[0];
  for (int i=0; i<global_variable::nranks; ++i) {
    noutmbs_max = std::max(noutmbs_max, pm->nmb_eachrank[i]);
    noutmbs_min = std::min(noutmbs_min, pm->nmb_eachrank[i]);
  }

  IOWrapperSizeT myoffset = section_offset;
  if (!single_file_per_rank) {
    myoffset += data_size*pm->gids_eachrank[global_variable::my_rank];
  }
  for (int m=0; m<noutmbs_max; ++m) {
    if (m < noutmbs_min) {
      auto x1fptr = Kokkos::subview(host.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      int fldcnt = x1fptr.size();
      if (resfile.Read_Reals_at_all(x1fptr.data(), fldcnt, myoffset,
                                    single_file_per_rank) != fldcnt) {
        RestartRegridFail("Input b0.x1f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);

      auto x2fptr = Kokkos::subview(host.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x2fptr.size();
      if (resfile.Read_Reals_at_all(x2fptr.data(), fldcnt, myoffset,
                                    single_file_per_rank) != fldcnt) {
        RestartRegridFail("Input b0.x2f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);

      auto x3fptr = Kokkos::subview(host.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x3fptr.size();
      if (resfile.Read_Reals_at_all(x3fptr.data(), fldcnt, myoffset,
                                    single_file_per_rank) != fldcnt) {
        RestartRegridFail("Input b0.x3f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);
      myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
    } else if (m < pm->nmb_thisrank) {
      auto x1fptr = Kokkos::subview(host.x1f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      int fldcnt = x1fptr.size();
      if (resfile.Read_Reals_at(x1fptr.data(), fldcnt, myoffset, single_file_per_rank)
          != fldcnt) {
        RestartRegridFail("Input b0.x1f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);

      auto x2fptr = Kokkos::subview(host.x2f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x2fptr.size();
      if (resfile.Read_Reals_at(x2fptr.data(), fldcnt, myoffset, single_file_per_rank)
          != fldcnt) {
        RestartRegridFail("Input b0.x2f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);

      auto x3fptr = Kokkos::subview(host.x3f, m, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      fldcnt = x3fptr.size();
      if (resfile.Read_Reals_at(x3fptr.data(), fldcnt, myoffset, single_file_per_rank)
          != fldcnt) {
        RestartRegridFail("Input b0.x3f field not read correctly from rst file.");
      }
      myoffset += fldcnt*sizeof(Real);
      myoffset += data_size-(x1fptr.size()+x2fptr.size()+x3fptr.size())*sizeof(Real);
    }
  }
}

void ProlongateCellCentered(Mesh *pm, DvceArray5D<Real> &coarse,
                            DvceArray5D<Real> &fine) {
  int nmb = pm->pmb_pack->nmb_thispack;
  int nvar = fine.extent_int(1);
  auto &indcs = pm->mb_indcs;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  bool &multi_d = pm->multi_d;
  bool &three_d = pm->three_d;

  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmb*nvar, Kokkos::AUTO);
  Kokkos::parallel_for("RestartRegridProlongCC", policy,
  KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - m*nvar);
    const int ni = cie - cis + 1;
    const int nj = cje - cjs + 1;
    const int nk = cke - cks + 1;
    const int nkji = nk*nj*ni;
    const int nji = nj*ni;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
      int k = idx/nji;
      int j = (idx - k*nji)/ni;
      int i = (idx - k*nji - j*ni) + cis;
      k += cks;
      j += cjs;
      int fi = 2*i - cis;
      int fj = 2*j - cjs;
      int fk = 2*k - cks;
      ProlongCC(m, v, k, j, i, fk, fj, fi, multi_d, three_d, coarse, fine);
    });
  });
}

void ProlongateFaceCentered(Mesh *pm, DvceFaceFld4D<Real> &coarse,
                            DvceFaceFld4D<Real> &fine) {
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &indcs = pm->mb_indcs;
  auto &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &cis = indcs.cis, &cie = indcs.cie;
  auto &cjs = indcs.cjs, &cje = indcs.cje;
  auto &cks = indcs.cks, &cke = indcs.cke;
  bool &multi_d = pm->multi_d;
  bool &three_d = pm->three_d;

  par_for("RestartRegridProlongFC1", DevExeSpace(), 0, nmb-1, cks, cke, cjs, cje,
  cis, cie+1, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int fi = (i - cis)*2 + is;
    int fj = (multi_d)? ((j - cjs)*2 + js) : j;
    int fk = (three_d)? ((k - cks)*2 + ks) : k;
    ProlongFCSharedX1Face(m, k, j, i, fk, fj, fi, multi_d, three_d,
                          coarse.x1f, fine.x1f);
  });

  par_for("RestartRegridProlongFC2", DevExeSpace(), 0, nmb-1, cks, cke, cjs, cje+1,
  cis, cie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int fi = (i - cis)*2 + is;
    int fj = (multi_d)? ((j - cjs)*2 + js) : j;
    int fk = (three_d)? ((k - cks)*2 + ks) : k;
    ProlongFCSharedX2Face(m, k, j, i, fk, fj, fi, three_d, coarse.x2f, fine.x2f);
  });

  par_for("RestartRegridProlongFC3", DevExeSpace(), 0, nmb-1, cks, cke+1, cjs, cje,
  cis, cie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int fi = (i - cis)*2 + is;
    int fj = (multi_d)? ((j - cjs)*2 + js) : j;
    int fk = (three_d)? ((k - cks)*2 + ks) : k;
    ProlongFCSharedX3Face(m, k, j, i, fk, fj, fi, multi_d, coarse.x3f, fine.x3f);
  });

  bool &one_d = pm->one_d;
  par_for("RestartRegridProlongFCInt", DevExeSpace(), 0, nmb-1, cks, cke, cjs, cje,
  cis, cie, KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    int fi = (i - cis)*2 + is;
    int fj = (j - cjs)*2 + js;
    int fk = (k - cks)*2 + ks;
    if (one_d) {
      fine.x1f(m, fk, fj, fi+1) =
          0.5*(fine.x1f(m, fk, fj, fi) + fine.x1f(m, fk, fj, fi+2));
    } else {
      ProlongFCInternal(m, fk, fj, fi, three_d, fine);
    }
  });
}

} // namespace

bool RestartRegrid::IsEnabled(ParameterInput *pin) {
  return pin->DoesBlockExist("restart_regrid") &&
         pin->GetOrAddBoolean("restart_regrid", "regrid", false);
}

IOWrapperSizeT RestartRegrid::LoadAndProlongate(ParameterInput *pin, Mesh *pm,
                                                IOWrapper &resfile,
                                                bool single_file_per_rank) {
  int factor = pin->GetOrAddInteger("restart_regrid", "factor", 2);
  if (factor != 2) {
    RestartRegridFail("<restart_regrid>/factor currently must be 2.");
  }
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int old_nx1 = indcs.nx1/factor;
  int old_nx2 = (indcs.nx2 > 1)? (indcs.nx2/factor) : 1;
  int old_nx3 = (indcs.nx3 > 1)? (indcs.nx3/factor) : 1;
  int nout1 = old_nx1 + 2*(indcs.ng);
  int nout2 = (old_nx2 > 1)? (old_nx2 + 2*(indcs.ng)) : 1;
  int nout3 = (old_nx3 > 1)? (old_nx3 + 2*(indcs.ng)) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;

  hydro::Hydro* phydro = pm->pmb_pack->phydro;
  mhd::MHD* pmhd = pm->pmb_pack->pmhd;
  adm::ADM* padm = pm->pmb_pack->padm;
  z4c::Z4c* pz4c = pm->pmb_pack->pz4c;
  radiation::Radiation* prad = pm->pmb_pack->prad;
  TurbulenceDriver* pturb = pm->pmb_pack->pturb;

  if (pz4c != nullptr || padm != nullptr) {
    RestartRegridFail("Z4c/ADM restart regrid is not implemented yet.");
  }

  int nhydro = 0, nmhd = 0, nrad = 0, nforce = 3;
  if (phydro != nullptr) {
    nhydro = phydro->nhydro + phydro->nscalars;
  }
  if (pmhd != nullptr) {
    nmhd = pmhd->nmhd + pmhd->nscalars;
  }
  if (prad != nullptr) {
    nrad = prad->prgeo->nangles;
  }

  if (pturb != nullptr) {
    char rng_data[sizeof(RNG_State)];
    if (global_variable::my_rank == 0 || single_file_per_rank) {
      if (resfile.Read_bytes(rng_data, 1, sizeof(RNG_State), single_file_per_rank)
          != sizeof(RNG_State)) {
        RestartRegridFail("RNG data size read from restart file is incorrect.");
      }
    }
#if MPI_PARALLEL_ENABLED
    if (!single_file_per_rank) {
      MPI_Bcast(rng_data, sizeof(RNG_State), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
    std::memcpy(&(pturb->rstate), &(rng_data[0]), sizeof(RNG_State));
  }

  IOWrapperSizeT variablesize = sizeof(IOWrapperSizeT);
  char *variabledata = new char[variablesize];
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    if (resfile.Read_bytes(variabledata, 1, variablesize, single_file_per_rank)
        != variablesize) {
      RestartRegridFail("Variable data size read from restart file is incorrect.");
    }
  }
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Bcast(variabledata, variablesize, MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif
  IOWrapperSizeT data_size;
  std::memcpy(&data_size, &(variabledata[0]), sizeof(IOWrapperSizeT));
  delete [] variabledata;

  IOWrapperSizeT headeroffset;
  if (global_variable::my_rank == 0 || single_file_per_rank) {
    headeroffset = resfile.GetPosition(single_file_per_rank);
  }
#if MPI_PARALLEL_ENABLED
  if (!single_file_per_rank) {
    MPI_Bcast(&headeroffset, sizeof(IOWrapperSizeT), MPI_CHAR, 0, MPI_COMM_WORLD);
  }
#endif

  IOWrapperSizeT data_size_expected = 0;
  if (phydro != nullptr) {
    data_size_expected += nout1*nout2*nout3*nhydro*sizeof(Real);
  }
  if (pmhd != nullptr) {
    data_size_expected += nout1*nout2*nout3*nmhd*sizeof(Real);
    data_size_expected += (nout1+1)*nout2*nout3*sizeof(Real);
    data_size_expected += nout1*(nout2+1)*nout3*sizeof(Real);
    data_size_expected += nout1*nout2*(nout3+1)*sizeof(Real);
  }
  if (prad != nullptr) {
    data_size_expected += nout1*nout2*nout3*nrad*sizeof(Real);
  }
  if (pturb != nullptr) {
    data_size_expected += nout1*nout2*nout3*nforce*sizeof(Real);
  }
  if (data_size_expected != data_size) {
    RestartRegridFail("Restart data size does not match 2x regrid expectations.");
  }

  HostArray5D<Real> ccin("rst-regrid-cc-in", 1, 1, 1, 1, 1);
  HostFaceFld4D<Real> fcin("rst-regrid-fc-in", 1, 1, 1, 1);
  DvceArray5D<Real> coarse_cc("rst-regrid-coarse-cc", 1, 1, 1, 1, 1);
  DvceFaceFld4D<Real> coarse_fc("rst-regrid-coarse-fc", 1, 1, 1, 1);

  IOWrapperSizeT section_offset = headeroffset;
  if (phydro != nullptr) {
    ReadCellCentered(resfile, pm, ccin, nhydro, nout3, nout2, nout1, data_size,
                     section_offset, single_file_per_rank, "hydro");
    Kokkos::realloc(coarse_cc, nmb, nhydro, nout3, nout2, nout1);
    Kokkos::deep_copy(coarse_cc, ccin);
    ProlongateCellCentered(pm, coarse_cc, phydro->u0);
    section_offset += nout1*nout2*nout3*nhydro*sizeof(Real);
  }

  if (pmhd != nullptr) {
    ReadCellCentered(resfile, pm, ccin, nmhd, nout3, nout2, nout1, data_size,
                     section_offset, single_file_per_rank, "mhd");
    Kokkos::realloc(coarse_cc, nmb, nmhd, nout3, nout2, nout1);
    Kokkos::deep_copy(coarse_cc, ccin);
    ProlongateCellCentered(pm, coarse_cc, pmhd->u0);
    section_offset += nout1*nout2*nout3*nmhd*sizeof(Real);

    ReadFaceCentered(resfile, pm, fcin, nout3, nout2, nout1, data_size,
                     section_offset, single_file_per_rank);
    Kokkos::realloc(coarse_fc.x1f, nmb, nout3, nout2, nout1+1);
    Kokkos::realloc(coarse_fc.x2f, nmb, nout3, nout2+1, nout1);
    Kokkos::realloc(coarse_fc.x3f, nmb, nout3+1, nout2, nout1);
    Kokkos::deep_copy(coarse_fc.x1f, fcin.x1f);
    Kokkos::deep_copy(coarse_fc.x2f, fcin.x2f);
    Kokkos::deep_copy(coarse_fc.x3f, fcin.x3f);
    ProlongateFaceCentered(pm, coarse_fc, pmhd->b0);
    section_offset += (nout1+1)*nout2*nout3*sizeof(Real);
    section_offset += nout1*(nout2+1)*nout3*sizeof(Real);
    section_offset += nout1*nout2*(nout3+1)*sizeof(Real);
  }

  if (prad != nullptr) {
    ReadCellCentered(resfile, pm, ccin, nrad, nout3, nout2, nout1, data_size,
                     section_offset, single_file_per_rank, "rad");
    Kokkos::realloc(coarse_cc, nmb, nrad, nout3, nout2, nout1);
    Kokkos::deep_copy(coarse_cc, ccin);
    ProlongateCellCentered(pm, coarse_cc, prad->i0);
    section_offset += nout1*nout2*nout3*nrad*sizeof(Real);
  }

  if (pturb != nullptr) {
    ReadCellCentered(resfile, pm, ccin, nforce, nout3, nout2, nout1, data_size,
                     section_offset, single_file_per_rank, "turb");
    Kokkos::realloc(coarse_cc, nmb, nforce, nout3, nout2, nout1);
    Kokkos::deep_copy(coarse_cc, ccin);
    ProlongateCellCentered(pm, coarse_cc, pturb->force);
  }

  Kokkos::fence();
  pin->SetBoolean("restart_regrid", "regrid", false);
  return headeroffset + data_size * pm->nmb_total;
}
