#ifndef PGEN_RESTART_REGRID_HPP_
#define PGEN_RESTART_REGRID_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file restart_regrid.hpp
//! \brief restart regrid helpers

#include "outputs/io_wrapper.hpp"

class Mesh;
class ParameterInput;

class RestartRegrid {
 public:
  static bool IsEnabled(ParameterInput *pin);
  static IOWrapperSizeT LoadAndProlongate(ParameterInput *pin, Mesh *pm, IOWrapper &resfile,
                                          bool single_file_per_rank=false);
};

#endif // PGEN_RESTART_REGRID_HPP_
