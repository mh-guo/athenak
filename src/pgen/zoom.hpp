#ifndef PGEN_ZOOM_HPP_
#define PGEN_ZOOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom.hpp
//  \brief definitions for Zoom class

#include "parameter_input.hpp"

class Zoom;

class Zoom
{
 private:
  /* data */
 public:
  Zoom(MeshBlockPack *ppack, ParameterInput *pin);
  ~Zoom() = default;

  // data
  bool is_set;
  Real r_in;               // radius of iner boundary
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary
  // functions
  void ZoomBoundaryConditions();

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
};

#endif // PGEN_ZOOM_HPP_