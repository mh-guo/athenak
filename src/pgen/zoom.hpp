#ifndef PGEN_ZOOM_HPP_
#define PGEN_ZOOM_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file zoom.hpp
//  \brief definitions for Zoom class

#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"

namespace zoom {

//----------------------------------------------------------------------------------------
//! \class Zoom

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

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // functions
  void ZoomBoundaryConditions();
  void ZoomRefine();

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
};

} // namespace zoom
#endif // PGEN_ZOOM_HPP_