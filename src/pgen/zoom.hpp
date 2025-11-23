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
//! \struct ZoomAMR
//! \brief structure to hold zoom level information

typedef struct ZoomAMR {
  int nlevels;                  // number of levels
  int max_level;                // maximum level number
  int min_level;                // minimum level number
  int level;                    // level number = max_level - zone
  Real radius;                  // radius of inner boundary
  Real runtime;                 // interval for zoom
  bool just_zoomed;             // flag for just zoomed
  bool first_emf;               // flag for first electric field
  bool dump_rst;                // flag for dumping restart file
} ZoomAMR;

typedef struct ZoomInterval {
  Real t_run_fac;               // interval factor
  Real t_run_pow;               // interval power law
  Real t_run_max;               // maximum interval
  Real t_run_fac_zone_0;        // runtime factor for zone 0
  Real t_run_fac_zone_1;        // runtime factor for zone 1
  Real t_run_fac_zone_2;        // runtime factor for zone 2
  Real t_run_fac_zone_3;        // runtime factor for zone 3
  Real t_run_fac_zone_4;        // runtime factor for zone 4
  Real t_run_fac_zone_5;        // runtime factor for zone 5
  Real t_run_fac_zone_6;        // runtime factor for zone 6
  Real t_run_fac_zone_max;      // runtime factor for zone max
} ZoomInterval;

typedef struct ZoomChange {
  Real dvol;                    // mask zone volume
  Real dmass;                   // mass change
  Real dengy;                    // energy change
} ZoomChange;

// runtime parameters, updated each zoom
typedef struct ZoomRun {
  int id;                       // run number
  int zone;                     // zone number = level_max - level
  int last_zone;                // last zone number
  int direction;                // direction of zoom
  Real next_time;               // time of next zoom
} ZoomRun;

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
  bool read_rst;           // flag for reading zoom data restart file
  bool write_rst;          // flag for writing zoom data restart file
  bool zoom_bcs;           // flag for zoom boundary conditions
  bool zoom_ref;           // flag for zoom refinement
  bool zoom_dt;            // flag for zoom time step
  bool fix_efield;         // flag for fixing electric field
  bool dump_diag;          // flag for dumping diagnostic output
  bool calc_cons_change;   // flag for calculating conserved variable change in mask zone
  int ndiag;               // cycles between diagostic output
  int mzoom;               // number of zoom meshblocks
  int nleaf;               // number of zoom meshblocks on each level
  int nvars;               // number of variables
  int nflux;               // number of fluxes through spherical surfaces
  int emf_flag;            // flag for modifying electric field
  Real r_in;               // radius of iner boundary
  Real d_zoom;             // density within inner boundary
  Real p_zoom;             // pressure within inner boundary
  Real emf_f0, emf_f1;     // electric field factor, e = f0 * e0 + f1 * e1
  Real emf_fmax;           // maximum electric field factor
  int  emf_zmax;           // maximum zone number for electric field
  Real re_fac;             // factor for electric field
  Real r0_efld;            // modify e if r < r0_efld

  ZoomAMR zamr;            // zoom AMR parameters
  ZoomInterval zint;       // zoom interval parameters
  ZoomChange zchg;         // zoom change of conserved variables in mask zone
  ZoomRun zrun;            // zoom time parameters

  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  
  DvceArray5D<Real> coarse_u0;  // coarse conserved variables
  DvceArray5D<Real> coarse_w0;  // coarse primitive variables
  // DvceArray5D<Real> coarse_wuh; // coarse primitive variables from hydro conserved variables

  // following only used for time-evolving flow
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  DvceEdgeFld4D<Real> emf0;   // edge-centered electric fields just after zoom
  DvceEdgeFld4D<Real> delta_efld; // change in electric fields

  HostArray2D<Real> max_emf0;  // maximum electric field

  // fluxes through spherical surfaces
  HostArray3D<Real> zoom_fluxes;

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  HostArray5D<Real> harr_5d;  // host copy of 5D arrays
  HostArray4D<Real> harr_4d;  // host copy of 4D arrays

  // array_sum::GlobalSum nc1, nc2, nc3, em1, em2, em3;

  // functions
  void Initialize();
  void Update(const bool restart);
  void PrintInfo();
  void BoundaryConditions();
  void AMR();
  void SetInterval();
  void DumpData();
  void RefineCondition();
  void UpdateVariables();
  void UpdateHydroVariables(int zm, int m);
  void SyncVariables();
  void UpdateGhostVariables();
  void ApplyVariables();
  void FixEField(DvceEdgeFld4D<Real> emf);
  void MeanEField(DvceEdgeFld4D<Real> emf);
  void AddEField(DvceEdgeFld4D<Real> emf);
  void AddDeltaEField(DvceEdgeFld4D<Real> emf);
  void UpdateDeltaEField(DvceEdgeFld4D<Real> emf);
  void SyncZoomEField(DvceEdgeFld4D<Real> emf, int zid);
  void SetMaxEField();
  void SphericalFlux(int n, int g);
  Real NewTimeStep(Mesh* pm);
  Real GRTimeStep(Mesh* pm);
  Real EMFTimeStep(Mesh* pm);
  IOWrapperSizeT RestartFileSize();
  void WriteRestartFile(IOWrapper &resfile);
  void ReadRestartFile(IOWrapper &resfile);

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
};

} // namespace zoom
#endif // PGEN_ZOOM_HPP_