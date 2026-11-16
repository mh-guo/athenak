//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
// Sample active-cell integrals around individual operations. RK scopes subtract
// the RK combination, so their delta is the integrated numerical flux update.
// Other scopes measure direct U modification. Nested source scopes overlap;
// they must not be added to their enclosing all_sources scope.
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "mesh/mesh.hpp"
#include "cyclic_zoom/cyclic_zoom.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
namespace zoom_diagnostics {
namespace {
struct Sum {
  using value_type = Totals;
  DvceArray5D<Real> u, u1;
  DualArray1D<RegionSize> size;
  int nx, ny, nz, is, js, ks;
  Real r, cx, cy, cz, g0, g1;
  KOKKOS_INLINE_FUNCTION void init(value_type &v) const { v = {}; }
  KOKKOS_INLINE_FUNCTION void join(value_type &a, const value_type &b) const {
    for (int j = 0; j < 4; ++j)
      for (int n = 0; n < 5; ++n) a.value[j][n] += b.value[j][n];
  }
  KOKKOS_INLINE_FUNCTION void operator()(int idx, value_type &v) const {
    int i = idx % nx + is;
    idx /= nx;
    int j = idx % ny + js;
    idx /= ny;
    int k = idx % nz + ks, m = idx / nz;
    auto s = size.d_view(m);
    Real x = CellCenterX(i - is, nx, s.x1min, s.x1max) - cx;
    Real y = CellCenterX(j - js, ny, s.x2min, s.x2max) - cy;
    Real z = CellCenterX(k - ks, nz, s.x3min, s.x3max) - cz;
    Real rr = (x * x + y * y + z * z) / (r * r);
    if (rr >= 256.) return;
    int bin = rr < 0.16 ? 0 : rr < 1. ? 1 : rr < 4. ? 2 : 3;
    Real vol = s.dx1 * s.dx2 * s.dx3;
    for (int n = 0; n < 5; ++n) {
      Real val = g0 * u(m, n, k, j, i);
      if (g1 != 0.) val += g1 * u1(m, n, k, j, i);
      // U_IEN = T^t_t + D; negate to measure energy excluding rest mass.
      v.value[bin][n] += (n == IEN ? -1. : 1.) * val * vol;
    }
  }
};
Totals measure(Mesh *pm, Real radius, Real g0, Real g1) {
  auto pack = pm->pmb_pack;
  auto ind = pm->mb_indcs;
  auto zr = pm->pzoom->zregion;
  Sum f;
  f.u = pack->pmhd->u0;
  f.u1 = pack->pmhd->u1;
  f.size = pack->pmb->mb_size;
  f.nx = ind.nx1;
  f.ny = ind.nx2;
  f.nz = ind.nx3;
  f.is = ind.is;
  f.js = ind.js;
  f.ks = ind.ks;
  f.r = radius;
  f.cx = zr.x1c;
  f.cy = zr.x2c;
  f.cz = zr.x3c;
  f.g0 = g0;
  f.g1 = g1;
  Totals result;
  Kokkos::parallel_reduce(
      "zoom_budget",
      Kokkos::RangePolicy<>(DevExeSpace(), 0,
                            pack->nmb_thispack * ind.nx1 * ind.nx2 * ind.nx3),
      f, result);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &result.value[0][0], 20, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  return result;
}
}  // namespace
BudgetScope::BudgetScope(Mesh *pm, const char *label, int stage, Real g0, Real g1)
    : mesh_(pm), label_(label), stage_(stage), enabled_(false), radius_(0), before_{} {
  static const int interval = []() {
    const char *s = std::getenv("ATHENA_ZOOM_BUDGET");
    return s ? std::atoi(s) : 0;
  }();
  if (interval <= 0 || pm->pzoom == nullptr || pm->pmb_pack->pmhd == nullptr ||
      !pm->pmb_pack->pcoord->is_general_relativistic)
    return;
  // Initial/regrid C2P is always measured. Evolution is sampled by cycle.
  if (stage != 0 && pm->ncycle % interval != 0) return;
  const char *r = std::getenv("ATHENA_ZOOM_BUDGET_RADIUS");
  radius_ = r ? std::atof(r) : pm->pzoom->zregion.radius;
  if (!(radius_ > 0.) || !std::isfinite(radius_)) return;
  enabled_ = true;
  before_ = measure(pm, radius_, g0, g1);
}
BudgetScope::~BudgetScope() {
  if (!enabled_) return;
  Totals after = measure(mesh_, radius_, 1., 0.);
  if (global_variable::my_rank != 0) return;
  std::ostringstream out;
  out << std::scientific << std::setprecision(17);
  for (int b = 0; b < 4; ++b) {
    out << "DBG_BUDGET cycle=" << mesh_->ncycle << " time=" << mesh_->time
        << " dt=" << mesh_->dt << " zone=" << mesh_->pzoom->zstate.zone
        << " stage=" << stage_ << " op=" << label_ << " radius=" << radius_
        << " bin=" << b;
    for (int n = 0; n < 5; ++n)
      out << " before" << n << "=" << before_.value[b][n] << " after" << n << "="
          << after.value[b][n] << " delta" << n << "="
          << after.value[b][n] - before_.value[b][n];
    out << "\n";
  }
  std::cout << out.str() << std::flush;
}
}  // namespace zoom_diagnostics

// Optional GRMHD diagnostics: measure the exterior interface, not the frozen core.
namespace {
struct ShellResult {
  Real max_t, max_b2rho, max_divb, mass, energy, magnetic, thermal;
  Real x, y, z, rho, lor, b2, gid, ii, jj, kk, edge_cells;
  int bad, cells;
};
struct ShellFunctor {
  using value_type = ShellResult;
  DvceArray5D<Real> w, u;
  DvceArray4D<Real> b1, b2f, b3;
  DualArray1D<RegionSize> size;
  int is, js, ks, nx, ny, nz, gid0;
  Real radius, gm1, spin, cx, cy, cz;
  bool flat;
  KOKKOS_INLINE_FUNCTION void init(value_type &v) const { v = {}; }
  KOKKOS_INLINE_FUNCTION void join(value_type &a, const value_type &v) const {
    if (v.max_t > a.max_t) {
      a.max_t = v.max_t;
      a.x = v.x;
      a.y = v.y;
      a.z = v.z;
      a.rho = v.rho;
      a.lor = v.lor;
      a.b2 = v.b2;
      a.gid = v.gid;
      a.ii = v.ii;
      a.jj = v.jj;
      a.kk = v.kk;
      a.edge_cells = v.edge_cells;
    }
    a.max_b2rho = fmax(a.max_b2rho, v.max_b2rho);
    a.max_divb = fmax(a.max_divb, v.max_divb);
    a.mass += v.mass;
    a.energy += v.energy;
    a.magnetic += v.magnetic;
    a.thermal += v.thermal;
    a.bad += v.bad;
    a.cells += v.cells;
  }
  KOKKOS_INLINE_FUNCTION void operator()(int idx, value_type &v) const {
    int i = idx % nx + is;
    idx /= nx;
    int j = idx % ny + js;
    idx /= ny;
    int k = idx % nz + ks, m = idx / nz;
    auto s = size.d_view(m);
    Real x = CellCenterX(i - is, nx, s.x1min, s.x1max);
    Real y = CellCenterX(j - js, ny, s.x2min, s.x2max);
    Real z = CellCenterX(k - ks, nz, s.x3min, s.x3max);
    Real r2 = SQR(x - cx) + SQR(y - cy) + SQR(z - cz);
    if (r2 <= radius * radius || r2 > 2.25 * radius * radius) return;
    Real bx = 0.5 * (b1(m, k, j, i) + b1(m, k, j, i + 1));
    Real by = 0.5 * (b2f(m, k, j, i) + b2f(m, k, j + 1, i));
    Real bz = 0.5 * (b3(m, k, j, i) + b3(m, k + 1, j, i));
    Real rho = w(m, IDN, k, j, i), eint = w(m, IEN, k, j, i);
    Real vx = w(m, IVX, k, j, i), vy = w(m, IVY, k, j, i), vz = w(m, IVZ, k, j, i);
    Real b2 = bx * bx + by * by + bz * bz, t = gm1 * eint / rho;
    v.cells++;
    if (rho <= 0.0 || eint < 0.0 || !Kokkos::isfinite(rho) || !Kokkos::isfinite(eint) ||
        !Kokkos::isfinite(u(m, IDN, k, j, i)) || !Kokkos::isfinite(t) ||
        !Kokkos::isfinite(b2) || !Kokkos::isfinite(vx + vy + vz) ||
        !Kokkos::isfinite(u(m, IEN, k, j, i))) {
      v.bad++;
      return;
    }
    Real div = (b1(m, k, j, i + 1) - b1(m, k, j, i)) / s.dx1 +
               (b2f(m, k, j + 1, i) - b2f(m, k, j, i)) / s.dx2 +
               (b3(m, k + 1, j, i) - b3(m, k, j, i)) / s.dx3;
    Real dx = fmin(s.dx1, fmin(s.dx2, s.dx3));
    v.max_divb = fmax(v.max_divb, fabs(div) * dx / (sqrt(b2) + 1.e-100));
    v.max_b2rho = fmax(v.max_b2rho, b2 / rho);  // coordinate B^2/rho, not comoving sigma
    Real vol = s.dx1 * s.dx2 * s.dx3;
    v.mass += u(m, IDN, k, j, i) * vol;
    v.energy += u(m, IEN, k, j, i) * vol;
    v.magnetic += 0.5 * b2 * vol;
    v.thermal += eint * vol;
    if (t > v.max_t) {
      Real gl[4][4], gu[4][4];
      ComputeMetricAndInverse(x, y, z, flat, spin, gl, gu);
      Real q = gl[1][1] * vx * vx + gl[2][2] * vy * vy + gl[3][3] * vz * vz +
               2 * (gl[1][2] * vx * vy + gl[1][3] * vx * vz + gl[2][3] * vy * vz);
      v.max_t = t;
      v.x = x;
      v.y = y;
      v.z = z;
      v.rho = rho;
      v.b2 = b2;
      v.lor = sqrt(1 + q);
      v.gid = gid0 + m;
      v.ii = i - is;
      v.jj = j - js;
      v.kk = k - ks;
      v.edge_cells =
          fmin(fmin(i - is, is + nx - 1 - i),
               fmin(fmin(j - js, js + ny - 1 - j), fmin(k - ks, ks + nz - 1 - k)));
    }
  }
};
}  // namespace

bool CyclicZoom::InterfaceDiagnostics(int stage) {
  if (debug_interval <= 0 || zstate.zone < 4 || (pmesh->ncycle % debug_interval != 0))
    return false;
  auto pack = pmesh->pmb_pack;
  auto mhd = pack->pmhd;
  if (mhd == nullptr || !pack->pcoord->is_general_relativistic ||
      !mhd->peos->eos_data.is_ideal)
    return false;
  auto ind = pmesh->mb_indcs;
  ShellFunctor f;
  f.w = mhd->w0;
  f.u = mhd->u0;
  f.b1 = mhd->b0.x1f;
  f.b2f = mhd->b0.x2f;
  f.b3 = mhd->b0.x3f;
  f.size = pack->pmb->mb_size;
  f.is = ind.is;
  f.js = ind.js;
  f.ks = ind.ks;
  f.nx = ind.nx1;
  f.ny = ind.nx2;
  f.nz = ind.nx3;
  f.gid0 = pmesh->gids_eachrank[global_variable::my_rank];
  f.cx = zregion.x1c;
  f.cy = zregion.x2c;
  f.cz = zregion.x3c;
  f.radius = zregion.radius;
  f.gm1 = mhd->peos->eos_data.gamma - 1;
  f.spin = pack->pcoord->coord_data.bh_spin;
  f.flat = pack->pcoord->coord_data.is_minkowski;
  ShellResult v;
  Kokkos::parallel_reduce(
      "debug_shell",
      Kokkos::RangePolicy<>(DevExeSpace(), 0,
                            pack->nmb_thispack * ind.nx1 * ind.nx2 * ind.nx3),
      f, v);
  Real sums[4] = {v.mass, v.energy, v.magnetic, v.thermal};
  Real maxima[2] = {v.max_b2rho, v.max_divb};
  int counts[2] = {v.bad, v.cells};
#if MPI_PARALLEL_ENABLED
  struct {
    double value;
    int rank;
  } local = {v.max_t, global_variable::my_rank}, winner;
  MPI_Allreduce(&local, &winner, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Bcast(&v, sizeof(v), MPI_BYTE, winner.rank, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "DBG_SHELL cycle=" << pmesh->ncycle << " stage=" << stage
              << " zone=" << zstate.zone << " dt=" << pmesh->dt
              << " next_time=" << zstate.next_time
              << " Tmax=" << v.max_t << " xyz=" << v.x << "," << v.y << "," << v.z
              << " rho=" << v.rho << " lor=" << v.lor << " B2=" << v.b2
              << " gid=" << v.gid << " ijk=" << v.ii << "," << v.jj << "," << v.kk
              << " edge_cells=" << v.edge_cells << " max_B2rho=" << maxima[0]
              << " divBnorm=" << maxima[1] << " sumD=" << sums[0] << " sumE=" << sums[1]
              << " sumB2half=" << sums[2] << " sumEint=" << sums[3]
              << " bad=" << counts[0] << " cells=" << counts[1] << std::endl;
    if (counts[0] > 0 || (debug_stop_temp > 0.0 && v.max_t > debug_stop_temp))
      std::cout << "DBG_STOP interface threshold reached" << std::endl;
  }
  return counts[0] > 0 || (debug_stop_temp > 0.0 && v.max_t > debug_stop_temp);
}
