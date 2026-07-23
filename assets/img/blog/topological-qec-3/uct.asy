import graph3;
import three;

from asyfuncs access drawSphere, drawCylinder;

size(607,300);

currentprojection = orthographic(
  camera=-(-0.249078, 0.949165, -0.192474),
  up=(-0.028702, 0.191416, 0.981089),
  zoom=0.746215
);

int latticeCells = 3;
triple latticeCenter = 0.5*latticeCells*(X + Y + Z);

real latticeEdgeRadius = 0.015;
real stringEdgeRadius = 0.05;
real stringVertexRadius = 0.055;

pen latticePen = opacity(0.72) + rgb(0.62, 0.62, 0.62);
pen stringPen = rgb(0.86, 0.08, 0.08);
pen topLayerPen = rgb(0.28, 0.58, 1.0);
pen intersectionPen = rgb(0.55, 0.10, 0.78);

triple latticePoint(real x, real y, real z) {
  return (x, y, z) - latticeCenter;
}

void drawLatticeEdge(real x0, real y0, real z0, real x1, real y1, real z1) {
  drawCylinder(
    latticePoint(x0, y0, z0),
    latticePoint(x1, y1, z1),
    latticeEdgeRadius,
    latticePen
  );
}

void drawHighlightedVerticalEdge(int x, int y, int z, pen p) {
  drawCylinder(
    latticePoint(x, y, z),
    latticePoint(x, y, z + 1),
    stringEdgeRadius,
    p
  );
}

// Unique lattice edges for the cubic cell complex.
for (int x = 0; x < latticeCells; ++x) {
  for (int y = 0; y <= latticeCells; ++y) {
    for (int z = 0; z <= latticeCells; ++z) {
      drawLatticeEdge(x, y, z, x + 1, y, z);
    }
  }
}

for (int x = 0; x <= latticeCells; ++x) {
  for (int y = 0; y < latticeCells; ++y) {
    for (int z = 0; z <= latticeCells; ++z) {
      drawLatticeEdge(x, y, z, x, y + 1, z);
    }
  }
}

for (int x = 0; x <= latticeCells; ++x) {
  for (int y = 0; y <= latticeCells; ++y) {
    for (int z = 0; z < latticeCells; ++z) {
      drawLatticeEdge(x, y, z, x, y, z + 1);
    }
  }
}

// Highlight all vertical edges in the top layer, except the string overlap.
for (int x = 0; x <= latticeCells; ++x) {
  for (int y = 0; y <= latticeCells; ++y) {
    if (!(x == 1 && y == 1)) {
      drawHighlightedVerticalEdge(x, y, latticeCells - 1, topLayerPen);
    }
  }
}

// The vertical string of primal z-edges at lattice coordinate (1,1,z).
for (int z = 0; z < latticeCells; ++z) {
  pen edgePen = z == latticeCells - 1 ? intersectionPen : stringPen;
  drawHighlightedVerticalEdge(1, 1, z, edgePen);
}
