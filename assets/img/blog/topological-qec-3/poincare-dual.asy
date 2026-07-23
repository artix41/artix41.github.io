import graph3;
import three;

from asyfuncs access drawSphere, drawCylinder;

size(607,300);

currentprojection = orthographic(
  camera=-(-0.243115, 0.941709, -0.232550),
  up=(-0.028338, 0.232745, 0.972125),
  zoom=1
);

int latticeCells = 2;
triple latticeCenter = 0.5*latticeCells*(X + Y + Z);

real latticeEdgeRadius = 0.015;
real stringEdgeRadius = 0.05;
real stringVertexRadius = 0.055;

pen latticePen = opacity(0.72) + rgb(0.62, 0.62, 0.62);
pen stringPen = rgb(0.86, 0.08, 0.08);
pen dualFacePen = opacity(0.32) + rgb(0.10, 0.38, 0.95);
pen dualFaceBoundaryPen = opacity(0.68) + rgb(0.10, 0.38, 0.95);

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

void drawDualFace(int z) {
  real x = 0.5;
  real y = 0.5;
  real zFace = z;
  real halfWidth = 0.5;

  path3 face = (
    latticePoint(x - halfWidth, y - halfWidth, zFace)
    -- latticePoint(x + halfWidth, y - halfWidth, zFace)
    -- latticePoint(x + halfWidth, y + halfWidth, zFace)
    -- latticePoint(x - halfWidth, y + halfWidth, zFace)
    -- cycle
  );

  draw(surface(face), surfacepen=dualFacePen, light=nolight);
  draw(face, dualFaceBoundaryPen + linewidth(0.8pt));
}

void drawStringEdge(int z) {
  drawCylinder(
    latticePoint(1, 1, z),
    latticePoint(1, 1, z + 1),
    stringEdgeRadius,
    stringPen
  );
}

// Poincare-dual plaquettes first, drawn as faces of the current lattice.
for (int z = 0; z < latticeCells+1; ++z) {
  drawDualFace(z);
}

// Unique lattice edges for a 4x4x4 cell complex.
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

// The vertical string of four primal z-edges at lattice coordinate (1,1,z).
for (int z = 0; z < latticeCells; ++z) {
  drawStringEdge(z);
}

for (int z = 0; z <= latticeCells; ++z) {
  drawSphere(latticePoint(1, 1, z), stringVertexRadius, stringPen);
}
