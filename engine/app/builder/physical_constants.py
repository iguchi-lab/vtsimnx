"""Standard dry-air properties shared by builder assembly and validation."""
DENSITY_DRY_AIR = 1.2  # kg/m3, reference volumetric-flow convention
SPECIFIC_HEAT_AIR = 1006.0  # J/(kg K)
DRY_AIR_RHO_CP = DENSITY_DRY_AIR * SPECIFIC_HEAT_AIR
