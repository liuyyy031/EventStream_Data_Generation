"""Executable domains live here; the shared generation core imports no domains."""
from .healthcare import HealthcarePackage, build_healthcare_temporal_models
from .transportation import TransportationPackage, build_transportation_temporal_models
from .distributed_systems import DistributedSystemsPackage, build_distributed_systems_temporal_models
from .energy import EnergyPackage, build_energy_temporal_models
from .environment_pollution import EnvironmentPollutionPackage, build_environment_pollution_temporal_models
from .ecology import EcologyPackage, build_ecology_temporal_models
from .public_health import PublicHealthPackage, build_public_health_temporal_models
from .hydrology import HydrologyPackage, build_hydrology_temporal_models
from .agriculture import AgriculturePackage, build_agriculture_temporal_models
from .climate import ClimatePackage, build_climate_temporal_models

DOMAIN_BUILDERS = {
    "transportation": (TransportationPackage, build_transportation_temporal_models),
    "healthcare": (HealthcarePackage, build_healthcare_temporal_models),
    "distributed_systems": (DistributedSystemsPackage, build_distributed_systems_temporal_models),
    "energy": (EnergyPackage, build_energy_temporal_models),
    "environment_pollution": (EnvironmentPollutionPackage, build_environment_pollution_temporal_models),
    "ecology": (EcologyPackage, build_ecology_temporal_models),
    "public_health": (PublicHealthPackage, build_public_health_temporal_models),
    "hydrology": (HydrologyPackage, build_hydrology_temporal_models),
    "agriculture": (AgriculturePackage, build_agriculture_temporal_models),
    "climate": (ClimatePackage, build_climate_temporal_models),
}
