"""Executable domains live here; the shared generation core imports no domains."""
from .healthcare import HealthcarePackage, build_healthcare_temporal_models
from .transportation import TransportationPackage, build_transportation_temporal_models
from .distributed_systems import DistributedSystemsPackage, build_distributed_systems_temporal_models

DOMAIN_BUILDERS = {
    "transportation": (TransportationPackage, build_transportation_temporal_models),
    "healthcare": (HealthcarePackage, build_healthcare_temporal_models),
    "distributed_systems": (DistributedSystemsPackage, build_distributed_systems_temporal_models),
}
