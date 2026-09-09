"""Stable transportation event-type identifiers for EventFlow v1.4."""

from pathlib import Path

from ...core.type_registry import EventTypeRegistry


REGISTRY_PATH = Path(__file__).with_name("event_type_registry_v2.json")

HEAVY_RAIN_ONSET = "transportation.road.environment.heavy_rain_onset"
FLOODING_ONSET = "transportation.road.obstruction.flooding_onset"
VEHICLE_COLLISION = "transportation.road.incident.vehicle_collision"
VEHICLE_BREAKDOWN = "transportation.road.incident.vehicle_breakdown"
ROAD_DEBRIS = "transportation.road.obstruction.road_debris"
TRAFFIC_SIGNAL_FAILURE = "transportation.road.fault.traffic_signal_failure"
ROAD_CLOSURE_STARTED = "transportation.road.management.road_closure_started"
ROADWORKS_STARTED = "transportation.road.management.roadworks_started"
CONGESTION_ONSET = "transportation.road.traffic_state.congestion_onset"
CONGESTION_EASING = "transportation.road.traffic_state.congestion_easing"
NORMAL_FLOW_RESTORED = "transportation.road.traffic_state.normal_flow_restored"

CONGESTION_TYPE_IDS = {CONGESTION_ONSET}
RECOVERY_TYPE_IDS = {CONGESTION_EASING, NORMAL_FLOW_RESTORED}


def load_event_type_registry() -> EventTypeRegistry:
    return EventTypeRegistry.load(REGISTRY_PATH)
