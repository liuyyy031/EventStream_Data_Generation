#!/usr/bin/env python3
"""v6 entry point with safe Judge handling and list-valued peak support."""

from __future__ import annotations

import demo_sts_stpp_v6 as _v6
from safe_judge_generator import SafeJudgeNetworkSDEGenerator
from stpp_adapter_v6_listfix import simulate_stpp_for_streasoner_v6_listfix


# Inject both compatibility layers into the unchanged v6 entry point.
_v6.NetworkSDEGenerator = SafeJudgeNetworkSDEGenerator
_v6.simulate_stpp_for_streasoner_v6 = simulate_stpp_for_streasoner_v6_listfix

demo_stpp_generation_v6 = _v6.demo_stpp_generation_v6


def main() -> int:
    return _v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
