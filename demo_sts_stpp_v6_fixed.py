#!/usr/bin/env python3
"""v6 smoke-test entry point with defensive Judge 1 response handling."""

from __future__ import annotations

import demo_sts_stpp_v6 as _v6
from safe_judge_generator import SafeJudgeNetworkSDEGenerator


# demo_sts_stpp_v6 resolves this module global when constructing its generator.
# Rebinding it here leaves the original file untouched.
_v6.NetworkSDEGenerator = SafeJudgeNetworkSDEGenerator

demo_stpp_generation_v6 = _v6.demo_stpp_generation_v6


def main() -> int:
    return _v6.main()


if __name__ == "__main__":
    raise SystemExit(main())
