"""Wall-clock governor for the submission run.

Evaluation is capped at 30 minutes for the public and private splits together.
Measured here: the harness itself costs about 68 seconds per split before our
code runs at all, and the previous solution's profile projected to 25.8-27.6
minutes for 96 scenarios. That is not enough headroom -- if either split holds
more devices than train's 461, the run overruns and scores nothing.

The deadlines below are re-derived whenever the planner configuration changes,
because a governor tuned for a 15-minute run will shred a 22-minute one.

So the planner degrades on a clock rather than betting that it will fit:
past the soft deadline it drops to a cheap search, and past the hard deadline it
returns the all-defer plan, which is always valid and always cheap to produce.
A late scenario planned badly costs a few hundred; a run that does not finish
costs everything.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from batteryswap_public.interfaces import Planner

LOGGER = logging.getLogger(__name__)

# Both derived from a measurement of the actual entry point, not guessed. On the
# shipped configuration `BATTERYSWAP_SPLITS=train python script.py` plans all 48
# train scenarios in **673 s** with nothing degraded or deferred, so the official
# 96 project to about **22.5 minutes** against the 30-minute cap.
#
# The soft deadline is therefore set above the expected total rather than below
# it: at 17 minutes it would have fired around scenario 72 of 96 on a *healthy*
# run and degraded a quarter of the submission for nothing. 25 minutes fires
# only if the run is more than 11 % over expectation -- which is the case it
# exists for, an evaluation machine slower than this one or a split with more
# than train's 461 devices. On a machine 30 % slower the governor takes over at
# roughly 82 % complete and the remainder finishes at the cheap search in a
# minute or two.
#
# The hard deadline leaves 2.5 minutes for the all-defer tail, which needs no
# planning at all, so it cannot itself cause an overrun.
SOFT_DEADLINE_SECONDS = 25 * 60
HARD_DEADLINE_SECONDS = 27 * 60 + 30


class BudgetedPlanner(Planner):
    """Wraps a planner and trades plan quality for finishing on time."""

    def __init__(
        self,
        inner,
        fast_config=None,
        *,
        soft_deadline: float = SOFT_DEADLINE_SECONDS,
        hard_deadline: float = HARD_DEADLINE_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.inner = inner
        self.fast_config = fast_config
        self.soft_deadline = float(soft_deadline)
        self.hard_deadline = float(hard_deadline)
        self._clock = clock
        self._started = clock()
        self._degraded = False
        self.scenarios_planned = 0
        self.scenarios_degraded = 0
        self.scenarios_deferred = 0

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    def _all_defer(self, locations: pd.DataFrame, settings) -> pd.DataFrame:
        # Mirrors CompetitionPlanner._all_defer without needing the scenario
        # start, which we may not have reached far enough to infer safely.
        id_column = "battery_id" if "battery_id" in locations else "battery"
        day = pd.Timestamp.now().normalize() + pd.Timedelta(days=365 * 50)
        return pd.DataFrame(
            {
                "day": pd.DatetimeIndex([day] * len(locations)),
                "battery": sorted(locations[id_column].astype(str)),
            }
        ).reset_index(drop=True)

    def plan(
        self,
        battery_data: pd.DataFrame,
        locations: pd.DataFrame,
        travel_costs: pd.DataFrame,
        settings,
    ) -> pd.DataFrame:
        elapsed = self.elapsed
        if elapsed > self.hard_deadline:
            self.scenarios_deferred += 1
            LOGGER.warning(
                "hard deadline passed at %.0fs; deferring scenario wholesale", elapsed
            )
            return self._all_defer(locations, settings)

        if elapsed > self.soft_deadline and not self._degraded and self.fast_config:
            LOGGER.warning(
                "soft deadline passed at %.0fs; switching to the cheap search", elapsed
            )
            self.inner.config = self.fast_config
            self._degraded = True

        plan = self.inner.plan(battery_data, locations, travel_costs, settings)
        self.scenarios_planned += 1
        if self._degraded:
            self.scenarios_degraded += 1
        return plan
