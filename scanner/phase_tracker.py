import numpy as np
from dataclasses import dataclass, field

@dataclass
class PhaseState:
    """
    Holds phase state information.
    """
    active: bool = False
    start_time: float | None = None
    start_health: float | None = None
    last_damage_time: float | None = None
    time_history: list[float] = field(default_factory=list)
    dps_history: list[float] = field(default_factory=list)


@dataclass
class VisibilityState:
    """
    Holds information regarding whether the boss bar is visible or not.
    """
    visible: bool = False
    stabilizing_until: float = 0.0
    threshold: int = 50
    stabilization_time: float = 0.20


class PhaseTracker:
    """
    Handles phase detection, visibility, and DPS tracking.
    Mirrors the original _update_phase_state + _is_health_bar_visible behavior.
    """

    def __init__(self, min_damage_fraction: float, idle_timeout: float):
        self.state = PhaseState()
        self.visibility = VisibilityState()
        self.min_damage_fraction = min_damage_fraction
        self.idle_timeout = idle_timeout
        self._last_health_fraction: float | None = None

    @staticmethod
    def _is_health_bar_visible(cropped_img, neg_mask, threshold: int) -> bool:
        """
        Determines whether the health bar is visible based on mask pixel count
        and basic sanity checks.
        """
        # Count mask pixels
        mask_pixels = np.count_nonzero(neg_mask)

        # If too few pixels, bar is not visible
        if mask_pixels < threshold:
            return False

        # If the cropped region is extremely dark or bright, it's probably fading
        mean_val = cropped_img.mean()
        if mean_val < 5 or mean_val > 250:
            return False

        return True

    def update(self, now: float, current_health: float, cropped_img, neg_mask) -> None:
        """
        Robust phase detection with:
        - health bar visibility detection
        - stabilization window after reappearing
        - no false phase starts from bar disappearing/reappearing
        - no DPS spikes from 0→N or N→0 transitions
        """

        prev_health = self._last_health_fraction or current_health

        # 1. Determine if the health bar is visible
        visible = self._is_health_bar_visible(
            cropped_img=cropped_img,
            neg_mask=neg_mask,
            threshold=self.visibility.threshold
        )

        # Bar disappeared → end phase immediately
        if not visible:
            # Bar disappeared
            self.visibility.visible = False

            # If a phase is active, KEEP IT ACTIVE
            if self.state.active:
                # Do not update damage or end the phase
                self._last_health_fraction = current_health
                return

            # If no phase is active, just reset tracking
            self._last_health_fraction = None
            return

        # Bar just reappeared → start stabilization window
        if visible and not self.visibility.visible:
            self.visibility.visible = True
            self.visibility.stabilizing_until = now + self.visibility.stabilization_time
            self._last_health_fraction = current_health
            return

        # Still stabilizing → ignore changes
        if now < self.visibility.stabilizing_until:
            self._last_health_fraction = current_health
            return

        # 2. Compute real damage
        damage = prev_health - current_health
        if abs(damage) < 1e-6:
            damage = 0.0

        # 3. Phase start
        if not self.state.active and damage > self.min_damage_fraction:
            self.state.active = True
            self.state.start_time = now
            self.state.start_health = prev_health
            self.state.last_damage_time = now
            self.state.time_history.clear()
            self.state.dps_history.clear()
            self._last_health_fraction = current_health
            return

        # 4. Phase continues
        if self.state.active:

            # Update last damage time
            if damage > self.min_damage_fraction:
                self.state.last_damage_time = now

            # Phase ends after idle timeout
            if now - self.state.last_damage_time > self.idle_timeout:
                self.state.active = False
                self._last_health_fraction = current_health
                return

            # Update DPS curve only if bar is visible
            if visible:
                elapsed = now - self.state.start_time
                if elapsed > 0:
                    total_damage = max(0.0, self.state.start_health - current_health)
                    dps = total_damage / elapsed
                    self.state.time_history.append(elapsed)
                    self.state.dps_history.append(dps)

        self._last_health_fraction = current_health
