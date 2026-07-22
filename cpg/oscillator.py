"""Oscilador de Hopf en task space para generar el ritmo de cada pata.

Formulación basada en Shafiee, Bellegarda & Ijspeert (2023),
"ManyQuadrupeds: Learning a Single Locomotion Policy for Diverse
Quadruped Robots" — referencia [6] del proyecto.

Ecuaciones (una instancia por pata, subíndice i):
    r̈ᵢ = α · (α/4 · (μᵢ - rᵢ) - ṙᵢ)
    θ̇ᵢ = ωᵢ

- rᵢ es la amplitud del oscilador (radio en task space, ver
  foot_mapping.py). La ecuación de r̈ᵢ es un sistema masa-resorte-
  amortiguador de segundo orden (masa=1, rigidez k=α²/4,
  amortiguamiento c=α) escrito como r̈ = -k(r-μ) - c·ṙ. Con
  c = 2·sqrt(k·m) = 2·sqrt(α²/4) = α, el sistema queda exactamente
  *críticamente amortiguado*: rᵢ converge a μᵢ tan rápido como es
  posible sin overshoot ni oscilación, evitando transitorios
  indeseados en la amplitud del paso cuando μᵢ cambia (p. ej. al
  arrancar o frenar la caminata).
- θᵢ es la fase del oscilador y avanza a velocidad angular constante
  ωᵢ (frecuencia de paso, rad/s). El desfase entre patas para formar
  un gait (ver QuadrupedCPG más abajo) se logra con offsets
  *iniciales* de θᵢ, no con distintos ωᵢ: todas las patas comparten
  la misma frecuencia angular para que el desfase relativo se
  mantenga constante en el tiempo (un gait periódico y sincronizado).
"""

import math
from dataclasses import dataclass

LEG_IDS = ("FL", "FR", "RL", "RR")

# Offsets de fase inicial (rad) para un gait de trote (trot gait).
# En un trote, las patas diagonalmente opuestas pisan en fase (mismo
# θ en todo instante) y cada diagonal está desfasada media vuelta
# (π rad) respecto a la otra diagonal:
#   diagonal FL-RR: offset 0
#   diagonal FR-RL: offset π (medio ciclo respecto a FL-RR)
# Como todas comparten el mismo ω (ver docstring del módulo), este
# desfase inicial se preserva durante toda la marcha.
TROT_PHASE_OFFSETS = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": math.pi,
    "RL": math.pi,
}


@dataclass
class HopfOscillator:
    """Oscilador de Hopf de una pata: estado (r, ṙ, θ)."""

    alpha: float  # ganancia de convergencia de r (rad/s), controla la
    # velocidad de convergencia crítica hacia mu (mayor alpha = más
    # rápido, pero valores muy altos exigen dt más chico para
    # integrar establemente).
    omega: float  # velocidad angular de la fase (rad/s); junto al
    # radio r fija la frecuencia de paso (cadencia).
    mu: float  # amplitud objetivo de r en steady-state (radio, ver
    # foot_mapping._step_length).
    r: float = 0.0  # amplitud actual
    r_dot: float = 0.0  # derivada de r
    theta: float = 0.0  # fase actual (rad); se usa para fijar el
    # offset inicial de cada pata (ver TROT_PHASE_OFFSETS).

    def _derivatives(self, r: float, r_dot: float) -> tuple[float, float, float]:
        """(ṙ, r̈, θ̇) evaluados en el estado (r, ṙ), según las
        ecuaciones del oscilador. θ̇ no depende de (r, ṙ) pero se
        retorna para reutilizar esta función en el integrador RK4.
        """
        r_ddot = self.alpha * (self.alpha / 4.0 * (self.mu - r) - r_dot)
        theta_dot = self.omega
        return r_dot, r_ddot, theta_dot

    def step(self, dt: float, method: str = "rk4") -> None:
        """Integra un paso de tiempo dt (segundos).

        method: "euler" (explícito, 1er orden) o "rk4" (Runge-Kutta 4,
        recomendado: r̈ depende de ṙ, así que un paso Euler grosero
        introduce error de amortiguamiento notorio si dt no es muy
        pequeño frente a 1/alpha).
        """
        if method == "euler":
            r_dot0, r_ddot0, theta_dot0 = self._derivatives(self.r, self.r_dot)
            self.r += dt * r_dot0
            self.r_dot += dt * r_ddot0
            self.theta += dt * theta_dot0
        elif method == "rk4":
            self._step_rk4(dt)
        else:
            raise ValueError(f"method inválido: {method!r} (usar 'euler' o 'rk4')")

        self.theta %= 2.0 * math.pi

    def _step_rk4(self, dt: float) -> None:
        # theta_dot = omega es constante (no depende de r, r_dot), así
        # que theta(t+dt) = theta(t) + omega*dt es exacto sin importar
        # el método de integración usado para (r, r_dot); solo (r,
        # r_dot) necesitan RK4 real.
        r0, rd0 = self.r, self.r_dot

        k1_r, k1_rd, _ = self._derivatives(r0, rd0)
        k2_r, k2_rd, _ = self._derivatives(r0 + 0.5 * dt * k1_r, rd0 + 0.5 * dt * k1_rd)
        k3_r, k3_rd, _ = self._derivatives(r0 + 0.5 * dt * k2_r, rd0 + 0.5 * dt * k2_rd)
        k4_r, k4_rd, _ = self._derivatives(r0 + dt * k3_r, rd0 + dt * k3_rd)

        self.r = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        self.r_dot = rd0 + (dt / 6.0) * (k1_rd + 2.0 * k2_rd + 2.0 * k3_rd + k4_rd)
        self.theta = self.theta + self.omega * dt


class QuadrupedCPG:
    """Conjunto de 4 osciladores de Hopf (uno por pata), con el
    desfase de fase inicial que define el gait (por default, trote).
    """

    def __init__(
        self,
        alpha: float,
        omega: float,
        mu: float,
        phase_offsets: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            alpha, omega, mu: parámetros del oscilador, compartidos
                por las 4 patas (mismo ritmo para todas).
            phase_offsets: offset inicial de θ por pata (rad); por
                default TROT_PHASE_OFFSETS (trote). Pasar otro dict
                (p. ej. todos 0.0 para "pronking", o offsets de 0,
                π/2, π, 3π/2 para "walk") para otro gait.
        """
        offsets = phase_offsets if phase_offsets is not None else TROT_PHASE_OFFSETS
        self.oscillators: dict[str, HopfOscillator] = {
            leg_id: HopfOscillator(alpha=alpha, omega=omega, mu=mu, theta=offsets[leg_id])
            for leg_id in LEG_IDS
        }

    def step(self, dt: float, method: str = "rk4") -> None:
        """Integra todas las patas un paso dt (mismo método para todas)."""
        for osc in self.oscillators.values():
            osc.step(dt, method=method)

    def state(self) -> dict[str, tuple[float, float]]:
        """Retorna {leg_id: (r, theta)} de las 4 patas."""
        return {leg_id: (osc.r, osc.theta) for leg_id, osc in self.oscillators.items()}
