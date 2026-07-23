"""Función de costo para iCEM (cpg/icem.py): evalúa un vector de
parámetros del CPG corriendo un rollout headless (cpg/headless_sim.py)
y devuelve un costo escalar a MINIMIZAR.

Premia el avance en x (adelante, convención de inverse_kinematics.py)
durante la fase de gait del rollout: costo = -delta_x_gait (ver
headless_sim.run_rollout, que ya separa ese delta del transitorio de
pararse desde la pose home). Minimizar costo == maximizar avance.

Decisión de diseño -- por qué L_STEP NO tiene signo libre (2026-07-22,
revisado)
-------------------------------------------------------------------
Se intentó, en este orden, delegar la decisión del signo de L_STEP
(cpg/foot_mapping.py) en vez de fijarlo a mano: análisis simbólico,
medición con headless_sim.py, y dejar que iCEM lo eligiera con rango
simétrico [-0.15, 0.15]. Las tres vías, y varias pruebas en el
simulador real con GUI, daban sistemáticamente "camina hacia atrás"
sin importar el signo probado ("-", "-0.15", "+").

La razón de fondo NO era el signo de L_STEP: con y=0 fijo al llamar a
leg_ik (ver cpg/run_cpg.py), el robot tiene un polígono de apoyo
lateral casi nulo y se VUELCA DE COSTADO (roll de 0° a ~73° en 2 s) en
vez de caminar -- toda medición de "avance en x" hecha hasta entonces,
en headless_sim.py o en el simulador real, en realidad medía un robot
cayéndose de lado. Corregido eso (offset lateral y=±HIP_LENGTH, roll
estable en ±3°), se volvió a comparar el signo con headless_sim.py
-- ahora sin ese confundidor de por medio -- y el resultado fue:
"+"  retrocede activamente incluso en régimen estable (~-55 mm/s).
"-"  casi neutral en régimen estable (~-5 mm/s, ~10x menor), aunque
     tampoco avanza.
Por eso el signo quedó fijo en "-" (el original de la literatura, ver
cpg/foot_mapping.py) y PARAM_BOUNDS["l_step"] es (0.0, 0.15) -- una
magnitud no-negativa que compone con ese signo ya fijo en la fórmula.
Que ningún signo lograra avance neto (solo "menos malo" vs "peor") es
un problema aparte, probablemente de magnitud/combinación de
parámetros más que de signo -- tarea razonable para que iCEM explore,
una vez confirmado en el simulador real que "-" con el offset lateral
no es contraproducente.

iCEM optimiza la MAGNITUD de L_STEP (y los otros 5 parámetros), no su
signo -- esa incertidumbre específica (signo) ya no está abierta.
"""

import numpy as np

from headless_sim import load_model, run_rollout

PARAM_NAMES = ("l_step", "l_clrnc", "l_pntr", "omega", "mu", "alpha")

# (low, high) por parámetro. l_step ya no tiene rango simétrico -- su
# signo está fijado a mano en cpg/foot_mapping.py, ver nota de diseño
# arriba.
PARAM_BOUNDS = {
    "l_step": (0.0, 0.15),
    "l_clrnc": (0.0, 0.08),
    "l_pntr": (0.0, 0.02),
    "omega": (2.0 * np.pi * 0.5, 2.0 * np.pi * 3.0),
    "mu": (0.2, 1.5),
    "alpha": (10.0, 100.0),
}

LOWER_BOUNDS = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES])
UPPER_BOUNDS = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES])

_model = None  # cacheado: el XML se parsea una sola vez (ver load_model)


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def vector_to_params(vector: np.ndarray) -> dict:
    """Vector (orden PARAM_NAMES) -> dict de kwargs para run_rollout."""
    return dict(zip(PARAM_NAMES, vector))


def evaluate(vector: np.ndarray, sim_duration: float = 4.0) -> float:
    """Costo de un candidato. sim_duration es más corto que el default
    de headless_sim.py (6.0 s) para que evaluar cientos de candidatos
    en iCEM sea rápido -- la prueba de validación (L_STEP=0.15, 6s)
    mostró que la tendencia de avance/retroceso ya es clara mucho
    antes de los 6s.
    """
    model = _get_model()
    params = vector_to_params(vector)
    result = run_rollout(model, params, sim_duration=sim_duration)
    return -result["delta_x_gait"]


def evaluate_batch(vectors: np.ndarray, sim_duration: float = 4.0) -> np.ndarray:
    """Evalúa varios candidatos (uno por fila). Secuencial, no
    paralelo: un mismo mujoco.MjModel no es seguro de compartir entre
    hilos sin darle un MjData (y potencialmente un model) por worker;
    paralelizar de verdad requeriría un pool de modelos, no se hizo
    acá por simplicidad.
    """
    return np.array([evaluate(v, sim_duration=sim_duration) for v in vectors])
