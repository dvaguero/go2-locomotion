"""iCEM (Improved Cross-Entropy Method, Pinneri et al. 2020 --
"Sample-Efficient Cross-Entropy Method for Real-Time Planning") para
ajustar los 6 parámetros del CPG (cpg/oscillator.py + cpg/foot_mapping.py):
l_step, l_clrnc, l_pntr, omega, mu, alpha.

Optimiza CONTRA cpg/cost_function.py, que evalúa cada candidato
corriendo un rollout headless (cpg/headless_sim.py) y premia el avance
en x del CoM del robot (costo = -delta_x_gait). Ejecutar con:
    python cpg/icem.py

Por qué L_STEP tiene rango (0.0, 0.15) y NO signo libre (2026-07-22,
revisado)
-------------------------------------------------------------------
Ver el docstring de cpg/cost_function.py para el detalle completo y la
historia completa (incluye varios intentos fallidos: análisis
simbólico, medición con headless_sim.py, y dejar que iCEM eligiera el
signo con rango simétrico [-0.15, 0.15] -- convergió a l_step=-0.15,
pero eso también caminaba hacia atrás en el simulador real).

En resumen, la razón de fondo de por qué NINGÚN signo de L_STEP
"funcionaba" no tenía que ver con L_STEP: con y=0 fijo en leg_ik, el
robot se volcaba de costado (roll ~73°) en vez de caminar, y esa caída
lateral confundía cualquier medición de avance en x, en headless_sim.py
y en el simulador real por igual. Corregido eso (offset lateral
y=±HIP_LENGTH en cpg/run_cpg.py/headless_sim.py, roll estable en ±3°),
se comparó el signo de nuevo y "-" (el original) resultó casi neutral
en régimen estable, mientras que "+" retrocedía activamente -- por eso
el signo quedó fijo en "-" y el rango acá es una magnitud no-negativa
(0.0, 0.15) que compone con ese signo. iCEM optimiza la magnitud de
L_STEP y los otros 5 parámetros, no el signo -- esa incertidumbre
específica ya no está abierta.

Qué partes de iCEM (paper original) se implementan y cuáles no
-------------------------------------------------------------------
El iCEM del paper está pensado para MPC: optimiza SECUENCIAS de
acciones a lo largo de un horizonte temporal, y su aporte principal
sobre CEM clásico es muestrear "ruido de color" (correlacionado en el
tiempo) en vez de ruido blanco, porque secuencias de acción suaves son
mejores puntos de partida que ruido i.i.d. para ese problema.

Acá NO hay una secuencia temporal que optimizar: los 6 parámetros del
CPG son constantes durante todo el rollout (no varían con t), así que
el ruido de color del paper no aplica -- se muestrea ruido blanco
Gaussiano normal, como en CEM clásico. Sí se toman las otras dos
mejoras de iCEM sobre CEM, que no dependen de que haya una secuencia
temporal:
  1) Tamaño de población decreciente por iteración (menos evaluaciones
     caras a medida que la distribución converge).
  2) Persistencia de elites: una fracción de los mejores candidatos de
     la iteración anterior se reinyecta en la población actual (y no
     se re-evalúa -- su costo ya se conoce, el rollout es
     determinístico) en vez de descartarlos, mejorando la eficiencia
     muestral.
Además se trackea el mejor candidato visto en CUALQUIER iteración, no
solo el de la última, porque con poblaciones chicas al final el óptimo
de una iteración intermedia puede ser mejor que el de la última.
"""

import numpy as np

import cost_function as cf

PARAM_NAMES = cf.PARAM_NAMES
LOWER_BOUNDS = cf.LOWER_BOUNDS
UPPER_BOUNDS = cf.UPPER_BOUNDS

# --- Hiperparámetros de iCEM ------------------------------------------------
N_ITERATIONS = 15
POP_SIZE_INITIAL = 40
POP_SIZE_MIN = 10
POP_DECAY = 0.85  # tamaño de población decae geométricamente por iteración
ELITE_FRACTION = 0.25
ELITE_KEEP_FRACTION = 0.3  # fracción de elites de la iteración anterior
# que se reinyecta (sin re-evaluar) en la población de la iteración actual
INIT_STD_FRACTION = 0.5  # std inicial, como fracción del rango de cada parámetro
MIN_STD_FRACTION = 0.02  # piso de std relativo (evita colapso prematuro)
SIM_DURATION = 4.0  # s por rollout evaluado (ver cost_function.evaluate)


def _population_size(iteration: int) -> int:
    size = int(POP_SIZE_INITIAL * (POP_DECAY**iteration))
    return max(POP_SIZE_MIN, size)


def _sample(mean: np.ndarray, std: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    samples = rng.normal(loc=mean, scale=std, size=(n, len(mean)))
    return np.clip(samples, LOWER_BOUNDS, UPPER_BOUNDS)


def run_icem(seed: int = 0, verbose: bool = True) -> tuple[np.ndarray, float]:
    """Corre iCEM y devuelve (mejor_vector, mejor_costo)."""
    rng = np.random.default_rng(seed)

    mean = (LOWER_BOUNDS + UPPER_BOUNDS) / 2.0
    std = INIT_STD_FRACTION * (UPPER_BOUNDS - LOWER_BOUNDS)
    min_std = MIN_STD_FRACTION * (UPPER_BOUNDS - LOWER_BOUNDS)

    best_vector = mean.copy()
    best_cost = np.inf

    # Elites persistidos de la iteración anterior, con su costo ya
    # calculado (no se re-evalúan: el rollout es determinístico dado
    # el mismo vector de parámetros).
    prev_elites = np.empty((0, len(PARAM_NAMES)))
    prev_elite_costs = np.empty((0,))

    for it in range(N_ITERATIONS):
        pop_size = _population_size(it)
        n_new = max(pop_size - len(prev_elites), 1)

        new_samples = _sample(mean, std, n_new, rng)
        new_costs = cf.evaluate_batch(new_samples, sim_duration=SIM_DURATION)

        if len(prev_elites) > 0:
            population = np.vstack([prev_elites, new_samples])
            costs = np.concatenate([prev_elite_costs, new_costs])
        else:
            population, costs = new_samples, new_costs

        order = np.argsort(costs)  # costo ascendente: mejor primero
        n_elite = max(1, int(ELITE_FRACTION * len(population)))
        elite_idx = order[:n_elite]
        elites = population[elite_idx]
        elite_costs = costs[elite_idx]

        if elite_costs[0] < best_cost:
            best_cost = float(elite_costs[0])
            best_vector = elites[0].copy()

        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), min_std)

        n_keep = max(1, int(ELITE_KEEP_FRACTION * n_elite))
        prev_elites = elites[:n_keep]
        prev_elite_costs = elite_costs[:n_keep]

        if verbose:
            print(
                f"iter {it:2d} | pop={len(population):3d} | "
                f"mejor costo iter={elite_costs[0]:+.5f} | "
                f"mejor costo global={best_cost:+.5f} "
                f"(avance ~{-best_cost * 1000:+.2f} mm)"
            )

    return best_vector, best_cost


if __name__ == "__main__":
    best_vector, best_cost = run_icem()
    best_params = cf.vector_to_params(best_vector)

    print("\nMejores parámetros encontrados:")
    for name, value in best_params.items():
        print(f"  {name} = {value:.5f}")
    print(f"delta_x_gait estimado: {-best_cost:+.4f} m (sim_duration={SIM_DURATION}s)")
    print(
        "\nADVERTENCIA: validar estos parámetros corriendo cpg/run_cpg.py "
        "contra el simulador real (unitree_mujoco.py) antes de confiar en "
        "ellos -- headless_sim.py no reprodujo con fidelidad total el "
        "comportamiento observado en el simulador real con GUI en pruebas "
        "anteriores (ver docstring de cpg/cost_function.py)."
    )
