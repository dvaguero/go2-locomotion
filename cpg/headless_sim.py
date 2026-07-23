"""Simulación MuJoCo headless para medir la dirección real de avance
del CPG, sin depender de unitree_sdk2py/DDS ni de un visor gráfico.

Carga external/unitree_mujoco/unitree_robots/go2/scene.xml (mujoco.MjModel/
MjData, sin pasar por unitree_mujoco.py ni el bridge) y corre el mismo
pipeline oscillator -> foot_mapping -> leg_ik que cpg/run_cpg.py, en un
loop de mj_step(). A diferencia de run_cpg.py, acá se puede leer
mj_data.qpos[0] (posición x del CoM en el mundo, componente x del
freejoint de la base) paso a paso.

BUG CRÍTICO CORREGIDO (2026-07-22): este script cargaba go2.xml
directamente en vez de scene.xml. go2.xml por sí solo NO tiene ningún
plano de suelo (ver <worldbody> ahí: solo geoms del robot) -- el suelo
(<geom name="floor" type="plane".../>) lo agrega scene.xml, que hace
<include file="go2.xml"/> y le suma el piso. Con go2.xml a secas el
robot simulaba en CAÍDA LIBRE (verificado: ncon=0 durante toda la
simulación, z cayendo de 0.27 a -122 m en 5 s sin tocar nada nunca).
Es decir: absolutamente TODO resultado de este script anterior a este
fix debe descartarse -- sin excepción, incluyendo (pero no limitado a):
  - las comparaciones de signo de L_STEP ("-", "+", con y sin timestep
    corregido a 0.005),
  - la corrida completa de iCEM (cpg/icem.py) contra este script,
  - la comparación OMEGA=2*pi*0.75/L_CLRNC=0.02 vs OMEGA=2*pi*1.5/
    L_CLRNC=0.04 hecha en el mismo turno en que se encontró este bug.
Cada una de esas mediciones midió el bamboleo del CoM por conservación
de momento de un robot cayendo por el vacío, no marcha real sobre el
suelo -- de ahí que sistemáticamente no coincidieran con el
comportamiento del simulador real. No era un problema de fidelidad
sutil (timestep, fricción, asincronía DDS); era la ausencia total de
física de contacto con el suelo.

Una vez corregido esto (cargando scene.xml), se encontró además que el
problema real no era de dirección de marcha (adelante/atrás) sino de
ESTABILIDAD LATERAL: con y=0 fijo en leg_ik, el robot vuelca de
costado (roll de 0° a ~73° en 2 s) en vez de caminar hacia atrás -- ver
la nota junto a LEG_Y_SIGN sobre la corrección aplicada.

También es la base de la simulación headless que necesita iCEM (Paso
6): load_model() y run_rollout() son funciones reusables, pensadas
para que cpg/cost_function.py las llame una vez por candidato
evaluado sin re-parsear el XML en cada llamada.

Requiere: pip install mujoco matplotlib (no requiere unitree_sdk2py).
Ejecutar con: python cpg/headless_sim.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from foot_mapping import foot_position
from inverse_kinematics import HIP_LENGTH, leg_ik
from oscillator import QuadrupedCPG

SCENE_XML = os.path.join(
    os.path.dirname(__file__),
    "..",
    "external",
    "unitree_mujoco",
    "unitree_robots",
    "go2",
    "scene.xml",
)
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "headless_sim_x.png")

# --- Parámetros (mismos valores que cpg/run_cpg.py, salvo aviso) -----------
# SIM_DURATION_S se alargó de 3.0 a 6.0 s (más ciclos de marcha = señal
# de desplazamiento más clara respecto al ruido numérico/transitorios,
# útil tanto para este diagnóstico como para evaluar candidatos en
# iCEM más adelante).
SIM_DURATION_S = 6.0
# DT = 0.005, no 0.002: el simulador real (unitree_mujoco.py línea 31)
# fuerza mj_model.opt.timestep = config.SIMULATE_DT = 0.005 s, y
# go2.xml no declara timestep propio (usaría el default de MuJoCo,
# 0.002 s). Con 0.002 este script integraba la física 2.5x más fino
# que el simulador real -- una diferencia de timestep que sí puede
# cambiar la dinámica de contacto lo suficiente como para invertir un
# efecto neto tan débil como el que se está midiendo acá. Se iguala a
# 0.005 para que este script sea un proxy fiel del simulador real (y
# de lo que verá iCEM más adelante).
DT = 0.005  # paso de control == paso de integración física

ALPHA = 50.0
OMEGA = 2.0 * np.pi * 1.5
MU = 1.0
L_STEP = 0.04  # valor de referencia (igual que run_cpg.py). El signo
# de L_STEP ya está fijado a mano en cpg/foot_mapping.py -- ver la nota
# ahí y en cpg/cost_function.py (rango (0.0, 0.15), no simétrico).
L_CLRNC = 0.04
L_PNTR = 0.005
H = 0.35
X_OFF = 0.0
Z_OFF = 0.0

# Y_OFFSET != 0 (2026-07-22): antes se llamaba a leg_ik con y=0.0 fijo.
# Ver la nota igual en cpg/run_cpg.py -- con y=0 las 4 patas quedan
# casi en línea, con un polígono de apoyo lateral casi nulo, y el
# robot se vuelca de costado (roll de 0° a ~73° en 2 s, medido con
# este mismo script una vez corregido el piso, ver más abajo). Separar
# los pies hacia afuera por HIP_LENGTH le da ancho de sustentación
# lateral real.
LEG_Y_SIGN = {"FL": 1.0, "RL": 1.0, "FR": -1.0, "RR": -1.0}

# kp=50.0/kd=3.5: igual a la postura final sostenida de
# external/unitree_mujoco/example/python/stand_go2.py (líneas 66-78),
# y a run_cpg.py (KP_STAND/KD_STAND) -- ver la nota ahí (2026-07-22)
# sobre por qué se corrigieron desde 60.0/5.0, valores sin referencia.
KP = 50.0
KD = 3.5
RAMP_TIME = 1.0  # ramp-in corto (pose home -> pie quieto) antes de oscilar

MOTOR_START_INDEX = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
LEG_ORDER = ("FR", "FL", "RR", "RL")


def _angles_to_motor_array(angles_by_leg: dict) -> np.ndarray:
    out = np.zeros(12, dtype=float)
    for leg_id, (q1, q2, q3) in angles_by_leg.items():
        idx = MOTOR_START_INDEX[leg_id]
        out[idx], out[idx + 1], out[idx + 2] = q1, q2, q3
    return out


def compute_stand_angles(x_off: float = X_OFF, z_off: float = Z_OFF, h: float = H) -> dict:
    target_z = z_off - h
    return {
        leg_id: leg_ik(x_off, LEG_Y_SIGN[leg_id] * HIP_LENGTH, target_z, leg_id)
        for leg_id in LEG_ORDER
    }


def _check_actuator_type(model: mujoco.MjModel) -> None:
    """Confirma programáticamente (no por lectura del XML) que los 12
    actuadores son <motor> de torque puro: gaintype fijo (ganancia=1,
    mjGAIN_FIXED=0) y biastype nulo (mjBIAS_NONE=0). Un <position> o
    <velocity> tendría biastype=mjBIAS_AFFINE (PD ya incorporado en el
    actuador, no habría que sumar kp/kd manualmente). Si algún día se
    cambia go2.xml a actuadores <position>, este chequeo debe fallar
    y avisar en vez de aplicar un PD duplicado en silencio.
    """
    gaintypes = set(int(model.actuator_gaintype[i]) for i in range(model.nu))
    biastypes = set(int(model.actuator_biastype[i]) for i in range(model.nu))
    print(f"actuator_gaintype (todos): {gaintypes} (0 = mjGAIN_FIXED)")
    print(f"actuator_biastype (todos): {biastypes} (0 = mjBIAS_NONE)")
    if gaintypes == {0} and biastypes == {0}:
        print("-> Actuadores <motor> (torque puro): se aplica PD manual, igual que "
              "unitree_sdk2py_bridge.LowCmdHandler.")
    else:
        raise RuntimeError(
            "Los actuadores de go2.xml no son <motor> de torque puro "
            f"(gaintype={gaintypes}, biastype={biastypes}). El PD manual de "
            "este script asume torque puro; revisar go2.xml antes de confiar "
            "en los resultados de esta simulación."
        )


def load_model() -> mujoco.MjModel:
    """Carga go2.xml una sola vez, con el timestep del simulador real
    (ver nota junto a DT). Reusar el mismo model entre rollouts (p. ej.
    en cpg/cost_function.py durante iCEM) evita re-parsear el XML en
    cada evaluación -- MjData sí se crea nuevo por rollout (barato,
    aísla el estado entre evaluaciones).
    """
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.opt.timestep = DT
    _check_actuator_type(model)
    return model


def run_rollout(
    model: mujoco.MjModel,
    params: dict,
    sim_duration: float = SIM_DURATION_S,
    ramp_time: float = RAMP_TIME,
    dt: float = DT,
    kp: float = KP,
    kd: float = KD,
    x_off: float = X_OFF,
    z_off: float = Z_OFF,
    h: float = H,
    record_history: bool = False,
) -> dict:
    """Corre un rollout del pipeline oscillator -> foot_mapping ->
    leg_ik desde la pose "home" y devuelve métricas de desplazamiento
    en x. Es la función que reusan tanto main() (diagnóstico/gráfico)
    como cpg/cost_function.py (evaluación de candidatos en iCEM).

    Args:
        model: modelo de load_model() (se reusa entre llamadas).
        params: dict con cualquiera de "l_step", "l_clrnc", "l_pntr",
            "omega", "mu", "alpha" -- los que falten usan el valor de
            referencia del módulo (L_STEP, L_CLRNC, etc.). Así iCEM
            puede pasar solo el subconjunto de parámetros que está
            optimizando.
        record_history: si True, guarda t_hist/x_hist completos (para
            graficar). iCEM no lo necesita en cada evaluación -- correr
            miles de rollouts sin acumular arrays es más rápido.

    Returns:
        dict con x0, x_ramp_end, x_final, delta_x_total, delta_x_gait,
        y t_hist/x_hist (None si record_history=False).
    """
    data = mujoco.MjData(model)
    num_motor = model.nu

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    l_step = params.get("l_step", L_STEP)
    l_clrnc = params.get("l_clrnc", L_CLRNC)
    l_pntr = params.get("l_pntr", L_PNTR)
    omega = params.get("omega", OMEGA)
    mu = params.get("mu", MU)
    alpha = params.get("alpha", ALPHA)

    cpg = QuadrupedCPG(alpha=alpha, omega=omega, mu=mu)
    stand_pos = _angles_to_motor_array(compute_stand_angles(x_off, z_off, h))
    # Ángulos de las 12 juntas en la pose "home" (después de las 7
    # componentes del freejoint: 3 de posición + 4 de cuaternión).
    rest_pos = data.qpos[7 : 7 + num_motor].copy()

    n_steps = int(sim_duration / dt)
    ramp_end_step = min(int(ramp_time / dt), n_steps - 1)
    t_hist = np.zeros(n_steps) if record_history else None
    x_hist = np.zeros(n_steps) if record_history else None

    x0 = float(data.qpos[0])
    x_ramp_end = x0
    for step in range(n_steps):
        t = step * dt

        if t < ramp_time:
            phase = np.tanh(t / (ramp_time / 3.0))
            q_target = phase * stand_pos + (1.0 - phase) * rest_pos
        else:
            cpg.step(dt, method="rk4")
            angles_by_leg = {}
            for leg_id, (r, theta) in cpg.state().items():
                x, z = foot_position(
                    r, theta, l_step, l_clrnc, l_pntr, h, x_off=x_off, z_off=z_off
                )
                y = LEG_Y_SIGN[leg_id] * HIP_LENGTH
                angles_by_leg[leg_id] = leg_ik(x, y, z, leg_id)
            q_target = _angles_to_motor_array(angles_by_leg)

        # PD manual, igual que unitree_sdk2py_bridge.LowCmdHandler
        # (ctrl = tau_ff + kp*(q_target - q_actual) + kd*(dq_target -
        # dq_actual)), leyendo pos/vel actuales de los sensores
        # jointpos/jointvel (go2.xml líneas 238-262), no de qpos/qvel
        # directamente, para no depender del orden de las juntas en el
        # árbol cinemático (que es FL,FR,RL,RR) -- los sensores están
        # declarados en el mismo orden que los actuadores (FR,FL,RR,RL),
        # que es lo que MOTOR_START_INDEX asume.
        q_actual = data.sensordata[:num_motor]
        dq_actual = data.sensordata[num_motor : 2 * num_motor]
        data.ctrl[:] = kp * (q_target - q_actual) + kd * (0.0 - dq_actual)

        mujoco.mj_step(model, data)

        if step == ramp_end_step:
            x_ramp_end = float(data.qpos[0])
        if record_history:
            t_hist[step] = t
            x_hist[step] = data.qpos[0]

    x_final = float(data.qpos[0])
    return {
        "x0": x0,
        "x_ramp_end": x_ramp_end,
        "x_final": x_final,
        "delta_x_total": x_final - x0,
        # Delta "solo gait": desde el fin del ramp-in hasta el final.
        # Aísla el efecto del CPG oscilando del transitorio de pararse
        # desde la pose home (que mueve el CoM en x igual sin importar
        # los parámetros del CPG, y puede enmascarar la tendencia real
        # del gait en corridas cortas si se mezcla con el delta total).
        "delta_x_gait": x_final - x_ramp_end,
        "t_hist": t_hist,
        "x_hist": x_hist,
    }


def main() -> None:
    model = load_model()
    result = run_rollout(model, params={}, record_history=True)

    print(f"x inicial (CoM, mundo):        {result['x0']:.4f} m")
    print(f"x al fin del ramp-in (t={RAMP_TIME:.1f}s): {result['x_ramp_end']:.4f} m")
    print(f"x final   (CoM, mundo):        {result['x_final']:.4f} m")
    print(f"delta x total                : {result['delta_x_total']:+.4f} m")
    print(f"delta x solo-gait (post-ramp): {result['delta_x_gait']:+.4f} m")
    delta_x_gait = result["delta_x_gait"]
    if delta_x_gait > 1e-4:
        print("RESULTADO: el robot AVANZÓ durante el gait (x post-ramp aumentó) -> camina hacia adelante.")
    elif delta_x_gait < -1e-4:
        print("RESULTADO: el robot RETROCEDIÓ durante el gait (x post-ramp disminuyó) -> camina hacia atrás.")
    else:
        print("RESULTADO: sin desplazamiento neto significativo durante el gait.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result["t_hist"], result["x_hist"])
    ax.axvline(RAMP_TIME, color="gray", linestyle="--", linewidth=1, label="fin del ramp-in")
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("x del CoM (m, marco mundo)")
    ax.set_title("cpg/headless_sim.py -- desplazamiento en x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=120)
    print(f"Gráfico guardado en: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
