"""Controlador CPG (Hopf oscillator + IK) para el GO2 en unitree_mujoco.

Cablea el pipeline oscillator.py -> foot_mapping.py -> inverse_kinematics.py
y envía los 12 ángulos resultantes al simulador MuJoCo de Unitree vía
unitree_sdk2py, siguiendo la misma interfaz de conexión/publicación que
external/unitree_mujoco/example/python/stand_go2.py.

Prerrequisitos (no incluidos en este repo, instalar aparte):
    pip install mujoco
    pip install -e external/unitree_sdk2_python
Luego, en otra terminal, levantar el simulador:
    python external/unitree_mujoco/simulate_python/unitree_mujoco.py
y recién entonces correr este script:
    python cpg/run_cpg.py

Estado actual: ENABLE_OSCILLATION = False.
Esta es la "prueba de pie quieto" (smoke test) pedida antes de agregar
oscilación: el robot debe pararse y quedar estable, sin moverse. Con
ENABLE_OSCILLATION = False el oscilador NUNCA se integra (no se llama
QuadrupedCPG.step), así que su estado queda congelado en (r=0,
theta=offset_inicial) para siempre. Con r=0, foot_mapping.foot_position
da x = X_OFF y z = Z_OFF - H para las 4 patas sin importar theta (el
término L_step(r=0) se anula), o sea un objetivo de pie fijo y
simétrico: parado quieto, sin oscilar. Cuando esa prueba quede
validada, cambiar ENABLE_OSCILLATION a True para empezar a caminar.

Sobre Z_OFF/H: para la prueba de pie quieto, la altura efectiva del
pie respecto a la cadera es (Z_OFF - H). Se pidió partir de
aproximadamente -0.3 m. Sin embargo, al validar
inverse_kinematics.forward_kinematics(...) contra el propio
stand_up_joint_pos de stand_go2.py (la pose de pie oficial de Unitree
para este mismo robot, mismos parámetros DH), el resultado es
z ≈ -0.35 m (no -0.30 m) — ver el mensaje de la sesión para el detalle
del cálculo. Por eso H = 0.35 aquí por default: es una postura más
extendida que el punto de partida sugerido, pero respaldada por la
pose oficial del fabricante en vez de una estimación aproximada. No
pude ejecutar el simulador gráfico en este entorno (requiere
mujoco.viewer con display, no disponible en esta sesión) para
confirmarlo visualmente — si el robot igual se ve muy estirado o
inestable al probarlo, bajar H hacia 0.30-0.32 y volver a probar; si
se ve muy agachado, no debería pasar con este valor pero subir H un
poco más también es válido.
"""

import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

from foot_mapping import foot_position
from inverse_kinematics import HIP_LENGTH, leg_ik
from oscillator import QuadrupedCPG

# =============================================================================
# Parámetros ajustables
# =============================================================================

# --- Oscilador de Hopf (cpg/oscillator.py) ---
ALPHA = 50.0  # ganancia de convergencia de r (rad/s). Más alto = r
# converge más rápido a MU, pero exige DT más chico para que la
# integración (incluso RK4) siga siendo estable/precisa.
OMEGA = 2.0 * np.pi * 1.5  # frecuencia angular de la fase (rad/s).
# 1.5 Hz de cadencia de paso es un punto de partida razonable para un
# trote lento del GO2; subir para caminar más rápido.
MU = 1.0  # amplitud objetivo de r en steady-state (adimensional, ver
# foot_mapping._step_length: L_step(r) = L_STEP * r, con r -> MU).

# --- Mapeo a task space (cpg/foot_mapping.py) ---
# L_STEP=0.0 (experimento decisivo anterior) confirmó que el temblor y
# la caída NO vienen de foot_mapping.py -- persistían incluso sin
# movimiento horizontal del pie. Con kp/kd ya corregidos (ver nota en
# KP_STAND/KD_STAND) pero el temblor todavía presente, se prueba ahora
# con un L_STEP chico (2026-07-22) para dar el primer paso posible con
# un ciclo conservador antes de reintroducir cualquier parámetro
# optimizado por iCEM (que se optimizó contra headless_sim.py, no
# validado como fiel al simulador real para estos parámetros tampoco).
L_STEP = 0.04  # longitud de paso máxima (m) en steady-state (r=MU).
L_CLRNC = 0.04  # altura de despegue del pie en fase swing (m).
L_PNTR = 0.005  # penetración/asentamiento del pie en fase stance (m).
H = 0.35  # altura nominal de la cadera sobre el suelo (m). Ver nota
# sobre Z_OFF/H en el docstring del módulo: 0.35 viene de
# cross-validar contra la pose de pie oficial de stand_go2.py, no de
# la estimación inicial de -0.3 m.
X_OFF = 0.0  # offset horizontal del centro del área de paso (m).
Z_OFF = 0.0  # offset vertical de calibración fino (m); z efectivo de
# pie quieto = Z_OFF - H.

# --- Control de bajo nivel ---
DT = 0.002  # paso de control (s); igual al dt de stand_go2.py.
# KP_STAND/KD_STAND corregidos (2026-07-22) para igualar los valores
# que usa external/unitree_mujoco/example/python/stand_go2.py en su
# postura final sostenida (kp=50.0, kd=3.5, líneas 66-78) -- ejemplo
# oficial de Unitree, probado, que sostiene al robot de pie sin
# problema. Antes este script usaba 60.0/5.0, valores propios sin
# comparar contra ninguna referencia real; con las patas oscilando
# (no solo estáticas como en stand_go2.py) unas ganancias más
# agresivas que la referencia probada son candidato razonable para
# temblor/inestabilidad.
KP_STAND = 50.0  # ganancia proporcional en la postura final (parado/caminando).
KD_STAND = 3.5  # ganancia derivativa en la postura final.
KP_REST = 20.0  # ganancia proporcional durante el ramp-in (más blanda).
KD_REST = 3.5  # ganancia derivativa durante el ramp-in (más blanda,
# mismo criterio que KP_REST: menos rigidez mientras el robot todavía
# no llega a la postura final, para no frenar bruscamente el
# movimiento de la interpolación).
RAMP_TIME = 2.0  # duración del ramp-in desde la pose de reposo (s).

# --- Interruptor de la prueba de pie quieto vs. caminata ---
# True TEMPORAL (2026-07-22): hace falta que el oscilador esté
# integrando para el experimento decisivo de L_STEP=0.0 -- si no,
# no hay ni rebote vertical que observar. Revertir a False si se
# vuelve a necesitar la prueba de pie quieto pura.
ENABLE_OSCILLATION = True  # False: prueba de pie quieto (sin integrar
# el oscilador). True: caminata con el CPG completo.

# =============================================================================
# Mapeo pata -> índices del arreglo motor_cmd (12,), convención de
# external/unitree_sdk2_python/unitree_sdk2py/test/lowlevel/unitree_go2_const.py:
# FR_0=0, FR_1=1, FR_2=2, FL_0=3, ..., RR_0=6, ..., RL_0=9, ...
# (orden hip, thigh, calf dentro de cada bloque de 3)
# =============================================================================
MOTOR_START_INDEX = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
LEG_ORDER = ("FR", "FL", "RR", "RL")

# Y_OFFSET != 0 (2026-07-22): antes se llamaba a leg_ik con y=0.0 fijo
# (pie directamente bajo el eje de abducción). Se encontró -- via
# headless_sim.py ya con piso real, ver su docstring -- que el robot
# no camina hacia atrás sino que se VUELCA DE COSTADO (roll de 0° a
# ~73° en 2s), y que con y=0 las 4 patas quedan casi en línea, con un
# polígono de apoyo lateral casi nulo: cualquier asimetría mínima
# entre patas lo vuelca sin que haya ningún feedback de balance que lo
# corrija (el CPG es puramente open-loop). Separar los pies hacia
# afuera por HIP_LENGTH (la misma distancia geométrica del offset de
# cadera, ver inverse_kinematics.py) les da un ancho de sustentación
# lateral real. Signo por pata: izquierda (FL, RL) positivo, derecha
# (FR, RR) negativo -- mismo criterio de lado que
# inverse_kinematics._hip_sign.
LEG_Y_SIGN = {"FL": 1.0, "RL": 1.0, "FR": -1.0, "RR": -1.0}

# Pose de reposo/inicio de la simulación: keyframe "home" de
# external/unitree_mujoco/unitree_robots/go2/go2.xml (línea 286),
# (hip, thigh, calf) = (0, 0.9, -1.8) para las 4 patas.
REST_HIP, REST_THIGH, REST_CALF = 0.0, 0.9, -1.8


def _angles_to_motor_array(angles_by_leg: dict) -> np.ndarray:
    """Arma el vector (12,) de motor_cmd a partir de {leg_id: (q1,q2,q3)}."""
    out = np.zeros(12, dtype=float)
    for leg_id, (q1, q2, q3) in angles_by_leg.items():
        idx = MOTOR_START_INDEX[leg_id]
        out[idx], out[idx + 1], out[idx + 2] = q1, q2, q3
    return out


def compute_stand_angles() -> dict:
    """Ángulos (q1,q2,q3) por pata para el objetivo de pie quieto
    (r=0, x=X_OFF, y=±HIP_LENGTH, z=Z_OFF-H), vía leg_ik.
    """
    target_z = Z_OFF - H
    return {
        leg_id: leg_ik(X_OFF, LEG_Y_SIGN[leg_id] * HIP_LENGTH, target_z, leg_id)
        for leg_id in LEG_ORDER
    }


def compute_cpg_angles(cpg: QuadrupedCPG, dt: float, step_oscillator: bool) -> dict:
    """Un paso del pipeline oscillator -> foot_mapping -> leg_ik.

    Si step_oscillator es False, el estado del CPG no se integra
    (queda congelado) y el resultado es el mismo objetivo estático de
    compute_stand_angles() -- es la prueba de pie quieto.
    """
    if step_oscillator:
        cpg.step(dt, method="rk4")

    angles_by_leg = {}
    for leg_id, (r, theta) in cpg.state().items():
        x, z = foot_position(r, theta, L_STEP, L_CLRNC, L_PNTR, H, x_off=X_OFF, z_off=Z_OFF)
        y = LEG_Y_SIGN[leg_id] * HIP_LENGTH
        angles_by_leg[leg_id] = leg_ik(x, y, z, leg_id)
    return angles_by_leg


def main() -> None:
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    crc = CRC()
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    rest_pos = np.array([REST_HIP, REST_THIGH, REST_CALF] * 4, dtype=float)
    stand_angles_by_leg = compute_stand_angles()
    stand_pos = _angles_to_motor_array(stand_angles_by_leg)

    print("Objetivo de pie quieto (leg_id: q_abduccion, q_cadera, q_rodilla):")
    for leg_id in LEG_ORDER:
        print(f"  {leg_id}: {tuple(round(a, 4) for a in stand_angles_by_leg[leg_id])}")
    print(f"z efectivo de pie quieto (Z_OFF - H) = {Z_OFF - H:.3f} m")
    print(f"ENABLE_OSCILLATION = {ENABLE_OSCILLATION}")

    cpg = QuadrupedCPG(alpha=ALPHA, omega=OMEGA, mu=MU)

    input("Presiona enter para iniciar el ramp-in a la postura de pie...")

    running_time = 0.0
    step_count = 0
    while True:
        step_start = time.perf_counter()
        running_time += DT
        step_count += 1

        if running_time < RAMP_TIME:
            # Ramp-in: interpolación tanh de reposo -> pie quieto,
            # mismo esquema que stand_go2.py (transición suave, sin
            # saltos bruscos que puedan hacer caer al robot).
            phase = np.tanh(running_time / (RAMP_TIME / 3.0))
            q_target = phase * stand_pos + (1 - phase) * rest_pos
            kp = phase * KP_STAND + (1 - phase) * KP_REST
            kd = phase * KD_STAND + (1 - phase) * KD_REST
        else:
            angles_by_leg = compute_cpg_angles(cpg, DT, ENABLE_OSCILLATION)
            q_target = _angles_to_motor_array(angles_by_leg)
            kp = KP_STAND
            kd = KD_STAND

        for i in range(12):
            cmd.motor_cmd[i].q = float(q_target[i])
            cmd.motor_cmd[i].kp = float(kp)
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = float(kd)
            cmd.motor_cmd[i].tau = 0.0

        # DEBUG temporal: confirmar en vivo que q_target avanza y que
        # cmd.motor_cmd[0].q realmente recibe ese valor antes de
        # publicar. Quitar una vez diagnosticado el problema de
        # movimiento nulo.
        if step_count % 50 == 0:
            print(
                f"[debug] step={step_count} t={running_time:.3f}s "
                f"q_target[0]={q_target[0]:.5f} "
                f"cmd.motor_cmd[0].q={cmd.motor_cmd[0].q:.5f} "
                f"kp={kp:.2f} kd={kd:.2f}"
            )

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time_until_next_step = DT - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
