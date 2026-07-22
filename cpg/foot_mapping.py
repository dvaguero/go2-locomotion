"""Mapeo de coordenadas del oscilador (r, θ) a la posición cartesiana
del pie, en el marco de la cadera de esa pata.

Formulación basada en Shafiee, Bellegarda & Ijspeert (2023),
"ManyQuadrupeds: Learning a Single Locomotion Policy for Diverse
Quadruped Robots" — referencia [6] del proyecto (misma fuente que
cpg/oscillator.py).

    x = x_off - L_step(r)·cos(θ)
    z = z_off - h + L_clrnc·sin(θ)   si sin(θ) > 0   (fase swing)
    z = z_off - h + L_pntr·sin(θ)    en caso contrario (fase stance)

θ recorre un ciclo completo [0, 2π) por paso. sin(θ) > 0 es la mitad
del ciclo en que el pie está en el aire avanzando hacia adelante fase
"swing"); sin(θ) <= 0 es la mitad en que el pie está apoyado en el
suelo, empujando el cuerpo hacia adelante (fase "stance").
"""

import math

# --- Significado físico de cada parámetro -----------------------------
#
# L_step (m): longitud de paso máxima. Escala cuánto se desplaza el
#   pie hacia adelante/atrás en x durante un ciclo (ver _step_length).
#   Se modula por r (amplitud del oscilador) para que el paso "crezca"
#   gradualmente desde 0 mientras r converge a μ (arranque suave de la
#   marcha) y pueda reducirse a 0 en tiempo real bajando μ (frenado
#   suave), sin discontinuidades en la trayectoria del pie.
#
# L_clrnc (m): "clearance" — altura máxima que se levanta el pie del
#   suelo durante la fase de swing (sin(θ) > 0). Debe ser suficiente
#   para que el pie no arrastre ni tropiece con el terreno al moverse
#   hacia adelante.
#
# L_pntr (m): "penetration" — desplazamiento vertical (hacia abajo,
#   valor típicamente pequeño) del pie durante la fase de stance
#   (sin(θ) <= 0). No es una penetración física real del suelo: en la
#   simulación el contacto rígido del suelo impide que el pie baje más
#   de la cuenta; este término modela un pequeño "asentamiento" de la
#   trayectoria de referencia (compliance) que ayuda a mantener el
#   contacto y suavizar la transición swing→stance. En general
#   L_pntr << L_clrnc (a veces incluso 0).
#
# h (m): altura nominal de la cadera sobre el suelo (altura del pie
#   con la pata extendida hacia abajo, postura de reposo/estar de
#   pie). Define el nivel base de z desde el cual se miden clearance y
#   penetration (-h es "pie apoyado en el suelo, sin swing ni stance
#   añadidos").
#
# x_off (m): offset horizontal del centro del área de paso respecto al
#   origen de la cadera (eje de abducción, mismo origen que
#   inverse_kinematics.leg_ik). Permite desplazar el punto medio de la
#   trayectoria del pie hacia adelante o atrás, por ejemplo para
#   compensar la posición del centro de masa del cuerpo.
#
# z_off (m): offset vertical adicional sobre la altura nominal -h.
#   Ajuste fino de calibración (p. ej. compensar holguras mecánicas o
#   adaptar la altura de pie) sin tener que redefinir h.


def _step_length(r: float, l_step: float) -> float:
    """L_step(r): longitud de paso efectiva, escalada por la amplitud
    del oscilador r.

    Se usa una relación lineal L_step(r) = l_step · r: es la opción
    más simple, consistente con el rol de r como amplitud del Hopf
    oscillator (r → μ en steady-state, ver oscillator.HopfOscillator).
    Con μ = 1 (convención típica), l_step es directamente la longitud
    de paso máxima alcanzada en steady-state; con otro μ, r converge a
    ese valor y L_step(r) escala proporcionalmente.
    """
    return l_step * r


def foot_position(
    r: float,
    theta: float,
    l_step: float,
    l_clrnc: float,
    l_pntr: float,
    h: float,
    x_off: float = 0.0,
    z_off: float = 0.0,
) -> tuple[float, float]:
    """Mapea (r, θ) del oscilador de una pata a (x, z) del pie.

    (x, z) queda en el marco de la cadera de esa pata (mismo marco que
    cpg/inverse_kinematics.py: x adelante, z arriba).

    Nota sobre y (lateral): este es un oscilador 2D (caminata solo
    hacia adelante, ver CLAUDE.md) y no produce la coordenada y. Al
    llamar a inverse_kinematics.leg_ik(x, y, z, leg_id) con el
    resultado de esta función, usar y = 0.0 fijo — el pie se mantiene
    directamente bajo el eje de abducción de la cadera, sin
    movimiento lateral.

    Args:
        r: amplitud actual del oscilador (oscillator.HopfOscillator.r).
        theta: fase actual del oscilador, en radianes
            (oscillator.HopfOscillator.theta).
        l_step, l_clrnc, l_pntr, h, x_off, z_off: parámetros descritos
            arriba.

    Returns:
        (x, z) en metros.
    """
    x = x_off - _step_length(r, l_step) * math.cos(theta)

    sin_theta = math.sin(theta)
    if sin_theta > 0.0:
        z = z_off - h + l_clrnc * sin_theta  # swing: pie en el aire
    else:
        z = z_off - h + l_pntr * sin_theta  # stance: pie apoyado
    return x, z
