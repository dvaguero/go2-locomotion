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

Signo de x -- historia completa, revisada (2026-07-22)
-------------------------------------------------------------------
Todos los intentos anteriores de fijar este signo (análisis simbólico,
medición con headless_sim.py, delegarlo a iCEM, y varias pruebas en el
simulador real con GUI que sistemáticamente daban "camina hacia
atrás" para "-", "-0.15" y "+") resultaron estar CONFUNDIDOS por un
bug de fondo, no relacionado con este signo en absoluto: con y=0 fijo
al llamar a leg_ik (ver cpg/run_cpg.py), el robot tiene un polígono de
apoyo lateral casi nulo y se VUELCA DE COSTADO (roll de 0° a ~73° en
2 s) en vez de caminar hacia atrás. Todas esas pruebas -- reales y en
headless_sim.py -- eran mediciones de un robot cayéndose de lado, no
de la dirección real de marcha del CPG. Ver la nota junto a
LEG_Y_SIGN en cpg/run_cpg.py y cpg/headless_sim.py para esa corrección
(offset lateral y=±HIP_LENGTH en vez de y=0).

Con esa corrección aplicada (piso real + offset lateral, roll estable
en ±3°), se volvió a medir el signo con headless_sim.py -- ahora sí
comparando peras con peras, sin el volcamiento de por medio:
    "+": retrocede activamente incluso después del transitorio inicial
         de asentamiento (pendiente en el plateau t=3-6s: -55 mm/s).
    "-": casi neutral después del mismo transitorio (pendiente en el
         plateau: -5 mm/s, ~10x menor) -- no avanza, pero no retrocede
         activamente tampoco.
Por eso el signo quedó en "-" (el original de la literatura): es
claramente el menos malo de los dos con el offset lateral ya aplicado,
aunque NINGUNO de los dos logra avance neto todavía -- eso es un
problema aparte (quizás L_STEP=0.04 sea insuficiente, o falte ajustar
otros parámetros en conjunto, tarea para la que iCEM es apropiado).

IMPORTANTE: esta comparación "-"/"+" es la primera hecha con el
offset lateral ya en su lugar, así que TODAVÍA NO está validada contra
el simulador real con GUI -- todas las pruebas reales anteriores se
hicieron sin el offset lateral, con el robot volcándose, así que no
sirven como validación de este signo. Confirmar "-" en el simulador
real antes de darlo por bueno.
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

    l_step se espera no-negativo (r siempre es >= 0, así que el signo
    de L_step(r) es el signo de l_step, y ese signo ya está resuelto
    por el "+" de la ecuación de x -- ver nota sobre el signo de x en
    el docstring del módulo). cpg/cost_function.py acota
    PARAM_BOUNDS["l_step"] a (0.0, 0.15) para que iCEM optimice solo la
    magnitud del paso, no el signo.
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
