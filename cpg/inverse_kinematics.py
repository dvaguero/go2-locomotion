"""Cinemática directa e inversa de una pata del Unitree GO2.

Dimensiones y convenio de ejes extraídos de
external/unitree_mujoco/unitree_robots/go2/go2.xml (ver cita en cada
constante y en las funciones). Marco de referencia: x hacia adelante,
y hacia la izquierda, z hacia arriba — es el marco del body
"base_link" del XML, ya que los bodies de cadera (FL_hip, FR_hip,
RL_hip, RR_hip) no tienen atributo `quat`, solo `pos`, por lo que
heredan la orientación del cuerpo.
"""

import math

# --- Dimensiones del GO2 (metros), tomadas de go2.xml ----------------------
# body "FL_thigh" pos="0 0.0955 0" (línea 86): offset del eje de
# abducción de la cadera al eje de flexión (muslo), a lo largo del eje
# Y local de la cadera.
HIP_LENGTH = 0.0955

# body "FL_calf" pos="0 0 -0.213" (línea 95): longitud del muslo, del
# eje de flexión de cadera al eje de la rodilla.
THIGH_LENGTH = 0.213

# body "FL_foot" pos="0 0 -0.213" (línea 108): longitud de la
# pantorrilla (calf), del eje de la rodilla al punto de contacto del
# pie.
CALF_LENGTH = 0.213

# go2.xml líneas 78/112/147/182: FL_hip y RL_hip están desplazados en
# +Y (izquierda) respecto al centro del cuerpo, FR_hip y RR_hip en -Y
# (derecha). El mismo signo se propaga al offset de cadera (HIP_LENGTH)
# de cada pata, porque el body "*_thigh" hereda ese lado.
_LEFT_LEGS = ("FL", "RL")
_RIGHT_LEGS = ("FR", "RR")


def _hip_sign(leg_id: str) -> float:
    """Signo del offset de cadera (HIP_LENGTH) según el lado de la pata."""
    if leg_id in _LEFT_LEGS:
        return 1.0
    if leg_id in _RIGHT_LEGS:
        return -1.0
    raise ValueError(f"leg_id inválido: {leg_id!r} (usar 'FL', 'FR', 'RL' o 'RR')")


def forward_kinematics(
    angles: tuple[float, float, float], leg_id: str
) -> tuple[float, float, float]:
    """Cinemática directa: ángulos articulares -> posición del pie.

    Args:
        angles: (q_hip_abduction, q_hip_flexion, q_knee) en radianes,
            con el mismo convenio de signo que las juntas del XML:
            - q_hip_abduction: rotación en torno al eje X local de la
              cadera (joint class "abduction", axis="1 0 0",
              go2.xml línea 12).
            - q_hip_flexion: rotación en torno al eje Y, aplicada
              después de la abducción (joint class "hip"/"front_hip"/
              "back_hip"; eje Y es el default del modelo,
              go2.xml línea 9, ninguna de esas clases lo sobreescribe).
            - q_knee: rotación en torno al eje Y, aplicada después de
              la flexión de cadera (joint class "knee", eje Y por
              default).
        leg_id: "FL", "FR", "RL" o "RR".

    Returns:
        (x, y, z): posición del pie respecto al origen de la cadera de
        esa pata (el eje de abducción), en el marco del cuerpo
        (x adelante, y izquierda, z arriba).

    Derivación:
        La cadena cinemática de una pata (ver body "FL_hip" ->
        "FL_thigh" -> "FL_calf" -> "FL_foot" en go2.xml, líneas
        78-108) es:

            cadera --Rx(q1)--> trasladar l1 en Y --Ry(q2)-->
            trasladar l2 en -Z --Ry(q3)--> trasladar l3 en -Z --> pie

        Como las dos rotaciones Ry (flexión de cadera y rodilla) son
        sobre el mismo eje, se componen sumando los ángulos
        (q2 + q3). Componiendo las traslaciones rotadas se obtiene:

            A = l2*cos(q2) + l3*cos(q2 + q3)
            x = -l2*sin(q2) - l3*sin(q2 + q3)
            y =  l1*cos(q1) + A*sin(q1)
            z =  l1*sin(q1) - A*cos(q1)

        con l1 = HIP_LENGTH con signo según el lado de la pata
        (_hip_sign). "A" es la distancia efectiva del muslo+pantorrilla
        proyectada en el plano perpendicular al eje de abducción; leg_ik
        la recupera a partir de (y, z) porque una rotación pura
        preserva la norma del vector que rota.
    """
    q1, q2, q3 = angles
    l1 = _hip_sign(leg_id) * HIP_LENGTH
    l2 = THIGH_LENGTH
    l3 = CALF_LENGTH

    a = l2 * math.cos(q2) + l3 * math.cos(q2 + q3)
    x = -l2 * math.sin(q2) - l3 * math.sin(q2 + q3)
    y = l1 * math.cos(q1) + a * math.sin(q1)
    z = l1 * math.sin(q1) - a * math.cos(q1)
    return x, y, z


def leg_ik(x: float, y: float, z: float, leg_id: str) -> tuple[float, float, float]:
    """Cinemática inversa analítica de una pata del GO2.

    Args:
        x, y, z: posición cartesiana deseada del pie, en metros,
            respecto al origen de la cadera de la pata `leg_id` (el
            eje de abducción), en el marco del cuerpo (x adelante,
            y izquierda, z arriba — mismo convenio que
            forward_kinematics).
        leg_id: "FL", "FR", "RL" o "RR".

    Returns:
        (q_hip_abduction, q_hip_flexion, q_knee) en radianes, con el
        mismo convenio de signo que las juntas del XML (ver
        forward_kinematics).

    Raises:
        ValueError: si (y, z) no es alcanzable con el offset de cadera
            de esa pata (sqrt(y^2+z^2) < |HIP_LENGTH|), o si leg_id no
            es válido.

    Razonamiento (inversa de la derivación en forward_kinematics):

    1) Abducción (q1). El offset de cadera l1 y el "alcance" A
       determinan (y, z) mediante una rotación pura en torno a X:
       (y, z) = Rot_x(q1) · (l1, -A). Una rotación preserva la norma,
       así que A = sqrt(y^2 + z^2 - l1^2) (raíz positiva: A es una
       suma de longitudes de eslabones proyectadas, físicamente >= 0
       para una pata extendida hacia abajo). Luego q1 se obtiene
       comparando el ángulo de (y, z) con el de (l1, -A):
       q1 = atan2(z, y) + atan2(A, l1).

    2) Rodilla (q3). Una vez descontado el efecto de la abducción, el
       problema se reduce a un brazo planar de 2 eslabones (l2, l3)
       que debe alcanzar el punto (x, A). Por el teorema del coseno,
       con L^2 = x^2 + A^2:
           cos(q3) = (L^2 - l2^2 - l3^2) / (2*l2*l3)
       Se toma q3 = -acos(...) (raíz negativa) porque el rango de la
       junta "knee" en el XML es siempre negativo (-2.7227 a -0.83776
       rad, go2.xml línea 23): la rodilla del GO2 solo dobla "hacia
       atrás" (como la de cualquier cuadrúpedo), nunca hacia adelante.

    3) Flexión de cadera (q2). Con q3 ya conocido, (A, -x) es también
       una rotación pura Rot(q2) de (P, Q), con
       P = l2 + l3*cos(q3), Q = l3*sin(q3), de donde:
           q2 = atan2(-x, A) - atan2(Q, P)
    """
    l1 = _hip_sign(leg_id) * HIP_LENGTH
    l2 = THIGH_LENGTH
    l3 = CALF_LENGTH

    r_yz_sq = y * y + z * z
    a_sq = r_yz_sq - l1 * l1
    if a_sq < -1e-9:
        raise ValueError(
            f"Punto ({x}, {y}, {z}) inalcanzable para {leg_id}: "
            f"sqrt(y^2+z^2)={math.sqrt(r_yz_sq):.4f} < |HIP_LENGTH|={abs(l1):.4f}"
        )
    a = math.sqrt(max(0.0, a_sq))

    q1 = math.atan2(z, y) + math.atan2(a, l1)

    l_sq = x * x + a * a
    cos_q3 = (l_sq - l2 * l2 - l3 * l3) / (2.0 * l2 * l3)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3 = -math.acos(cos_q3)

    p = l2 + l3 * math.cos(q3)
    q_term = l3 * math.sin(q3)
    q2 = math.atan2(-x, a) - math.atan2(q_term, p)

    return q1, q2, q3
