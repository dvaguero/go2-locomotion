"""Test de round-trip FK -> IK -> FK para cpg/inverse_kinematics.py.

Para cada pata, muestrea ángulos articulares dentro de los rangos
físicos reales del GO2 (go2.xml, clases "abduction"/"front_hip"/
"back_hip"/"knee"), calcula la posición del pie con
forward_kinematics, resuelve leg_ik para esa posición, y verifica que
forward_kinematics(leg_ik(...)) reproduce el mismo XYZ con un error
menor a 1 mm.

Se compara la posición reconstruida (no los ángulos) porque ese es el
criterio correcto para validar un IK: lo que importa es que el pie
llegue al punto pedido, no que se recupere el mismo vector de ángulos
usado para generar el punto de prueba.

Ejecutar con: python cpg/test_ik.py
"""

import math

from inverse_kinematics import forward_kinematics, leg_ik

# Rangos articulares tomados de go2.xml (radianes):
ABDUCTION_RANGE = (-1.0472, 1.0472)  # línea 12
FRONT_HIP_RANGE = (-1.5708, 3.4907)  # línea 16 (FL, FR)
BACK_HIP_RANGE = (-0.5236, 4.5379)  # línea 19 (RL, RR)
KNEE_RANGE = (-2.7227, -0.83776)  # línea 23 (las 4 patas)

LEGS = ("FL", "FR", "RL", "RR")
TOLERANCE_M = 1e-3  # 1 mm


def _hip_flexion_range(leg_id: str) -> tuple[float, float]:
    return FRONT_HIP_RANGE if leg_id in ("FL", "FR") else BACK_HIP_RANGE


def _sample_angles(leg_id: str) -> list[tuple[float, float, float]]:
    """Genera ángulos de prueba dentro del rango físico de cada junta."""
    ab_lo, ab_hi = ABDUCTION_RANGE
    hip_lo, hip_hi = _hip_flexion_range(leg_id)
    knee_lo, knee_hi = KNEE_RANGE

    fracs = (0.25, 0.5, 0.75)
    samples = []
    for f1 in fracs:
        for f2 in fracs:
            for f3 in fracs:
                q1 = ab_lo + f1 * (ab_hi - ab_lo)
                q2 = hip_lo + f2 * (hip_hi - hip_lo)
                q3 = knee_lo + f3 * (knee_hi - knee_lo)
                samples.append((q1, q2, q3))
    return samples


def run() -> bool:
    all_ok = True
    max_error = 0.0
    n_checked = 0

    for leg_id in LEGS:
        leg_max_error = 0.0
        for angles in _sample_angles(leg_id):
            x, y, z = forward_kinematics(angles, leg_id)
            ik_angles = leg_ik(x, y, z, leg_id)
            x2, y2, z2 = forward_kinematics(ik_angles, leg_id)

            error = math.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
            leg_max_error = max(leg_max_error, error)
            max_error = max(max_error, error)
            n_checked += 1

            if error >= TOLERANCE_M:
                all_ok = False
                print(
                    f"[FAIL] {leg_id} angles={tuple(round(a, 4) for a in angles)} "
                    f"xyz=({x:.4f},{y:.4f},{z:.4f}) "
                    f"ik={tuple(round(a, 4) for a in ik_angles)} "
                    f"xyz2=({x2:.4f},{y2:.4f},{z2:.4f}) error={error * 1000:.4f} mm"
                )

        print(f"{leg_id}: {len(_sample_angles(leg_id))} puntos, error máximo = "
              f"{leg_max_error * 1000:.6f} mm")

    print(f"\nTotal puntos verificados: {n_checked}")
    print(f"Error máximo global: {max_error * 1000:.6f} mm (tolerancia: {TOLERANCE_M * 1000:.1f} mm)")
    print("RESULTADO:", "OK" if all_ok else "FALLÓ")
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
