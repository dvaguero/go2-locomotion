# Proyecto: Locomoción GO2 — CPG+CEM y RL

## Contexto
Proyecto de curso de robótica. Dos enfoques independientes para lograr
caminata de un robot cuadrúpedo Unitree GO2, en simulación MuJoCo.

1. **cpg/**: osciladores en task space (uno por pata) + Cross-Entropy
   Method (iCEM, Pinneri et al. 2020) para ajustar sus parámetros.
   Caminata 2D (solo hacia adelante). Usa unitree_mujoco / unitree_sdk2
   como interfaz de simulación y control (misma interfaz que el robot real).
   Requiere cinemática inversa analítica de la pierna del GO2 (XYZ del
   pie -> 3 ángulos articulares).

2. **rl/**: código base MuJoCo Playground con GO2 ya incorporado
   (carpeta rl/mujoco_playground_el7009_project-main/ — ver
   "Estructura actual" sobre por qué no es lo mismo que
   external/mujoco_playground/). Solo se debe modificar la función de
   recompensa en
   rl/mujoco_playground_el7009_project-main/mujoco_playground/_src/locomotion/go2/joystick.py,
   agregando funciones NUEVAS sin cambiar firmas existentes ni tocar el
   resto del código. Objetivo: caminata omnidireccional (adelante,
   atrás, lateral, rotación). Entrenar con train_go2.py, evaluar con
   evaluate_policy.py.

## Estructura actual
- `external/unitree_mujoco/`: simulador MuJoCo oficial de Unitree, usado
  como entorno de simulación para el enfoque CPG+CEM (cpg/).
- `external/unitree_sdk2_python/`: SDK de Unitree (bindings Python) para
  enviar comandos de control al simulador (o al robot real) con la misma
  interfaz; usado por cpg/.
- `external/mujoco_playground/`: repo oficial de Google DeepMind MuJoCo
  Playground, clonado solo como referencia y para reproducibilidad
  (vía .repos). **No es el código que se usa para entrenar ni donde se
  edita joystick.py** — no modificar nada aquí para el enfoque RL.
- `rl/mujoco_playground_el7009_project-main/`: versión de la cátedra de
  MuJoCo Playground, con el GO2 ya incorporado. **Este es el código que
  realmente se usa** para entrenar (train_go2.py) y evaluar
  (evaluate_policy.py). La función de recompensa a modificar está en
  `rl/mujoco_playground_el7009_project-main/mujoco_playground/_src/locomotion/go2/joystick.py`
  (no en `external/mujoco_playground/...`).

## Reglas para el agente
- No modificar archivos fuera de cpg/ y de la función de recompensa en
  joystick.py, salvo instrucción explícita.
- Python idiomático, con type hints y docstrings breves.
- Cada script debe poder ejecutarse con `python nombre_script.py` desde
  la raíz del repo (o indicar claramente si necesita otra ruta).
- No instalar paquetes globalmente; usar el entorno virtual del proyecto
  (uv / venv).
- Explicar el razonamiento físico/matemático detrás de implementaciones
  de cinemática, osciladores o funciones de costo — no solo entregar
  código, ya que debo justificarlo en el informe.
- Antes de un cambio grande, sugerir hacer commit del estado actual.

## Comandos frecuentes
- Entrenar RL: `python rl/mujoco_playground_el7009_project-main/train_go2.py`
- Evaluar RL: `python rl/mujoco_playground_el7009_project-main/evaluate_policy.py`
- Probar CPG: `python cpg/run_cpg.py`
