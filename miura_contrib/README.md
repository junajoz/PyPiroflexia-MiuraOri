# miura_contrib

Esta carpeta contiene los códigos nuevos desarrollados para extender el análisis de estructuras Miura-Ori dentro de PyPiroflexia.

El objetivo de este módulo es permitir experimentos de sensibilidad geométrica y mecánica sin modificar directamente todos los archivos del solver original. Los scripts se organizan alrededor de configuraciones Miura nominales, defectuosas o con parámetros variables.

## Contenido

### Configuración geométrica

- `ConfigMiuraFromAB.py`  
  Construye una geometría Miura-Ori a partir de campos celulares `a_cells` y `b_cells`. Permite que cada celda tenga valores distintos de los parámetros geométricos `a` y `b`.

- `ConfigMiuraRandomAB.py`  
  Genera configuraciones Miura-Ori con variaciones aleatorias en los parámetros celulares `a` y `b`.

- `defect_fields.py`  
  Contiene funciones auxiliares para construir campos nominales y aplicar defectos locales, defectos por fila, defectos por columna o defectos tipo clúster.

### Drivers de experimentación

- `Miura_Folding_random.py`  
  Introduce perturbaciones nodales aleatorias sobre una geometría Miura nominal.

- `Miura_Folding_random_ab.py`  
  Evalúa la respuesta de una estructura Miura cuando los parámetros celulares `a` y `b` varían aleatoriamente.

- `Miura_Folding_single_defect.py`  
  Compara una estructura nominal con una estructura que contiene un defecto geométrico localizado.

- `Miura_Folding_defect_scan.py`  
  Realiza un barrido de defectos en distintas posiciones de la teselación para estimar sensibilidad espacial.

- `Miura_Folding_seam_defect.py`  
  Modela defectos tipo costura, entendidos como franjas de celdas alteradas por fila o columna.

- `Miura_Folding_distributed_load.py`  
  Compara diferentes esquemas de carga nodal aplicados sobre la estructura Miura.

- `Miura_Folding_cyclic.py`  
  Implementa ciclos de carga y descarga para explorar la posible aparición de lazos de histéresis energética.

- `Miura_Folding_varKpf.py`  
  Evalúa estructuras Miura con rigidez de pliegue variable espacialmente.

### Soporte para rigidez variable

- `PrepareData_varKpf.py`  
  Versión extendida de `PrepareData.py` que permite usar una rigidez de pliegue `kpf` variable por pliegue.

- `PathAnalysis_varKpf.py`  
  Versión de análisis incremental compatible con campos de rigidez de pliegue no uniformes.

- `GlobalK_fast_ver_varKpf.py`  
  Ensamblador rápido adaptado para trabajar con rigidez variable por pliegue.

- `GlobalK_edu_ver_varKpf.py`  
  Ensamblador educativo corregido para usar el valor de rigidez correspondiente a cada pliegue.

## Ejecución

Los scripts están pensados para ejecutarse dentro del entorno del repositorio PyPiroflexia, conservando las importaciones planas del código original.

Ejemplo:

```bash
cd python/miura_contrib
PYTHONPATH="..:." python Miura_Folding_single_defect.py
