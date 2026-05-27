# Módulo Miura-Ori para PyPiroflexia

Este paquete separa el aporte desarrollado sobre PyPiroflexia para estudiar la respuesta cuasiestática de estructuras Miura-Ori con imperfecciones geométricas, defectos localizados, defectos tipo costura, variación espacial de rigidez de pliegue y diferentes distribuciones de carga.

El módulo fue construido comparando el repositorio original (`PyPiroflexia-main.zip`) con la versión modificada (`PyPiroflexia-miura.zip`). Se aislaron únicamente los archivos Python nuevos y se dejaron aparte las modificaciones hechas sobre archivos del núcleo del solver.

## Estructura

```text
PyPiroflexia_miura_module/
├── miura_contrib/          # Códigos nuevos del aporte
├── modified_core/          # Copias completas de archivos originales modificados
├── patches/                # Parches diff para revisar cambios sobre el núcleo
├── tools/                  # Script auxiliar para copiar el módulo a un repo local
├── MANIFEST.md             # Resumen de archivos incluidos
└── README.md
```

## Contenido principal

### 1. Geometría Miura con campos variables

- `ConfigMiuraFromAB.py`: construye una geometría Miura a partir de campos celulares `a_cells` y `b_cells`.
- `ConfigMiuraRandomAB.py`: genera geometrías Miura con variaciones aleatorias en los parámetros de celda `a` y `b`.
- `defect_fields.py`: utilidades para construir campos nominales y aplicar defectos locales, por fila, por columna o por clúster.

### 2. Drivers de experimentación

- `Miura_Folding_random.py`: perturbación nodal aleatoria sobre la geometría nominal.
- `Miura_Folding_random_ab.py`: variación aleatoria de los parámetros `a` y `b` por celda.
- `Miura_Folding_single_defect.py`: comparación entre estructura nominal y estructura con un defecto localizado.
- `Miura_Folding_defect_scan.py`: barrido de defectos locales para estimar sensibilidad según ubicación.
- `Miura_Folding_seam_defect.py`: defectos tipo costura por filas o columnas de celdas.
- `Miura_Folding_distributed_load.py`: comparación entre distintos esquemas de carga nodal.
- `Miura_Folding_cyclic.py`: ciclo de carga/descarga para explorar histéresis energética.
- `Miura_Folding_varKpf.py`: análisis con rigidez de pliegue variable espacialmente.

### 3. Soporte para rigidez de pliegue variable

- `PrepareData_varKpf.py`: versión de preparación de datos que acepta un arreglo `kpf` por pliegue.
- `PathAnalysis_varKpf.py`: análisis incremental usando ensambladores compatibles con `kpf` variable.
- `GlobalK_fast_ver_varKpf.py`: versión rápida con validación de formas para rigidez variable.
- `GlobalK_edu_ver_varKpf.py`: versión educativa corregida para usar `angles['kpf'][fel]` en cada pliegue.

## Cambios hechos sobre archivos originales

Los archivos siguientes no se mezclaron dentro del módulo nuevo. Quedan en `modified_core/` y sus diferencias están en `patches/`.

| Archivo | Cambio principal |
|---|---|
| `Miura_Folding.py` | Agrega perturbación aleatoria en coordenada `z` antes de llamar a `PrepareData`. |
| `PathAnalysis.py` | Reemplaza `.A` por `.toarray()` al resolver sistemas con matrices dispersas. |
| `VisualFold.py` | Activa renderizado `off_screen`, fija cámara y evita abrir ventana interactiva al guardar GIF. |

## Instalación rápida en un repo local

Opción recomendada: copiar el módulo dentro del directorio `python/` del repositorio PyPiroflexia.

```bash
python tools/install_into_repo.py /ruta/a/PyPiroflexia-main/python
```

Esto crea:

```text
/ruta/a/PyPiroflexia-main/python/miura_contrib/
```

Para ejecutar un driver desde esa ubicación:

```bash
cd /ruta/a/PyPiroflexia-main/python/miura_contrib
PYTHONPATH="..:." python Miura_Folding_single_defect.py
```

En Windows PowerShell, el equivalente es:

```powershell
cd C:\ruta\a\PyPiroflexia-main\python\miura_contrib
$env:PYTHONPATH = "..;."
python Miura_Folding_single_defect.py
```

## Aplicar parches al núcleo, si se desea

Los parches son opcionales. Se incluyen porque algunos experimentos se benefician de las correcciones en `PathAnalysis.py` y `VisualFold.py`.

Desde la raíz del repositorio PyPiroflexia:

```bash
patch -p1 < /ruta/a/PyPiroflexia_miura_module/patches/core_changes.patch
```

Alternativamente, se pueden revisar los archivos de `modified_core/` y copiar manualmente solo los cambios necesarios.

## Dependencias

El módulo usa las dependencias del repositorio original y, adicionalmente, las bibliotecas habituales del flujo de análisis:

```text
numpy
scipy
matplotlib
pyvista      # solo si se usa VisualFold/GIF
```

## Notas de limpieza

No se incluyeron archivos generados de salida, como `.png`, `.gif`, `__pycache__`, `.history`, `variable_history` ni `files.zip`. El objetivo de esta carpeta es entregar únicamente código fuente y trazabilidad de cambios.
