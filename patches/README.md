# patches

Esta carpeta contiene los parches que documentan las diferencias entre el repositorio original de PyPiroflexia y la versión modificada durante el desarrollo del aporte.

Los parches permiten revisar de forma precisa qué líneas cambiaron en los archivos del núcleo del solver.

## Archivos incluidos

- `Miura_Folding.patch`  
  Cambios asociados al driver base de Miura-Ori.

- `PathAnalysis.patch`  
  Cambios asociados al análisis incremental y a la compatibilidad con matrices dispersas.

- `VisualFold.patch`  
  Cambios asociados al flujo de visualización y generación de animaciones.

- `core_changes.patch`  
  Parche agregado que reúne los cambios principales del núcleo modificado.

## Uso

Desde la raíz del repositorio PyPiroflexia, un parche puede aplicarse con:

```bash
patch -p1 < patches/core_changes.patch
