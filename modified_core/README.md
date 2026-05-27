# modified_core

Esta carpeta contiene copias completas de archivos del núcleo original de PyPiroflexia que fueron modificados durante el desarrollo del aporte.

Se incluyen aquí para documentar con claridad qué archivos del solver base fueron intervenidos, sin mezclar esas modificaciones con los scripts nuevos de `miura_contrib/`.

## Archivos incluidos

### `Miura_Folding.py`

Archivo base de simulación Miura-Ori. La modificación principal consiste en agregar una perturbación aleatoria sobre la geometría antes de ejecutar el flujo de análisis. Esta intervención sirvió como punto de partida para estudiar la sensibilidad de la estructura frente a imperfecciones geométricas.

### `PathAnalysis.py`

Archivo encargado del análisis incremental de trayectoria. La modificación principal reemplaza el uso de `.A` por `.toarray()` al trabajar con matrices dispersas. Esto mejora la compatibilidad con versiones recientes de `scipy` y evita errores al resolver sistemas lineales.

### `VisualFold.py`

Archivo de visualización de la deformación. Las modificaciones se orientan a facilitar el guardado de animaciones, activar renderizado fuera de pantalla y evitar la apertura de ventanas interactivas durante la generación de GIFs.

## Uso recomendado

Estos archivos no deben copiarse automáticamente sobre el solver original sin revisión previa. Su función principal es servir como respaldo y trazabilidad de los cambios realizados.

Si se desea incorporar estas modificaciones al repositorio principal, se recomienda revisar primero los parches disponibles en la carpeta `patches/`.

## Relación con `patches/`

Mientras `modified_core/` contiene la versión completa de cada archivo modificado, `patches/` contiene únicamente las diferencias respecto al archivo original. Por tanto:

- usar `modified_core/` si se quiere revisar el archivo completo;
- usar `patches/` si se quiere aplicar o inspeccionar solamente los cambios.
