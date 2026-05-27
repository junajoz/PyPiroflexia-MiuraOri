"""
Copia el módulo miura_contrib dentro del directorio python/ de un repositorio
PyPiroflexia local.

Uso:
    python tools/install_into_repo.py /ruta/a/PyPiroflexia-main/python
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Uso: python tools/install_into_repo.py /ruta/a/PyPiroflexia-main/python")
        return 1

    target_python_dir = Path(sys.argv[1]).expanduser().resolve()
    if not target_python_dir.exists() or not target_python_dir.is_dir():
        print(f"Error: no existe el directorio destino: {target_python_dir}")
        return 1

    module_root = Path(__file__).resolve().parents[1]
    source = module_root / "miura_contrib"
    target = target_python_dir / "miura_contrib"

    if not source.exists():
        print(f"Error: no se encontró el módulo fuente: {source}")
        return 1

    if target.exists():
        shutil.rmtree(target)

    shutil.copytree(source, target)
    print(f"Módulo copiado en: {target}")
    print("Para ejecutar drivers:")
    print(f"  cd {target}")
    print('  PYTHONPATH="..:." python Miura_Folding_single_defect.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
