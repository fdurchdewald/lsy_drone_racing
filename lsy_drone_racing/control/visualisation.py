
import acados_template, importlib.metadata, subprocess, pathlib, re, json, sys, os

print("\n--- Python package version ---")
try:
    # wenn über pip/mamba installiert
    print("acados_template =", importlib.metadata.version("acados_template"))
except importlib.metadata.PackageNotFoundError:
    print("acados_template: no wheel, probably editable install")

print("\n--- Shared library commit ---")
# acados_template lädt beim Import die Datei <site-packages>/acados_template/version.json
ver_file = pathlib.Path(acados_template.__file__).with_name("version.json")
if ver_file.exists():
    data = json.loads(ver_file.read_text())
    print("commit:", data.get("commit", "<unknown>"))
else:
    print("version.json not found (editable build?)")
PY
