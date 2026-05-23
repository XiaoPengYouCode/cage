from __future__ import annotations

from pathlib import Path


def render_abaqus_odb_export_script(
    *,
    odb_filename: str,
    output_filename: str,
    field_name: str = "U",
    step_name: str = "Load",
) -> str:
    return f"""from odbAccess import openOdb

path = {odb_filename!r}
req_data = {field_name!r}
step = {step_name!r}
data_file = open({output_filename!r}, "w")
myodb = openOdb(path=path)
val = myodb.steps[step].frames[-1].fieldOutputs[req_data].values
for i in range(0, len(val)):
    seq = val[i].nodeLabel
    u1 = val[i].data[0]
    u2 = val[i].data[1]
    u3 = val[i].data[2]
    data_file.write("%10.6E\\t" % seq)
    data_file.write("%10.6E\\t" % u1)
    data_file.write("%10.6E\\t" % u2)
    data_file.write("%10.6E\\n" % u3)
data_file.close()
myodb.close()
"""


def write_abaqus_odb_export_script(
    destination: Path,
    *,
    odb_filename: str,
    output_filename: str,
    field_name: str = "U",
    step_name: str = "Load",
) -> Path:
    destination = Path(destination)
    destination.write_text(
        render_abaqus_odb_export_script(
            odb_filename=odb_filename,
            output_filename=output_filename,
            field_name=field_name,
            step_name=step_name,
        ),
        encoding="utf-8",
    )
    return destination


__all__ = [
    "render_abaqus_odb_export_script",
    "write_abaqus_odb_export_script",
]
