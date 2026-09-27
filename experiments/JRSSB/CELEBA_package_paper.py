#!/usr/bin/env python3
"""Package the current Agent 3 report and original-resolution figures for paper use."""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(
    "/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid/paper/agent3_paper_2026-09-13.zip"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=HERE / "agent3.md")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not str(args.output.resolve()).startswith("/cwork/"):
        raise ValueError("Paper outputs must be saved under /cwork")
    original = args.report.read_bytes()
    report = original.decode("utf-8")
    contents = {"source_agent3.md": original}
    metadata = []
    figures = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report)
    if not figures:
        raise RuntimeError("No figures found in report")
    for target in dict.fromkeys(re.findall(r"\]\(([^)]+)\)", report)):
        path = (args.report.parent / target).resolve()
        if target in figures:
            destination = "figures/" + path.name
        elif path.suffix in (".csv", ".json"):
            destination = "tables/" + path.name
        else:
            # Historical report is a reference outside the current paper package.
            report = re.sub(
                r"\[([^\]]+)\]\(" + re.escape(target) + r"\)",
                r"\1 (not included in this paper bundle)",
                report,
            )
            continue
        data = path.read_bytes()
        if destination in contents and contents[destination] != data:
            raise RuntimeError(f"Conflicting package paths: {destination}")
        contents[destination] = data
        report = report.replace("](" + target + ")", "](" + destination + ")")
        record = dict(
            path=destination,
            source_path=str(path),
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
        )
        if target in figures:
            with Image.open(io.BytesIO(data)) as image:
                record.update(
                    width=image.width, height=image.height, dpi=image.info.get("dpi")
                )
        metadata.append(record)
    contents["agent3.md"] = report.encode("utf-8")
    contents["assets.json"] = (json.dumps(metadata, indent=2) + "\n").encode()
    contents["README.txt"] = (
        "Open agent3.md for the portable paper report. figures/ contains every embedded figure at original resolution, without resampling.\n"
        "source_agent3.md is an exact snapshot of the repository report before package-relative link rewriting.\n"
        "tables/ contains directly linked CSV/JSON artifacts; result.json records external reproduction inputs and is not a promise that those inputs are included.\n"
        "The historical report is not part of the current paper package. assets.json records figure dimensions and source hashes.\n"
        "This is a paper-production bundle, not a dataset or training-checkpoint archive.\n"
    ).encode()
    contents["SHA256SUMS.txt"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(contents.items())
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        for target in re.findall(r"\]\(([^)]+)\)", report):
            if target not in archive.namelist():
                raise RuntimeError(f"Broken portable report link: {target}")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".zip.sha256").write_text(f"{digest}  {args.output.name}\n")
    print(f"{args.output}\nFigures: {len(figures)}\nSHA256: {digest}")


if __name__ == "__main__":
    main()
