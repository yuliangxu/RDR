"""Render new generated-only JRSSB ranked portrait pages from the frozen selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pymupdf as pdf
from matplotlib import get_data_path

REFERENCE = Path(__file__).resolve().parents[2] / "results/CelebA/publication_figures_20261003_v2"
TARGETS = ("Smallest RDR", "Closest to 1", "Largest RDR")
PANELS = {("feature", "lower"): "7A", ("feature", "upper"): "7B",
          ("pixel", "lower"): "S5A", ("pixel", "upper"): "S5B"}
TRUNCATION = {"lower": "Qₗ", "upper": "Qᵤ"}
WIDTH, HEIGHT = 649, 259  # 10 x 4 native 64-pixel portraits and one-pixel gutters


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_generated(reference, filename):
    with (reference / filename).open(newline="") as stream:
        return [r for r in csv.DictReader(stream) if r["panel"] == "ranked" and r["repeat"] == "0"
                and r["branch"] in TRUNCATION and r["source"] == r["branch"]]


def group_key(row):
    return row["representation"], row["branch"], int(row["target"])


def selected_groups(reference):
    selected = read_generated(reference, "selected_images.csv")
    summaries = read_generated(reference, "image_ranges.csv")
    check(len(selected) == 480 and len(summaries) == 12, "Unexpected selection or range count")
    ranges = {group_key(row): row for row in summaries}
    groups = {}
    for level, branch in PANELS:
        for target, title in enumerate(TARGETS):
            key = level, branch, target
            rows = sorted((r for r in selected if group_key(r) == key), key=lambda r: int(r["display_rank"]))
            check([int(r["display_rank"]) for r in rows] == list(range(40)), f"Invalid saved order: {key}")
            check(len({r["source_id"] for r in rows}) == len({r["test_row"] for r in rows}) == 40,
                  f"Duplicate portrait within group: {key}")
            scores = [float(r["rdr"]) for r in rows]
            summary = ranges[key]
            check(summary["group"] == title and int(summary["shown_count"]) == 40, f"Invalid group: {key}")
            check(all(all(r[k] == v for k, v in summary.items()) for r in rows), f"Row metadata differs: {key}")
            check((min(scores), max(scores)) == (float(summary["score_min"]), float(summary["score_max"])),
                  f"Range differs from selected scores: {key}")
            groups[key] = rows
    return groups, ranges


def image_array(doc, xref):
    pix = pdf.Pixmap(doc, xref)
    check(pix.n == 3 and not pix.alpha, "Expected an opaque RGB montage")
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def generated_reference_montages(filename):
    """Read only the bottom image objects. MuPDF coordinates increase downward."""
    with pdf.open(filename) as doc:
        check(len(doc) == 1, f"Expected one reference page: {filename}")
        page = doc[0]
        objects = sorted(page.get_image_info(xrefs=True), key=lambda obj: obj["bbox"][1])
        check(len(objects) == 6, "Expected two rows of three montages in reference PDF")
        top = sorted(objects[:3], key=lambda obj: obj["bbox"][0])
        bottom = sorted(objects[3:], key=lambda obj: obj["bbox"][0])
        check(len({obj["xref"] for obj in objects}) == 6, "Unexpected reference image reuse")
        check(max(obj["bbox"][3] for obj in top) < min(obj["bbox"][1] for obj in bottom), "Overlapping rows")
        montages = []
        for i, obj in enumerate(bottom):
            x0, y0, x1, y1 = obj["bbox"]
            check(abs(top[i]["bbox"][0] - x0) < .01 and abs(bottom[0]["bbox"][1] - y0) < .01
                  and (i == 0 or bottom[i-1]["bbox"][2] < x0), "Invalid generated-row layout")
            a, b, c, d, _, _ = obj["transform"]
            check(a > 0 and d > 0 and b == c == 0, "Unexpected montage rotation or reflection")
            pixels = image_array(doc, obj["xref"])
            h, w = pixels.shape[:2]
            check((w, h) == (826, 330), "Unexpected reference image dimensions")
            # The original PDF enlarged the native montage with nearest-neighbor interpolation.
            x = ((np.arange(WIDTH) + .5) * w / WIDTH).astype(int)
            y = ((np.arange(HEIGHT) + .5) * h / HEIGHT).astype(int)
            native = pixels[y[:, None], x]
            check(np.all(native[64::65] == 255) and np.all(native[:, 64::65] == 255), "Misaligned gutters")
            montages.append((native, obj["xref"], list(obj["bbox"])))
        return montages


def centered(page, text, center, baseline, font, resource, size, max_width):
    width = font.text_length(text, fontsize=size)
    check(width <= max_width, f"Text exceeds panel width: {text}")
    page.insert_text((center - width / 2, baseline), text, fontname=resource, fontsize=size,
                     color=(.09, .09, .09))


def render_page(filename, title, montages, labels):
    # Dimensions and font sizes are points at seven-inch manuscript width.
    page_width, margin, gap = 504, 5, 7
    col_width = (page_width - 2*margin - 2*gap) / 3
    image_top = 43
    image_bottom = image_top + col_width * HEIGHT / WIDTH
    normal, bold = pdf.Font("helv"), pdf.Font("hebo")
    title_font = pdf.Font(fontfile=str(Path(get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"))
    with pdf.open() as doc:
        page = doc.new_page(width=page_width, height=image_bottom + 4)
        page.insert_font(fontname="Normal", fontbuffer=normal.buffer)
        page.insert_font(fontname="Bold", fontbuffer=bold.buffer)
        page.insert_font(fontname="TitleFont", fontbuffer=title_font.buffer)
        centered(page, title, page_width/2, 12, title_font, "TitleFont", 11, page_width - 2*margin)
        for i, (pixels, label) in enumerate(zip(montages, labels)):
            left = margin + i * (col_width + gap)
            middle = left + col_width/2
            centered(page, TARGETS[i], middle, 27, bold, "Bold", 9.5, col_width)
            centered(page, label, middle, 38, normal, "Normal", 8, col_width)
            pix = pdf.Pixmap(pdf.csRGB, WIDTH, HEIGHT, pixels.tobytes(), False)
            page.insert_image(pdf.Rect(left, image_top, left + col_width, image_bottom), pixmap=pix)
        doc.set_metadata({"title": title})
        doc.save(filename, garbage=4, deflate=True, no_new_id=True)


def verify_and_preview(filename, title, expected, labels):
    with pdf.open(filename) as doc:
        check(len(doc) == 1 and not doc.embfile_count(), "Unexpected pages or attachments")
        page = doc[0]
        objects = sorted(page.get_image_info(xrefs=True), key=lambda obj: obj["bbox"][0])
        all_images = {i for i in range(1, doc.xref_length()) if doc.xref_get_key(i, "Subtype") == ("name", "/Image")}
        check(len(objects) == len(all_images) == 3 and {o["xref"] for o in objects} == all_images,
              "Expected exactly three displayed images and no other embedded images")
        check(not page.get_xobjects() and not list(page.annots() or []), "Unexpected form or annotation")
        text = page.get_text()
        check(title in text and all(t in text for t in TARGETS), "Missing panel or group title")
        for i, (obj, pixels, label) in enumerate(zip(objects, expected, labels)):
            rect = pdf.Rect(obj["bbox"])
            check(all(abs(obj["bbox"][1] - o["bbox"][1]) < .01 for o in objects)
                  and (i == 0 or objects[i-1]["bbox"][2] < rect.x0), "Unexpected montage placement")
            header = page.get_textbox(pdf.Rect(rect.x0, 15, rect.x1, rect.y0))
            check(TARGETS[i] in header and label in header, "Wrong title or range above montage")
            check(np.array_equal(image_array(doc, obj["xref"]), pixels), "Delivered montage pixels differ")
        page.get_pixmap(dpi=300, alpha=False).save(filename.with_suffix(".png"))
        return [obj["xref"] for obj in objects]


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(reference, output, archive=None):
    check(not output.exists(), f"Choose a new output directory: {output}")
    seal = (reference / "COMPLETE.json").read_bytes()
    check(sha(seal) == (reference / "COMPLETE.sha256").read_text().split()[0], "Invalid reference seal")
    frozen = json.loads(seal)["files"]
    stems = {(level, branch): f"images_{level}_{branch}_ranked" for level, branch in PANELS}
    for name in ["selected_images.csv", "image_ranges.csv", "protocol.json", *(stem+".pdf" for stem in stems.values())]:
        check(sha((reference / name).read_bytes()) == frozen[name], f"Reference file changed: {name}")
    groups, ranges = selected_groups(reference)
    archive_pixels = {}
    if archive is not None:
        from experiments.CelebA.synthetic_ranked_pixels import archive_montages
        archive_pixels = archive_montages(archive, reference, groups)
        check(archive_pixels.keys() == groups.keys(), "Incomplete generated archive")
    audit = {"reference_seal_sha256": sha(seal), "groups": [],
             "count": sum(map(len, groups.values())), "unique_source_ids": len({r["source_id"] for rows in groups.values() for r in rows})}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "figures"
        stage.mkdir()
        for (level, branch), stem in stems.items():
            native_objects = generated_reference_montages(reference / (stem + ".pdf"))
            montages, labels = [], []
            for target, (native, _, _) in enumerate(native_objects):
                key = level, branch, target
                pixels = archive_pixels.get(key, native)
                check(np.array_equal(pixels, native), f"Native archive differs from reference generated montage: {key}")
                montages.append(pixels)
                row = ranges[key]
                labels.append(f"n = 40    RDR [{float(row['score_min']):.3g}, {float(row['score_max']):.3g}]")
            rep = {"feature": "Inception pool3 feature RDR", "pixel": "RGB pixel RDR"}[level]
            title = f"{rep} · {branch.capitalize()} truncation ({TRUNCATION[branch]})"
            filename = stage / (stem + ".pdf")
            render_page(filename, title, montages, labels)
            xrefs = verify_and_preview(filename, title, montages, labels)
            for i, (pixels, source_xref, source_bbox) in enumerate(native_objects):
                audit["groups"].append({"panel": PANELS[level, branch], "group": TARGETS[i], "label": labels[i],
                                        "reference_xref": source_xref, "reference_bbox_points_top_left": source_bbox,
                                        "output_xref": xrefs[i], "native_rgb_sha256": sha(pixels.tobytes())})
        write_rows(stage / "selected_images.csv", [r for rows in groups.values() for r in rows])
        write_rows(stage / "image_ranges.csv", [ranges[key] for key in groups])
        audit["files_sha256"] = {p.name: sha(p.read_bytes()) for p in sorted(stage.iterdir())}
        (stage/"verification.json").write_text(json.dumps(audit, indent=2) + "\n")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Verify and use original generated pixel arrays and frozen predictions")
    args = parser.parse_args()
    render(args.reference, args.output, args.archive)
