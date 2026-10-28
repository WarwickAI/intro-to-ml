"""Build ml_lab.ipynb and ml_lab_solutions.ipynb from the workshop page on the website.

Usage: python3 workshop3/build_notebooks.py path/to/website/src/content/courses/ml-ai-flagship/workshop-3/index.mdx

The page is the source of truth. Code blocks become code cells, with the block after
each one's <details> answer used for the solutions. Text questions get a "Your answer"
cell. <NeuronDiagram> tags reuse the existing hidden-code diagram cells, and an MDX
comment of the form {/* notebook: ... */} becomes a notebook-only markdown cell.
"""

import html
import json
import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
DIAGRAMS = {"one-neuron": "One neuron", "neurons-add-up": "Neurons add up", "layers-and-parameters": "Layers and parameters", "loss-bowl": "Loss bowl"}
COLAB_BULLET = re.compile(r"^- Open the \[notebook in Colab\]\(.*?\) and save your own copy\.$", re.M)
NB_BULLET = "- Save your own copy in Colab (**File → Save a copy in Drive**)."


def segments(body):
    out, buf, lines, i = [], [], body.split("\n"), 0

    def flush():
        text = "\n".join(buf).strip("\n")
        buf.clear()
        if text and not text.startswith("There are no slides"):
            out.append(("text", text))

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append(("code", "\n".join(lines[i + 1:j])))
            i = j + 1
        elif line.strip() == "<details>":
            flush()
            j = i
            while lines[j].strip() != "</details>":
                j += 1
            inner = "\n".join(l for l in lines[i + 1:j] if not l.strip().startswith("<summary>")).strip("\n")
            if inner.startswith("```"):
                out.append(("answer_code", "\n".join(inner.split("\n")[1:-1])))
            else:
                out.append(("answer_text", inner))
            i = j + 1
        elif line.startswith("<NeuronDiagram"):
            flush()
            out.append(("diagram", re.search(r'diagram="([^"]+)"', line).group(1)))
            i += 1
        elif line.startswith("{/* notebook:"):
            flush()
            out.append(("note", re.fullmatch(r"\{/\* notebook: (.*) \*/\}", line).group(1)))
            i += 1
        elif line.startswith("import "):
            i += 1
        else:
            if line.startswith("## ") or line.startswith("### "):
                flush()
            buf.append(line)
            i += 1
    flush()
    return out


def lines_of(text):
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + [parts[-1]]


def md_cell(text):
    text = text.replace("](/education/", "](https://warwick.ai/education/")
    text = COLAB_BULLET.sub(NB_BULLET, text)
    return {"cell_type": "markdown", "metadata": {}, "source": lines_of(text)}


def code_cell(source, form=False):
    return {"cell_type": "code", "metadata": {"cellView": "form"} if form else {}, "source": lines_of(source), "execution_count": None, "outputs": []}


def hint(student, answer):
    diffs = [re.sub(r"\s+#.*$", "", a).strip() for s, a in zip(student.split("\n"), answer.split("\n")) if s != a]
    return "\n\n<details><summary>Hint</summary><code>" + "<br>".join(html.escape(d) for d in diffs) + "</code></details>"


def header(body, solutions):
    title = "# Workshop 3: Solutions" if solutions else "# Workshop 3: Neural networks in PyTorch"
    rows, t = [], 0
    for m in re.finditer(r"^## (.+?) \((\d+) minutes?\)$", body, re.M):
        rows.append(f"| {t}–{t + int(m.group(2))} | {m.group(1)} |")
        t += int(m.group(2))
    return [md_cell(title), md_cell("| Minutes | Activity |\n|---|---|\n" + "\n".join(rows))]


def cells_from(segs, solutions, forms):
    cells, k = [], 0
    while k < len(segs):
        kind, val = segs[k]
        nxt = segs[k + 1] if k + 1 < len(segs) else (None, None)
        after = segs[k + 2] if k + 2 < len(segs) else (None, None)
        if kind == "text" and nxt[0] == "answer_text":
            *before, question = val.split("\n\n")
            if before:
                cells.append(md_cell("\n\n".join(before)))
            cells += [md_cell(question), md_cell(nxt[1])] if solutions else [md_cell(question + "\n\nYour answer: …")]
            k += 2
        elif kind == "text" and nxt[0] == "code" and after[0] == "answer_code" and not solutions:
            cells.append(md_cell(val + hint(nxt[1], after[1])))
            k += 1
        elif kind in ("text", "note"):
            cells.append(md_cell(val))
            k += 1
        elif kind == "code" and nxt[0] == "answer_code":
            cells.append(code_cell(nxt[1] if solutions else val))
            k += 2
        elif kind == "code":
            cells.append(code_cell(val))
            k += 1
        elif kind == "diagram":
            cells.append(code_cell(forms[DIAGRAMS[val]], form=True))
            k += 1
        else:
            raise ValueError(f"stray {kind} at {k}")
    return cells


def write_notebook(path, body, solutions, forms):
    nb = json.loads(path.read_text())
    nb["cells"] = header(body, solutions) + cells_from(segments(body), solutions, forms)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def form_cells(path):
    nb = json.loads(path.read_text())
    forms = {}
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if cell["metadata"].get("cellView") == "form":
            forms[source.splitlines()[0].removeprefix("# @title ").strip()] = source
    missing = set(DIAGRAMS.values()) - set(forms)
    if missing:
        raise SystemExit(f"form cells missing from {path.name}: {missing}")
    return forms


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    body = Path(sys.argv[1]).read_text().split("---\n", 2)[2]
    forms = form_cells(LAB / "ml_lab.ipynb")
    write_notebook(LAB / "ml_lab.ipynb", body, False, forms)
    write_notebook(LAB / "ml_lab_solutions.ipynb", body, True, forms)
    print("built")
