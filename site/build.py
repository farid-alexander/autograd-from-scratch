"""Build the GitHub Pages version of the article from README.md.

    python site/build.py        # writes _site/index.html and copies images/

README.md stays the single source. This script
- swaps the README's Amazon Attribution tags for the ones that count clicks
  from the Pages site, so the two places are measured separately,
- points relative links (the .py files, LICENSE) at the files on GitHub,
- keeps the math away from Markdown and leaves it for KaTeX in the browser.
"""
import html
import pathlib
import re
import shutil

import markdown
from pygments.formatters import HtmlFormatter

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"
REPO = "https://github.com/farid-alexander/autograd-from-scratch"
PAGE_URL = "https://farid-alexander.github.io/autograd-from-scratch/"
DESCRIPTION = ("Build a working autograd engine in about 60 lines of plain "
               "Python and check it against PyTorch: how the forward pass "
               "records a graph and how backward() sends gradients through it.")

# Amazon Attribution ad groups: README link -> Pages link, per edition.
TAGS = {
    "maas_adg_FF2CFA1CAA9C9A480A5C43C36AC2F16E":   # Kindle
        "maas_adg_DF6D3FACBAE3B2418250A1CEC6B73802",
    "maas_adg_F18C9CB4D7B524FC5D6A506C5C99FB91":   # paperback
        "maas_adg_DFD31D0BA3B70A4C96A95276A3F4356F",
}

FENCE = re.compile(r"^```.*?^```[ \t]*$", re.M | re.S)
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)([^\n$]+?)\$")
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|images/)([^)\s]+)\)")


def protect_math(text):
    """Replace math outside code fences with placeholders Markdown ignores."""
    saved = []

    def stash(kind):
        def repl(m):
            saved.append((kind, m.group(1)))
            return f"MATHPLACEHOLDER{len(saved) - 1}X"
        return repl

    pieces, last = [], 0
    for m in FENCE.finditer(text):
        prose = text[last:m.start()]
        prose = DISPLAY_MATH.sub(stash("display"), prose)
        prose = INLINE_MATH.sub(stash("inline"), prose)
        pieces += [prose, m.group(0)]
        last = m.end()
    prose = DISPLAY_MATH.sub(stash("display"), text[last:])
    pieces.append(INLINE_MATH.sub(stash("inline"), prose))
    return "".join(pieces), saved


def restore_math(body, saved):
    def repl(m):
        kind, tex = saved[int(m.group(1))]
        tex = html.escape(tex.strip())
        if kind == "display":
            return f'<div class="math-display">$${tex}$$</div>'
        return f"${tex}$"
    body = re.sub(r"<p>MATHPLACEHOLDER(\d+)X</p>", repl, body)
    return re.sub(r"MATHPLACEHOLDER(\d+)X", repl, body)


def pygments_css():
    light = HtmlFormatter(style="default").get_style_defs(".highlight")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".highlight")
    return (light + "\n@media (prefers-color-scheme: dark) {\n"
            + dark + "\n}\n")


def main():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for readme_tag, pages_tag in TAGS.items():
        assert readme_tag in text, f"README lacks the tag {readme_tag}"
        text = text.replace(readme_tag, pages_tag)
    text = RELATIVE_LINK.sub(lambda m: f"]({REPO}/blob/main/{m.group(1)})",
                             text)

    title = re.search(r"^# (.+)$", text, re.M).group(1)
    text, saved = protect_math(text)
    body = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "codehilite", "toc"],
        extension_configs={"codehilite": {"guess_lang": False,
                                          "css_class": "highlight"}},
    )
    body = restore_math(body, saved)
    # Open Amazon and GitHub links in the same tab, but mark outside links.
    body = body.replace('<a href="https://', '<a rel="noopener" href="https://')

    template = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    page = (template
            .replace("{{title}}", html.escape(title))
            .replace("{{description}}", html.escape(DESCRIPTION))
            .replace("{{page_url}}", PAGE_URL)
            .replace("{{repo}}", REPO)
            .replace("{{pygments_css}}", pygments_css())
            .replace("{{body}}", body))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    (OUT / "index.html").write_text(page, encoding="utf-8")
    shutil.copytree(ROOT / "images", OUT / "images")
    (OUT / ".nojekyll").write_text("")
    print(f"wrote {OUT / 'index.html'} ({len(page):,} bytes, "
          f"{len(saved)} math expressions)")


if __name__ == "__main__":
    main()
