#!/usr/bin/env python3
"""Build the self-contained demo/index.html: inlines the frozen net's
weights (as JSON) and demo/forward_pass.js's source directly into
demo/template.html. The result needs no server, no build step, and no
network access - it opens correctly via a plain file:// URL. Also writes
demo/weights.json, a standalone copy used by the Node-based JS/numpy
parity test (tests/test_js_parity.py) so it does not need to parse HTML.

Reads no label data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from poker_handnet.dag import CompiledNet  # noqa: E402
from poker_handnet.export_js import net_to_json_obj  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "weights" / "handnet_v1.npz"
DEMO_DIR = REPO_ROOT / "demo"


def main() -> None:
    net = CompiledNet.from_npz(str(WEIGHTS_PATH))
    net_json_obj = net_to_json_obj(net)
    weights_json = json.dumps(net_json_obj)

    (DEMO_DIR / "weights.json").write_text(weights_json + "\n")

    forward_js = (DEMO_DIR / "forward_pass.js").read_text()
    template = (DEMO_DIR / "template.html").read_text()

    html = template.replace(
        "/*__WEIGHTS_JSON_PLACEHOLDER__*/null", weights_json
    ).replace("/*__FORWARD_PASS_JS_PLACEHOLDER__*/", forward_js)

    if "__PLACEHOLDER__" in html or "/*__" in html:
        raise AssertionError("a template placeholder was not substituted")

    out_path = DEMO_DIR / "index.html"
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(html) / 1024:.1f} KiB)")
    print(f"wrote {DEMO_DIR / 'weights.json'}")


if __name__ == "__main__":
    main()
