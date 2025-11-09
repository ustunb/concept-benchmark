#!/usr/bin/env python
import argparse
import json
import os
import base64
from pathlib import Path


PROMPT = """You are helping to brainstorm visual concepts that distinguish two SYNTHETIC robot classes. Do not use any real-world knowledge of the names; rely ONLY on the example images listed below.
Return exactly {n} JSONL lines, one object per line: {{"text":"<short visual concept>"}}.
No brackets, no commas after objects, no numbering, no commentary, no code fences.

Requirements:
- Each concept is a short visual phrase (<= 30 chars)
- Avoid class names or near-synonyms
- Avoid near-duplicates
- Prefer parts, shapes, textures, colors, materials, layouts, counts
- Focus on features that are discriminative between the classes, based on the examples

Classes:
{classes}

Example images:
{examples_block}
"""


settings = {
    "classes": ["Glorp", "Drent"],
    "out_dir": "results/robots/llm_glorp_drent_1002",
    "format": "jsonl",
    "image_root": "data/robot_images",
    "ext": ".png",
    "llm": True,
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "api_key_env": "GEMINI_API_KEY",
    "out_file": "candidates.llm.jsonl",
    "from_catalog": "results/robots/cbm_run_1002_subconcepts__draw_only/catalog.csv",
    "per_class": 20,
    "seed": 1002,
    "label-map": "1=Glorp,0=Drent",
    "debug": True,
    "force_all_subtypes": True,
    "n_concepts": 12,
}


def build_prompt(classes, examples_by_class, n):
    items = "\n- " + "\n- ".join(classes)
    lines = []
    for c in classes:
        paths = examples_by_class.get(c, [])
        if paths:
            lines.append(f"{c}:")
            lines.extend([f"- {p}" for p in paths])
    examples_block = "\n".join(lines) if lines else "(no examples provided)"
    return PROMPT.format(classes=items, examples_block=examples_block, n=n)

def write_stub(outd: Path, fmt: str):
    stub = [
        {"text": "round head"},
        {"text": "square head"},
        {"text": "two legs"},
        {"text": "wheeled base"},
        {"text": "antenna"},
        {"text": "metallic texture"},
        {"text": "blue color"},
        {"text": "yellow color"},
    ]
    if fmt == "json":
        payload = {"concepts": [x["text"] for x in stub]}
        (outd / "candidates.stub.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        with (outd / "candidates.stub.jsonl").open("w", encoding="utf-8") as f:
            for it in stub:
                f.write(json.dumps(it) + "\n")
    else:
        (outd / "candidates.stub.txt").write_text("\n".join([x["text"] for x in stub]) + "\n", encoding="utf-8")

def _extract_jsonl(text: str):
    items = []
    try:
        blob = json.loads(text)
        if isinstance(blob, list):
            for it in blob:
                if isinstance(it, str) and it.strip():
                    items.append({"text": it.strip()})
                elif isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"].strip():
                    items.append({"text": it["text"].strip()})
        elif isinstance(blob, dict):
            for key in ("concepts", "items", "data"):
                seq = blob.get(key)
                if isinstance(seq, list):
                    for it in seq:
                        if isinstance(it, str) and it.strip():
                            items.append({"text": it.strip()})
                        elif isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"].strip():
                            items.append({"text": it["text"].strip()})
    except Exception:
        pass
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        line = line.strip("`").strip().strip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                t = obj["text"].strip()
                if t:
                    items.append({"text": t})
        except Exception:
            continue
    seen = set()
    uniq = []
    for it in items:
        t = it["text"].strip()
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append({"text": t})
    return uniq

class _LLMBase:
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompt: str, image_paths):
        raise NotImplementedError


class _LLMRegistry:
    _providers = {}

    @classmethod
    def register(cls, name):
        def deco(kls):
            cls._providers[name.lower()] = kls
            return kls
        return deco

    @classmethod
    def create(cls, name, model_name, api_key):
        kls = cls._providers.get(name.lower())
        if not kls:
            raise SystemExit(f"unsupported provider: {name}")
        return kls(model_name, api_key)


def _encode_images_b64(image_paths):
    out = []
    for p in image_paths:
        try:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            ext = Path(p).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif ext == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
            out.append((b64, mime))
        except Exception:
            continue
    return out


@_LLMRegistry.register("gemini")
class _GeminiClient(_LLMBase):
    def generate(self, prompt: str, image_paths):
        try:
            import google.generativeai as genai
            from PIL import Image
        except Exception as e:
            raise SystemExit(
                "google-generativeai or pillow not installed.\n"
                "pip install --upgrade \"google-ai-generativelanguage>=0.6.6\" \"google-generativeai>=0.8.5\" "
                "\"protobuf>=5.29.3,<6\" \"grpcio-status>=1.70.0\" pillow"
            )
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        imgs = []
        for p in image_paths:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                pass
        parts = [prompt] + imgs if imgs else [prompt]
        resp = model.generate_content(parts)
        return (getattr(resp, "text", None) or "").strip()


@_LLMRegistry.register("openai")
class _OpenAIClient(_LLMBase):
    def generate(self, prompt: str, image_paths):
        try:
            from openai import OpenAI
        except Exception:
            raise SystemExit("openai SDK not installed. pip install --upgrade openai pillow")
        client = OpenAI(api_key=self.api_key)
        content = [{"type": "text", "text": prompt}]
        for b64, mime in _encode_images_b64(image_paths):
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0
        )
        return (resp.choices[0].message.content or "").strip()


@_LLMRegistry.register("anthropic")
class _AnthropicClient(_LLMBase):
    def generate(self, prompt: str, image_paths):
        try:
            import anthropic
        except Exception:
            raise SystemExit("anthropic SDK not installed. pip install --upgrade anthropic pillow")
        client = anthropic.Anthropic(api_key=self.api_key)
        content = [{"type": "text", "text": prompt}]
        for b64, mime in _encode_images_b64(image_paths):
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        resp = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        text_parts = []
        for part in getattr(resp, "content", []) or []:
            if getattr(part, "type", None) == "text":
                text_parts.append(part.text)
        return (" ".join(text_parts)).strip()


def _make_client(provider: str, model_name: str, api_key: str) -> _LLMBase:
    return _LLMRegistry.create(provider, model_name, api_key)


def parse_cli(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=settings["classes"])
    p.add_argument("--out-dir", default=settings["out_dir"])
    p.add_argument("--format", choices=["json", "jsonl", "txt"], default=settings["format"])
    p.add_argument("--image-root", default=settings["image_root"])
    p.add_argument("--ext", default=settings["ext"])
    p.add_argument("--indices", action="append", default=[])
    p.add_argument("--llm", action="store_true")
    p.add_argument("--provider", default=settings["provider"])
    p.add_argument("--model", default=settings["model"])
    p.add_argument("--api-key-env", default="")
    p.add_argument("--out-file", default=settings["out_file"])
    p.add_argument("--from-catalog", default=settings.get("from_catalog", ""))
    p.add_argument("--per-class", type=int, default=settings.get("per_class", 0))
    p.add_argument("--seed", type=int, default=settings.get("seed", 0))
    p.add_argument("--label-field", default=settings.get("label_field", ""))
    p.add_argument("--path-field", default=settings.get("path_field", ""))
    p.add_argument("--label-map", default=(settings.get("label_map", settings.get("label-map", ""))))
    p.add_argument("--debug", action="store_true", default=settings.get("debug", False))
    p.add_argument("--force-all-subtypes", action="store_true",
                   default=settings.get("force_all_subtypes", False))
    p.add_argument("--subtypes-csv", default=settings.get("subtypes_csv", "catalog_with_spurious.csv"))
    p.add_argument("--n-concepts", type=int, default=settings.get("n_concepts", 12))
    args, _ = p.parse_known_args(argv)
    return vars(args)

def run(cfg):
    classes = cfg.get("classes") or []
    out_dir = cfg.get("out_dir") or settings["out_dir"]
    fmt = cfg.get("format") or settings["format"]
    image_root = Path(cfg.get("image_root") or settings["image_root"])
    ext = cfg.get("ext") or settings["ext"]
    index_specs = cfg.get("indices") or []
    use_llm = bool(cfg.get("llm") or settings["llm"])
    provider = (cfg.get("provider") or settings["provider"]).lower()
    model_name = str(cfg.get("model") or settings["model"])
    debug = bool(cfg.get("debug", True))
    api_key_env = (
        cfg.get("api_key_env")
        or {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
            provider, settings.get("api_key_env", "")
        )
    )
    out_file = cfg.get("out_file") or settings["out_file"]
    force_all_subtypes = bool(cfg.get("force_all_subtypes", False))
    subtypes_csv_name = str(cfg.get("subtypes_csv") or "catalog_with_spurious.csv")
    n_concepts = int(cfg.get("n_concepts") or settings.get("n_concepts", 12))

    if not classes:
        raise SystemExit("missing --classes")

    examples = {c: [] for c in classes}
    from_catalog = (cfg.get("from_catalog") or settings.get("from_catalog", "")).strip()
    per_class = int(cfg.get("per_class") or settings.get("per_class", 0) or 0)
    seed = int(cfg.get("seed") or settings.get("seed", 0) or 0)
    if from_catalog and (per_class > 0 or force_all_subtypes):
        import csv, random
        cat_path = Path(from_catalog)
        if not cat_path.exists():
            raise SystemExit(f"from_catalog not found: {from_catalog}")

        with cat_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or [])]
            rows = [row for row in reader]

        if not rows:
            raise SystemExit(f"empty catalog: {from_catalog}")

        label_candidates = [
            "robot_type", "label", "class", "target", "y", "model", "name",
            "cls", "class_id", "class_idx", "label_id", "label_idx",
            "category", "category_id", "group", "type", "gt_label", "ground_truth", "tag"
        ]
        path_candidates = [
            "path", "filepath", "filename", "file", "file_name", "file_path",
            "image", "image_path", "img", "img_path", "imgfile",
            "relative_path", "relpath", "relative_file", "relative_filepath",
            "uri", "url"
        ]

        def _pick_field(hdrs, override, cands):
            if override:
                for h in hdrs:
                    if h.lower() == override.lower():
                        return h
                for h in hdrs:
                    hl, ol = h.lower(), override.lower()
                    if ol in hl or hl in ol:
                        return h
            # prefer exact header matches before substring heuristics
            for cand in cands:
                for h in hdrs:
                    if h.lower() == cand.lower():
                        return h
            # then allow substring matches
            for cand in cands:
                cl = cand.lower()
                for h in hdrs:
                    hl = h.lower()
                    if cl in hl or hl in cl:
                        return h
            return None

        label_field = _pick_field(headers, (cfg.get("label_field") or settings.get("label_field", "")).strip(), label_candidates)
        path_field = _pick_field(headers, (cfg.get("path_field") or settings.get("path_field", "")).strip(), path_candidates)

        if not label_field or not path_field:
            raise SystemExit(f"missing required columns in {from_catalog}; headers={headers}")

        if debug:
            print("catalog headers:", headers)
            print("selected fields:", {"label_field": label_field, "path_field": path_field})

        lm_raw = str((cfg.get("label_map") or settings.get("label_map") or settings.get("label-map", "") or "")).strip()
        label_map = {}
        if lm_raw:
            if lm_raw.startswith("{"):
                try:
                    label_map = {str(k): str(v) for k, v in json.loads(lm_raw).items()}
                except Exception:
                    pass
            else:
                for pair in [t for t in lm_raw.split(",") if "=" in t]:
                    k, v = pair.split("=", 1)
                    label_map[str(k).strip()] = str(v).strip()

        rng = random.Random(seed)
        buckets = {c: [] for c in classes}

        # Build search bases once (robust to CWD/repo differences)
        bases = []
        try:
            ir = image_root if image_root.is_absolute() else (Path.cwd() / image_root)
            bases.append(ir.resolve())
        except Exception:
            bases.append(image_root)
        try:
            bases.append(cat_path.parent.resolve())
        except Exception:
            bases.append(cat_path.parent)
        try:
            bases.append(Path.cwd().resolve())
        except Exception:
            bases.append(Path.cwd())
        try:
            bases.append(Path(__file__).resolve().parent)
        except Exception:
            pass
        # de-dup while preserving order
        _seen = set()
        _dedup = []
        for b in bases:
            s = str(b)
            if s not in _seen:
                _seen.add(s)
                _dedup.append(b)
        bases = _dedup

        miss_prints = 0

        for r in rows:
            label_raw = str(r.get(label_field, "")).strip()
            lbl = None
            # label_map takes precedence (so you can flip 0/1 → Drent/Glorp as needed)
            if label_map:
                mapped = label_map.get(label_raw) or label_map.get(label_raw.lower())
                if mapped:
                    if mapped.isdigit():
                        idx = int(mapped)
                        if 0 <= idx < len(classes):
                            lbl = classes[idx]
                    else:
                        for c in classes:
                            if c.lower() == mapped.lower():
                                lbl = c
                                break
            if lbl is None and label_raw.isdigit():
                idx_lbl = int(label_raw)
                if 0 <= idx_lbl < len(classes):
                    lbl = classes[idx_lbl]
            if lbl is None:
                for c in classes:
                    if c.lower() == label_raw.lower():
                        lbl = c
                        break
            if lbl is None:
                continue

            q = str(r.get(path_field, "")).strip()
            if not q:
                continue
            if q.startswith("http://") or q.startswith("https://"):
                continue

            # Normalize slashes defensively; Path handles either, but
            # this guards against weird UNC-like strings getting treated as a single filename.
            q = q.strip('"').strip("'").replace("\\", "/")

            # Candidate paths to probe using precomputed bases
            candidates = []
            q_path = Path(q)
            if q_path.is_absolute():
                candidates.append(q_path)
            else:
                for base in bases:
                    candidates.append(base / q)
                    if not image_root.is_absolute():
                        candidates.append(base / image_root / q)

            found = None
            try_exts = []
            if ext:
                try_exts.append(ext if str(ext).startswith(".") else f".{ext}")
            try_exts += [".png", ".jpg", ".jpeg", ".webp"]

            # Probe direct paths and common suffix variants
            for base in candidates:
                if base.exists():
                    found = base
                    break
                stem = base.with_suffix("")  # drop any suffix
                for e in try_exts:
                    p = stem.with_suffix(e)
                    if p.exists():
                        found = p
                        break
                if found:
                    break

            # As a last resort, if q looks like just a filename, do a shallow rglob under bases
            if not found and ("/" not in q and "\\" not in q):
                name = Path(q).name
                for base in bases:
                    # try a shallow search (one level) first
                    direct = base / name
                    if direct.exists():
                        found = direct
                        break
                    for e in try_exts:
                        p = (base / name).with_suffix(e)
                        if p.exists():
                            found = p
                            break
                    if found:
                        break
                    # one-level down
                    for sub in (base.iterdir() if base.exists() and base.is_dir() else []):
                        if not sub.is_dir():
                            continue
                        probe = sub / name
                        if probe.exists():
                            found = probe
                            break
                        for e in try_exts:
                            p = probe.with_suffix(e)
                            if p.exists():
                                found = p
                                break
                        if found:
                            break
                    if found:
                        break

            if not found:
                if debug and miss_prints < 12:
                    print("MISS:", str(candidates[0]), "q=", q)
                    miss_prints += 1
                continue

            buckets[lbl].append(str(found))

        if debug:
            pre_counts = {c: len(buckets.get(c, [])) for c in classes}
            print("bucket counts before sampling:", pre_counts)
            print("roots tried:", [str(x) for x in bases])

        if force_all_subtypes:
            sp_path = cat_path.with_name(subtypes_csv_name)
            if not sp_path.exists():
                raise SystemExit(f"subtypes csv not found: {sp_path}")
            with sp_path.open(newline="", encoding="utf-8") as f:
                sp_reader = csv.DictReader(f)
                sp_headers = [h.strip() for h in (sp_reader.fieldnames or [])]
                sp_rows = [row for row in sp_reader]

            sp_label_field = _pick_field(sp_headers, (cfg.get("label_field") or settings.get("label_field", "")).strip(), label_candidates)
            sp_path_field = _pick_field(sp_headers, (cfg.get("path_field") or settings.get("path_field", "")).strip(), path_candidates)

            for need in ("hand_shape", "hand_shape_subtype", "foot_shape", "foot_shape_subtype"):
                if need not in sp_headers:
                    raise SystemExit(f"missing column '{need}' in {sp_path.name}")

            anno = {c: [] for c in classes}
            miss_prints = 0

            for r in sp_rows:
                label_raw = str(r.get(sp_label_field, "")).strip()
                lbl = None
                if label_map:
                    mapped = label_map.get(label_raw) or label_map.get(label_raw.lower())
                    if mapped:
                        if mapped.isdigit():
                            idx = int(mapped)
                            if 0 <= idx < len(classes):
                                lbl = classes[idx]
                        else:
                            for c in classes:
                                if c.lower() == mapped.lower():
                                    lbl = c
                                    break
                if lbl is None and label_raw.isdigit():
                    idx_lbl = int(label_raw)
                    if 0 <= idx_lbl < len(classes):
                        lbl = classes[idx_lbl]
                if lbl is None:
                    for c in classes:
                        if c.lower() == label_raw.lower():
                            lbl = c
                            break
                if lbl is None:
                    continue

                q = str(r.get(sp_path_field, "")).strip()
                if not q:
                    continue
                if q.startswith("http://") or q.startswith("https://"):
                    continue
                q = q.strip('"').strip("'").replace("\\", "/")

                candidates = []
                q_path = Path(q)
                if q_path.is_absolute():
                    candidates.append(q_path)
                else:
                    for base in bases:
                        candidates.append(base / q)
                        if not image_root.is_absolute():
                            candidates.append(base / image_root / q)

                found = None
                try_exts = []
                if ext:
                    try_exts.append(ext if str(ext).startswith(".") else f".{ext}")
                try_exts += [".png", ".jpg", ".jpeg", ".webp"]

                for base in candidates:
                    if base.exists():
                        found = base
                        break
                    stem = base.with_suffix("")
                    for e in try_exts:
                        p = stem.with_suffix(e)
                        if p.exists():
                            found = p
                            break
                    if found:
                        break

                if not found:
                    if debug and miss_prints < 12:
                        print("MISS (subtypes):", str(candidates[0]), "q=", q)
                        miss_prints += 1
                    continue

                anno[lbl].append((
                    str(found),
                    str(r.get("hand_shape", "")).strip(),
                    str(r.get("hand_shape_subtype", "")).strip(),
                    str(r.get("foot_shape", "")).strip(),
                    str(r.get("foot_shape_subtype", "")).strip(),
                ))

            req_hand = {
                ("round", "circle"), ("round", "oval"), ("round", "oval2"),
                ("edgy", "triangle"), ("edgy", "square"), ("edgy", "trapezoid"),
            }
            req_foot = {
                ("flat", "trapezoid"), ("flat", "rounded"), ("flat", "square"),
                ("flat", "5sided"), ("flat", "lshaped"),
                ("pointy", "trapezoid"), ("pointy", "rounded"), ("pointy", "square"),
                ("pointy", "3sided"), ("pointy", "4sided"),
            }

            for c in classes:
                pool = anno.get(c, [])
                by_foot, by_hand = {}, {}
                for path, hs, hsub, fs, fsub in pool:
                    by_foot.setdefault((fs, fsub), []).append((path, hs, hsub, fs, fsub))
                    by_hand.setdefault((hs, hsub), []).append((path, hs, hsub, fs, fsub))

                target_n = 12
                selected = []
                covered_hand = set()

                for fs, fsub in req_foot:
                    cand_list = by_foot.get((fs, fsub), [])
                    if not cand_list:
                        continue
                    rng.shuffle(cand_list)
                    chosen = None
                    for it in cand_list:
                        if (it[1], it[2]) not in covered_hand:
                            chosen = it
                            break
                    if chosen is None:
                        chosen = cand_list[0]
                    selected.append(chosen)
                    covered_hand.add((chosen[1], chosen[2]))
                    if len(selected) >= target_n:
                        break

                if len(selected) < target_n:
                    for hs, hsub in (req_hand - covered_hand):
                        cand_list = by_hand.get((hs, hsub), [])
                        if not cand_list:
                            continue
                        rng.shuffle(cand_list)
                        selected.append(cand_list[0])
                        covered_hand.add((hs, hsub))
                        if len(selected) >= target_n:
                            break

                seen_paths = set(it[0] for it in selected)
                if len(selected) < target_n:
                    rest = [it for it in pool if it[0] not in seen_paths]
                    rng.shuffle(rest)
                    selected.extend(rest[: max(0, target_n - len(selected))])
                    seen_paths = set(it[0] for it in selected)

                if len(selected) < target_n:
                    backup = [p for p in buckets.get(c, []) if p not in seen_paths]
                    rng.shuffle(backup)
                    selected.extend([(p, "", "", "", "") for p in backup[: max(0, target_n - len(selected))]])

                paths = []
                seen = set()
                for it in selected:
                    pth = it[0]
                    if pth not in seen:
                        seen.add(pth)
                        paths.append(pth)
                examples[c] = paths[:target_n]

        else:
            for c in classes:
                pool = buckets.get(c, [])
                rng.shuffle(pool)
                examples[c] = pool[:per_class]

        missing = [c for c in classes if not examples.get(c)]
        if missing:
            raise SystemExit(f"no examples collected for: {', '.join(missing)}")
    else:
        for spec in index_specs:
            if "=" not in spec:
                continue
            cname, idxs = spec.split("=", 1)
            cname = cname.strip()
            if cname not in examples:
                continue
            for s in [t.strip() for t in idxs.split(",") if t.strip()]:
                idx = s.zfill(3)
                p = str(image_root / f"robot_{idx}{ext}")
                examples[cname].append(p)

    outd = Path(out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    prompt_text = build_prompt(classes, examples, n_concepts)
    (outd / "llm_prompt.txt").write_text(prompt_text, encoding="utf-8")
    if not use_llm:
        write_stub(outd, fmt)

    total_examples = sum(len(examples.get(c, [])) for c in classes)
    if use_llm and total_examples == 0:
        raise SystemExit("no examples provided; pass --from-catalog with --per-class, or --indices for each class")

    if use_llm:
        image_list = []
        for c in classes:
            image_list.extend(examples.get(c, []))
        key = os.environ.get(api_key_env, "")
        if not key:
            raise SystemExit(f"missing API key in env: {api_key_env}")
        client = _make_client(provider, model_name, key)
        raw = client.generate(prompt_text, image_list)

        items = _extract_jsonl(raw)
        if len(items) > n_concepts:
            items = items[:n_concepts]
        outp = outd / out_file
        if len(items) < n_concepts:
            rawp = outd / "candidates.llm.raw.txt"
            rawp.write_text(raw, encoding="utf-8")
            raise SystemExit(f"expected {n_concepts} JSONL lines, parsed {len(items)}; raw saved to {rawp}")
        with outp.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"wrote: {outp} ({len(items)} lines)")

    ex_counts = {c: len(examples.get(c, [])) for c in classes}
    print("wrote:", outd / "llm_prompt.txt")
    print("examples:", ex_counts)
    if not use_llm:
        stub_name = "candidates.stub." + ("jsonl" if fmt == "jsonl" else ("json" if fmt == "json" else "txt"))
        print("wrote stub candidates:", outd / stub_name)

cfg = settings.copy()
cfg.update(parse_cli([]))
run(cfg)
