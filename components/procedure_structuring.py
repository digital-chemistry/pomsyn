import json
import os
import re
import shutil
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from google import genai


def _strip_code_fences(txt: str) -> str:
    return re.sub(r"^```[\w-]*\s*|\s*```$", "", txt.strip(), flags=re.DOTALL)

def _extract_json_block(txt: str) -> str:
    i = txt.find("{")
    j = txt.rfind("}")
    return txt[i:j+1] if i != -1 and j != -1 and j > i else txt

def _escape_bad_backslashes(s: str) -> str:
    return re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', s)

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _safe_json_loads(raw: str):
    txt = _strip_code_fences(raw)
    txt = _extract_json_block(txt)
    try:
        return json.loads(txt)
    except Exception:
        pass
    cleaned = _escape_bad_backslashes(txt)
    cleaned = _remove_trailing_commas(cleaned)
    return json.loads(cleaned)

ALLOWED_OPS = {
    'Add','Stir','Heat','Cool','Wait','Filter','Wash','Dry','Dissolve','Evaporate',
    'Crystallize','Precipitate','Adjust pH','Separate','Decant','Centrifuge','Extract',
    'Transfer','Purge','Grind'
}

OP_ALIASES = {
    "recrystallize": "Crystallize",
    "re-crystallize": "Crystallize",
    "recrystalize": "Crystallize",
    "re-crystalize": "Crystallize",
    "neutralize": "Adjust pH",
    "acidify": "Adjust pH",
    "basify": "Adjust pH",
    "filtration": "Filter",
    "wash": "Wash",
    "evaporate to dryness": "Evaporate",
    "vacuum dry": "Dry",
    "air dry": "Dry",
    "quench": "Cool",
    "shake": "Stir",
    "shaking": "Stir",
    "mix": "Stir",
    "mixing": "Stir",
    "combine": "Add",
    "charge": "Add",       
    "load": "Add",
}

UNIT_ALIASES = {
    ### volumes
    "ml": "mL", "m l": "mL",
    "l": "L",
    "µl": "µL", "μl": "µL", "ul": "µL",

    ### mass
    "g": "g", "mg": "mg", "µg": "µg", "μg": "µg",

    ### time
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "minute": "min", "minutes": "min", "min": "min",
    "second": "s", "sec": "s", "s": "s",

    ### temperature
    "c": "°C", "°c": "°C", "degc": "°C", "deg c": "°C",
    "k": "K", "°f": "°F", "f": "°F",

    ### concentration
    "m": "mol/L", "mol/l": "mol/L", "mol·l⁻¹": "mol/L", "mol·l-1": "mol/L",
    "%": "%", "v/v": "v/v", "w/w": "w/w", "w/v": "w/v",
}

def _norm_unit(u: str | None) -> str | None:
    if not u:
        return None
    s = str(u).strip()
    s = s.replace("μ", "µ")
    s_simple = re.sub(r"\s+|·", "", s).lower()
    if s_simple in UNIT_ALIASES:
        return UNIT_ALIASES[s_simple]
    if s in {"°C", "K", "°F", "mL", "L", "µL", "g", "mg", "µg", "min", "h", "s"}:
        return s
    return s

def _norm_op(op: str) -> str:
    if not isinstance(op, str):
        return op
    k = re.sub(r"\s+", " ", op).strip().lower()
    canon = OP_ALIASES.get(k)
    return canon if canon else op.strip()

ACT_OPS = {
    "Add","Stir","Dissolve","Heat","Cool","Transfer","Filter","Extract","Precipitate",
    "Crystallize","Wash","Dry","Evaporate","Adjust pH","Purge","Grind","Wait","Separate",
    "Decant","Centrifuge","Shake"
}

def _tokenize_name(nm: str):
    return [t for t in re.split(r"[^A-Za-z0-9µ·∙°]+", nm) if t]

def _first_line_hit(lines, name: str):
    name_norm = name.strip()
    for i, ln in enumerate(lines, 1):
        if name_norm and name_norm in ln:
            return i
    toks = _tokenize_name(name_norm)
    if toks:
        for i, ln in enumerate(lines, 1):
            if all(t in ln for t in toks[:2]):
                return i
    return None

def _normalize_steps(steps):
    for i, s in enumerate(steps, 1):
        s.setdefault("step_number", i)
        usage = s.get("reagent_usage", [])
        if usage is None:
            usage = []
        elif isinstance(usage, dict):
            usage = [usage]
        elif not isinstance(usage, list):
            usage = []
        fixed = []
        for u in usage:
            rid = u.get("reagent_id")
            if rid:
                fixed.append({"reagent_id": rid})
        s["reagent_usage"] = fixed
        if "source_line_idx" in s:
            try:
                s["source_line_idx"] = int(s["source_line_idx"])
            except Exception:
                s.pop("source_line_idx", None)
    return steps

def _build_step_ranges(steps, last_line):
    if not steps:
        return []
    pairs = []
    indices = []
    for s in steps:
        idx = s.get("source_line_idx")
        indices.append(idx if isinstance(idx, int) and idx >= 1 else None)
    prev = 1
    for k, idx in enumerate(indices):
        start = idx if idx else prev
        nxt = next((indices[j] for j in range(k + 1, len(indices)) if indices[j]), None)
        end = (nxt - 1) if nxt else (last_line if last_line and last_line >= start else start)
        pairs.append((start, end))
        prev = start
    return pairs


### Structuring Pipeline

class colors:
    YELLOW = "\033[93m"; GREEN = "\033[92m"; RESET = "\033[0m"; RED = "\033[91m"; BLUE = "\033[94m"

class StructuringPipeline:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.prompt_template = None

    def setup(self):
        api_key = self.config.get("api_key")
        if not api_key:
            print(f"{colors.RED}ERROR missing GOOGLE API key{colors.RESET}")
            return False

        self.client = genai.Client(api_key=api_key)
        self.model_id = self.config.get("model_name", "gemini-3.1-flash-lite-preview")

        pf = self.config.get("prompt_file")
        self.prompt_template = self._get_prompt_from_file(pf)

        if not self.prompt_template:
            print(f"{colors.RED}CRITICAL ERROR: No structuring prompt file found.{colors.RESET}")
            print(f"{colors.YELLOW}Checked paths: {pf}{colors.RESET}")
            return False

        out = self.config.get("output_folder")
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)
        return True

    def _get_prompt_from_file(self, paths: list[str] | str) -> Optional[str]:
        search_paths = [paths] if isinstance(paths, (str, Path)) else paths
        
        if search_paths:
            for path_str in search_paths:
                if path_str:
                    p = Path(path_str)
                    if p.exists():
                        try:
                            content = p.read_text(encoding="utf-8").strip()
                            if content:
                                return content
                        except Exception as e:
                            print(f"{colors.YELLOW}Warning: Could not read {path_str}: {e}{colors.RESET}")
                            continue 
        
        return None

    def _call_agent_json(self, prompt, tag):
        dbg_dir = os.path.join(self.config["output_folder"], "_debug")
        os.makedirs(dbg_dir, exist_ok=True)

        def ask(p):
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=p,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0
                }
            )
            return resp.text.strip()

        raw = ask(prompt)
        try:
            return _safe_json_loads(raw)
        except Exception as e1:
            with open(os.path.join(dbg_dir, f"{tag}_raw_attempt1.txt"), "w", encoding="utf-8") as f:
                f.write(raw)
            retry_prompt = (
                "Return ONLY valid JSON. No markdown. No comments. "
                "Respond with an object that has exactly one key \"procedure_steps\".\n\n"
                + prompt
            )
            raw2 = ask(retry_prompt)
            with open(os.path.join(dbg_dir, f"{tag}_raw_attempt2.txt"), "w", encoding="utf-8") as f:
                f.write(raw2)
            try:
                return _safe_json_loads(raw2)
            except Exception as e2:
                with open(os.path.join(dbg_dir, f"{tag}_parse_error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Parse errors\nattempt1 {e1}\nattempt2 {e2}\n")
                raise

    def _validate_steps(self, steps_json, manifest):
        id_to_name = {f"R{i}": (r.get("name") or r.get("chemical") or "")
                    for i, r in enumerate(manifest.get("reagents", []), 1)}

        steps = steps_json.get("procedure_steps")
        if not isinstance(steps, list):
            raise ValueError("procedure_steps must be a list")

        used = set()
        out = []

        for i, s in enumerate(steps, 1):
            s.setdefault("step_number", i)

            op_raw = (s.get("unit_operation") or "").strip()
            op = _norm_op(op_raw)
            if not op or op not in ALLOWED_OPS:
                op = "Stir"
            s["unit_operation"] = op

            usage = s.get("reagent_usage") or []
            if isinstance(usage, dict):
                usage = [usage]

            fixed = []
            for u in usage:
                rid = (u.get("reagent_id") or "").strip()
                if rid in id_to_name:
                    item = {"reagent_id": rid, "name": id_to_name[rid]}
                    amt = u.get("amount")
                    if isinstance(amt, dict) and "value" in amt and "unit" in amt:
                        item["amount"] = {"value": amt["value"], "unit": _norm_unit(amt["unit"])}
                    fixed.append(item)
                    used.add(rid)
                else:
                    continue
            s["reagent_usage"] = fixed

            params = s.pop("parameters", None)

            def _norm_unit(u):
                if not isinstance(u, str): return u
                u = u.strip().replace("deg", "").replace("℃", "°C").replace("ºC", "°C")
                return {"c":"°C","°c":"°C","f":"°F","°f":"°F","k":"K"}.get(u.lower(), u)

            if params is not None:
                items = params if isinstance(params, list) else [params]
                for p in items:
                    if not isinstance(p, dict):
                        continue

                    t = p.get("time") or p.get("duration")
                    if isinstance(t, dict):
                        if "end_condition" in t and t["end_condition"]:
                            s["time"] = {"end_condition": t["end_condition"]}
                        else:
                            val = t.get("value"); unit = _norm_unit(t.get("unit"))
                            if isinstance(val, (int, float)) and isinstance(unit, str):
                                s["time"] = {"value": float(val), "unit": unit}

                    tgt = p.get("target") if isinstance(p.get("target"), dict) else {}
                    if "temperature" in p and isinstance(p["temperature"], dict):
                        tgt.setdefault("temperature", p["temperature"])
                    elif "temperature" in p and isinstance(p["temperature"], (int, float)):
                        tgt["temperature"] = {"value": float(p["temperature"]), "unit": "°C"}

                    TT = tgt.get("temperature")
                    if isinstance(TT, (int, float)):
                        TT = {"value": float(TT), "unit": "°C"}
                    if isinstance(TT, dict) and "value" in TT:
                        unit = _norm_unit(TT.get("unit")) or "°C"
                        s.setdefault("target", {})["temperature"] = {"value": float(TT["value"]), "unit": unit}

                    ph = tgt.get("pH") or tgt.get("ph") or p.get("target_ph")
                    if isinstance(ph, (int, float)):
                        s.setdefault("target", {})["pH"] = {"value": float(ph)}
                    elif isinstance(ph, dict) and "value" in ph:
                        s.setdefault("target", {})["pH"] = {"value": float(ph["value"])}

                    rid = p.get("reagent_id")
                    if isinstance(rid, str):
                        for key in ("amount", "volume", "mass"):
                            valdict = p.get(key)
                            if isinstance(valdict, dict):
                                val = valdict.get("value"); unit = _norm_unit(valdict.get("unit"))
                                if isinstance(val, (int, float)) and isinstance(unit, str):
                                    for u in s.get("reagent_usage", []) or []:
                                        if u.get("reagent_id") == rid:
                                            u["amount"] = {"value": float(val), "unit": unit}

            if isinstance(s.get("time"), dict) and "unit" in s["time"]:
                s["time"]["unit"] = _norm_unit(s["time"]["unit"])
            if isinstance(s.get("target"), dict):
                TT = s["target"].get("temperature")
                if isinstance(TT, dict) and "unit" in TT:
                    TT["unit"] = _norm_unit(TT["unit"])
                if isinstance(s["target"].get("pH"), dict):
                    s["target"]["pH"].pop("unit", None)

            out.append(s)

        all_ids = {f"R{i}" for i in range(1, len(id_to_name) + 1)}
        missing = sorted(all_ids - used)
        steps_json["procedure_steps"] = out
        return {"used_ids": sorted(used), "missing_ids": missing, "total_ids": len(id_to_name)}

    def process_file(self, text_filepath):
        filename = os.path.basename(text_filepath)
        manifest_filename = filename.replace(".txt", "_manifest.json")
        manifest_filepath = os.path.join(self.config["manifests_folder"], manifest_filename)
        if not os.path.exists(manifest_filepath):
            tqdm.write(f" {colors.YELLOW}Manifest not found for {filename} Skipping{colors.RESET}")
            return False
        try:
            with open(text_filepath, "r", encoding="utf-8") as f:
                source_text = f.read()
            with open(manifest_filepath, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            r_lines = []
            for i, r in enumerate(manifest.get("reagents", []), 1):
                nm = r.get("name") or r.get("chemical")
                r_lines.append(f"R{i}: {nm}")
            reagents_str = "\n".join(r_lines)

            prompt = (
                self.prompt_template
                .replace("{{source_text_here}}", source_text)
                .replace("{{reagents_list_here}}", reagents_str)
            )

            steps_json = self._call_agent_json(prompt, tag=filename[:-4])

            lines = [ln.strip() for ln in source_text.splitlines() if ln.strip()]
            steps = steps_json.get("procedure_steps", [])
            steps = _normalize_steps(steps)

            if not steps:
                steps = [{"step_number": 1, "unit_operation": "Stir", "reagent_usage": []}]
                steps_json["procedure_steps"] = steps

            id_to_name = {f"R{i}": (r.get("name") or r.get("chemical") or "")
                          for i, r in enumerate(manifest.get("reagents", []), 1)}
            all_ids = set(id_to_name.keys())
            used_ids = {
                u["reagent_id"]
                for s in steps
                for u in (s.get("reagent_usage") or [])
                if isinstance(u, dict) and "reagent_id" in u
            }
            missing = sorted(all_ids - used_ids)

            last_line = len(lines) if lines else 1

            def _safe_build_ranges(_steps, _last):
                if not _steps:
                    return []
                return _build_step_ranges(_steps, _last)

            ranges = _safe_build_ranges(steps, last_line)

            def _clamp_idx(idx: int) -> int:
                if not steps:
                    return 0
                if idx < 0:
                    return 0
                if idx >= len(steps):
                    return len(steps) - 1
                return idx

            for rid in missing:
                nm = id_to_name[rid]
                hit = _first_line_hit(lines, nm)

                attach_idx = next((i for i, s in enumerate(steps) if s.get("unit_operation") in ACT_OPS), 0)

                if hit and ranges:
                    for i, (a, b) in enumerate(ranges):
                        if a <= hit <= b:
                            attach_idx = i
                            break

                attach_idx = _clamp_idx(attach_idx)
                steps[attach_idx].setdefault("reagent_usage", [])
                steps[attach_idx]["reagent_usage"].append({"reagent_id": rid})

            steps_json["procedure_steps"] = steps

            stats = self._validate_steps(steps_json, manifest)
            if stats["missing_ids"]:
                tqdm.write(f" {colors.YELLOW}{filename} missing {stats['missing_ids']}{colors.RESET}")
            else:
                tqdm.write(f" {colors.BLUE}{filename} ok. reagents used {len(stats['used_ids'])} of {stats['total_ids']}{colors.RESET}")

            out_json_path = os.path.join(self.config["output_folder"], filename.replace(".txt", "_steps.json"))
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(steps_json, f, indent=2, ensure_ascii=False)
            print(f"{colors.BLUE}saved json steps to {out_json_path}{colors.RESET}")
            return True

        except Exception as e:
            tqdm.write(f" - {colors.RED}Error processing {filename}: {e}{colors.RESET}")
            return False

    def run(self):
        print(f"\033[92m  >>>  Starting Phase 2: Structuring\033[0m")
        if not self.setup():
            return
        src = self.config["source_folder"]
        files = [os.path.join(src, f) for f in os.listdir(src) if f.startswith("SYN") and f.endswith(".txt")]
        print(f"Found {len(files)} source text files for structuring.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_workers", 5)) as ex:
            list(tqdm(ex.map(self.process_file, files), total=len(files), desc="Structuring Steps"))
        print(f"Complete. Steps at '{self.config['output_folder']}'.")
