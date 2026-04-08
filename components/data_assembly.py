import json
import os
import re
from pathlib import Path
import concurrent.futures
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm


STEP_HDR_RE = re.compile(r"^\s*\d+\.\s*\*\*Operation:\*\*\s*(.+?)\s*$")
FIELD_RE    = re.compile(r"^\s*-\s*\*\*(.+?)\*\*:\s*(.*)$")

def _norm_unit(u: Optional[str]) -> Optional[str]:
    if not u or not isinstance(u, str):
        return u
    s = u.strip()
    s = s.replace("deg", "")
    s = s.replace("μ", "µ")
    low = s.lower()
    if low in {"c", "°c"}:       return "°C"
    if low in {"f", "°f"}:       return "°F"
    if low in {"k"}:             return "K"
    if low in {"hour", "hours", "hr", "hrs"}:  return "h"
    if low in {"minute", "minutes", "min", "mins"}: return "min"
    if low in {"ml"}:            return "mL"
    if low in {"l"}:             return "L"
    if low in {"g"}:             return "g"
    if low in {"mg"}:            return "mg"
    if low in {"mol/l", "mol·l⁻¹", "mol·l-1", "m"}: return "mol/L"
    return s

def _ensure_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

def _lift_parameters_into_step(s: Dict[str, Any]) -> None:
    params = s.pop("parameters", None)
    if not params:
        pass
    else:
        items = _ensure_list(params)
        for p in items:
            if not isinstance(p, dict):
                continue

            t = p.get("time") or p.get("duration")
            if isinstance(t, dict):
                if "end_condition" in t and t["end_condition"]:
                    s["time"] = {"end_condition": str(t["end_condition"])}
                else:
                    val = t.get("value"); unit = _norm_unit(t.get("unit"))
                    if isinstance(val, (int, float)) and unit:
                        s["time"] = {"value": float(val), "unit": unit}

            tgt = p.get("target") or {}
            if "temperature" in p and isinstance(p["temperature"], dict):
                tgt = dict(tgt)
                tgt["temperature"] = p["temperature"]

            TT = tgt.get("temperature")
            if isinstance(TT, dict):
                tval = TT.get("value"); tunit = _norm_unit(TT.get("unit"))
                if isinstance(tval, (int, float)) and tunit:
                    s.setdefault("target", {})["temperature"] = {"value": float(tval), "unit": tunit}

            ph_obj = tgt.get("pH") or tgt.get("ph") or p.get("target_ph")
            if isinstance(ph_obj, dict) and "value" in ph_obj:
                try:
                    s.setdefault("target", {})["pH"] = {"value": float(ph_obj["value"])}
                except Exception:
                    s.setdefault("target", {})["pH"] = {"value": ph_obj["value"]}

            rid = p.get("reagent_id")
            if rid and isinstance(rid, str):
                for key in ("amount", "volume", "mass"):
                    kv = p.get(key)
                    if isinstance(kv, dict):
                        aval = kv.get("value")
                        aunit = _norm_unit(kv.get("unit"))
                        if isinstance(aval, (int, float)) and aunit:
                            usage = s.get("reagent_usage")
                            if usage is None: usage = []
                            if isinstance(usage, dict): usage = [usage]
                            found = False
                            for u in usage:
                                if u.get("reagent_id") == rid:
                                    u["amount"] = {"value": float(aval), "unit": aunit}
                                    found = True
                                    break
                            if not found:
                                s.setdefault("reagent_usage", usage)
                                s["reagent_usage"].append({
                                    "reagent_id": rid,
                                    "amount": {"value": float(aval), "unit": aunit}
                                })

    if isinstance(s.get("time"), dict) and "unit" in s["time"]:
        s["time"]["unit"] = _norm_unit(s["time"]["unit"])
    if isinstance(s.get("target"), dict):
        TT2 = s["target"].get("temperature")
        if isinstance(TT2, dict) and "unit" in TT2:
            TT2["unit"] = _norm_unit(TT2["unit"])
        pH = s["target"].get("pH")
        if isinstance(pH, dict) and "unit" in pH:
            pH.pop("unit", None)

def _canonicalize_usage_names(usage: Any, id_to_name: Dict[str, str]) -> List[Dict[str, Any]]:
    usage_list = usage or []
    if isinstance(usage_list, dict):
        usage_list = [usage_list]
    fixed: List[Dict[str, Any]] = []
    for u in usage_list:
        if not isinstance(u, dict):
            continue
        rid = u.get("reagent_id")
        nm  = u.get("name")
        if rid and not nm:
            nm = id_to_name.get(rid)
        elif nm and not rid:
            rid = next((k for k, v in id_to_name.items() if v == nm), None)
        if rid:
            entry = {"reagent_id": rid}
            if nm: entry["name"] = nm
            if isinstance(u.get("amount"), dict):
                aval = u["amount"].get("value")
                aunit = _norm_unit(u["amount"].get("unit"))
                if isinstance(aval, (int, float)) and aunit:
                    entry["amount"] = {"value": float(aval), "unit": aunit}
            fixed.append(entry)
    return fixed

def _normalize_steps(steps: List[Dict[str, Any]], id_to_name: Dict[str, str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(steps, 1):
        s.setdefault("step_number", i)
        s["reagent_usage"] = _canonicalize_usage_names(s.get("reagent_usage"), id_to_name)
        _lift_parameters_into_step(s)
        s["reagent_usage"] = _canonicalize_usage_names(s.get("reagent_usage"), id_to_name)
        out.append(s)
    return out


### Assembly Pipeline

class colors:
    YELLOW = "\033[93m"; GREEN = "\033[92m"; RESET = "\033[0m"; RED = "\033[91m"

class AssemblyPipeline:
    def __init__(self, config):
        self.config = config

    def _read_text(self, p: Path) -> str:
        return p.read_text(encoding="utf-8")

    def _read_json(self, p: Path):
        return json.loads(self._read_text(p))

    def _reagent_maps(self, manifest):
        id_to_name = {}
        name_to_id = {}
        for i, r in enumerate(manifest.get("reagents", []), 1):
            rid = f"R{i}"
            nm = r.get("name") or r.get("chemical") or ""
            id_to_name[rid] = nm
            name_to_id[nm] = rid
        return id_to_name, name_to_id

    def _usage_from_ids(self, ids, id_to_name):
        usage = []
        for rid in ids:
            rid = rid.strip()
            if rid in id_to_name:
                usage.append({"reagent_id": rid, "name": id_to_name[rid]})
        return usage

    def _read_steps_json(self, p_json: Path, id_to_name):
        data = self._read_json(p_json)
        steps = data.get("procedure_steps", [])
        for s in steps:
            usage = s.get("reagent_usage", [])
            if isinstance(usage, dict):
                usage = [usage]
            fixed = []
            for u in usage or []:
                rid = u.get("reagent_id")
                nm = u.get("name")
                if rid and not nm:
                    nm = id_to_name.get(rid)
                elif nm and not rid:
                    rid = next((k for k, v in id_to_name.items() if v == nm), None)
                if rid and nm:
                    entry = {"reagent_id": rid, "name": nm}
                    if isinstance(u.get("amount"), dict):
                        entry["amount"] = u["amount"]
                    fixed.append(entry)
            s["reagent_usage"] = fixed

            params = s.pop("parameters", None)
            if params:
                items = params if isinstance(params, list) else [params]
                for p in items:
                    t = p.get("time") or p.get("duration")
                    if isinstance(t, dict):
                        if "end_condition" in t:
                            s["time"] = {"end_condition": t["end_condition"]}
                        else:
                            val = t.get("value"); unit = (t.get("unit") or "").strip()
                            if val is not None and unit:
                                s["time"] = {"value": val, "unit": unit}

                    tgt = p.get("target") or {}
                    if "temperature" in p:
                        tgt["temperature"] = p["temperature"]
                    TT = tgt.get("temperature")
                    if isinstance(TT, dict) and "value" in TT:
                        unit = (TT.get("unit") or "").strip()
                        s.setdefault("target", {})["temperature"] = {"value": TT["value"], "unit": unit}

                    ph = tgt.get("pH") or tgt.get("ph") or p.get("target_ph")
                    if isinstance(ph, dict) and "value" in ph:
                        s.setdefault("target", {})["pH"] = {"value": ph["value"]}

                    rid = p.get("reagent_id")
                    for key in ("amount", "volume", "mass"):
                        valdict = p.get(key)
                        if rid and isinstance(valdict, dict):
                            val = valdict.get("value"); unit = (valdict.get("unit") or "").strip()
                            if val is not None and unit:
                                for u in s.get("reagent_usage", []):
                                    if u.get("reagent_id") == rid:
                                        u["amount"] = {"value": val, "unit": unit}
        for i, s in enumerate(steps, 1):
            s.setdefault("step_number", i)
        return steps

    def _build_final(self, base_name: str, manifest, steps):
        return {
            "procedure_information": {"id": base_name, "title": base_name},
            "product_summary": {},
            "reagents": manifest.get("reagents", []),
            "procedure_steps": steps
        }

    def process_file(self, manifest_filepath):
        filename = os.path.basename(manifest_filepath).replace("_manifest.json", "")
        steps_json_path = Path(self.config["markdown_folder"]) / f"{filename}_steps.json"
        steps_md_path   = Path(self.config["markdown_folder"]) / f"{filename}_steps.md"

        try:
            manifest = self._read_json(Path(manifest_filepath))
            id_to_name, _ = self._reagent_maps(manifest)

            if steps_json_path.exists():
                raw_steps = self._read_steps_json(steps_json_path, id_to_name)
            elif steps_md_path.exists():
                raw_steps = self._read_steps_markdown(steps_md_path, id_to_name)
            else:
                tqdm.write(f" {colors.YELLOW}steps not found for {filename}{colors.RESET}")
                return None

            steps = _normalize_steps(raw_steps, id_to_name)
            final = self._build_final(filename, manifest, steps)
            return final

        except Exception as e:
            tqdm.write(f" - {colors.RED}assembly error for {filename}: {e}{colors.RESET}")
            return None

    def run(self):
        print(f"\033[92m  >>>  Starting Phase 3: Assembly\033[0m")
        manifests = [str(p) for p in Path(self.config["manifests_folder"]).glob("*_manifest.json")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["max_workers"]) as ex:
            results = list(tqdm(ex.map(self.process_file, manifests),
                                total=len(manifests),
                                desc="Merging JSON"))
        final = [r for r in results if r is not None]
        print(f"Transcriber phase complete. Assembled {len(final)} procedures.")
        return final
