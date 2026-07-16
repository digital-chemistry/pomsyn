import os
import json
import re
import shutil
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from google import genai
from tqdm import tqdm


_AMOUNT_RE = re.compile(r"^\s*(?:~|≈)?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Zµμ·/°\-]+)?\s*$")

UNIT_ALIASES = {
    "ml": "mL", "mL": "mL",
    "l": "L", "L": "L",
    "g": "g", "mg": "mg",
    "µl": "µL", "μl": "µL", "ul": "µL",
    "m": "mol/L", "mol/l": "mol/L", "mol·l⁻¹": "mol/L", "mol·l-1": "mol/L",
}

def _norm_unit(u: Optional[str]) -> Optional[str]:
    if not u: return None
    u = u.strip().replace("μ", "µ")
    return UNIT_ALIASES.get(u.lower(), u)

def parse_amount(s: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not s or not isinstance(s, str): return (None, None)
    m = _AMOUNT_RE.match(s.strip())
    if not m: return (None, None)
    try:
        v = float(m.group("val"))
    except Exception:
        return (None, None)
    unit = _norm_unit(m.group("unit"))
    return (v, unit)

def try_aggregate_amount(a: Optional[str], b: Optional[str]) -> Optional[str]:
    v1,u1 = parse_amount(a)
    v2,u2 = parse_amount(b)
    if v1 is None: return b
    if v2 is None: return a
    if u1 and u2 and u1 == u2:
        total = v1 + v2
        text = str(int(total)) if abs(total - int(total)) < 1e-9 else f"{total:g}"
        return f"{text}{u1}"
    return a

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

MOLAR_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:M|mol/?L|mol·L⁻¹)\b", re.I)
PCT_RE   = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(w/?w|v/?v|w/?v)?\b", re.I)
PURITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:purity|pure)\b", re.I)

_CONC_WHITELIST = (
    "acid", "hydrochloric", "sulfuric", "nitric", "phosphoric",
    "naoh", "koh", "ammonia", "ammonium hydroxide", "solution"
)

_CONC_BLACKLIST = ("water",)

def _looks_like_concentrated_stock(name: str) -> bool:
    nm = (name or "").strip().casefold()
    if any(bad in nm for bad in _CONC_BLACKLIST):
        return False
    return any(tok in nm for tok in _CONC_WHITELIST)

def parse_concentration_local(name: str) -> Optional[Dict[str, Any]]:
    nm = (name or "")
    if not _looks_like_concentrated_stock(nm):
        return None
    m = MOLAR_RE.search(nm)
    if m:
        return {"value": float(m.group(1)), "unit": "mol/L"}
    m = PCT_RE.search(nm)
    if m:
        val = float(m.group(1))
        basis_raw = m.group(2)
        conc = {"value": val, "unit": "%"}
        if basis_raw:
            basis = basis_raw.lower().replace("/", "")
            mapping = {"ww": "w/w", "vv": "v/v", "wv": "w/v"}
            if basis in mapping:
                conc["basis"] = mapping[basis]
        return conc
    return None

def parse_purity_local(name: str) -> Optional[float]:
    m = PURITY_RE.search(name or "")
    if m:
        return float(m.group(1))
    return None

def parse_concentration(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    m = MOLAR_RE.search(text)
    if m:
        val = float(m.group(1))
        return {"value": val, "unit": "mol/L"}
    m = PCT_RE.search(text)
    if m:
        val = float(m.group(1))
        basis_raw = m.group(2)
        conc = {"value": val, "unit": "%"}
        if basis_raw:
            basis = basis_raw.lower().replace("/", "")
            if basis in {"ww","vv","wv"}:
                mapping = {"ww":"w/w", "vv":"v/v", "wv":"w/v"}
                conc["basis"] = mapping[basis]
        return conc
    return None

def parse_purity(text: str) -> Optional[float]:
    if not text: return None
    m = PURITY_RE.search(text)
    if m:
        return float(m.group(1))
    return None

def _stock_signature(r: Dict[str, Any]) -> Tuple:
    conc = r.get("concentration") or {}
    pur  = r.get("purity_percent")
    sig = (
        (r.get("name") or "").strip().casefold(),
        conc.get("value"), conc.get("unit"), conc.get("basis"),
        pur if isinstance(pur, (int, float)) else None,
    )
    return sig

def _postprocess_reagents(raw_reagents: List[Dict[str, Any]], source_text: str) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []

    for r in raw_reagents:
        name = (r.get("name") or "").strip()
        amt  = r.get("amount")
        if isinstance(amt, str) and amt.strip().lower() == "null":
            amt = None

        conc = parse_concentration_local(name)
        pur  = parse_purity_local(name)

        entry: Dict[str, Any] = {"name": name, "amount": amt}
        if conc: entry["concentration"] = conc
        if pur is not None: entry["purity_percent"] = pur
        processed.append(entry)

    by_sig: Dict[Tuple, Dict[str, Any]] = {}
    for r in processed:
        sig = _stock_signature(r)
        if sig not in by_sig:
            by_sig[sig] = dict(r)
        else:
            by_sig[sig]["amount"] = try_aggregate_amount(by_sig[sig].get("amount"), r.get("amount"))

    return list(by_sig.values())

_AMOUNT_RE = re.compile(
    r"^\s*(~|≈)?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Zµμ·/°\-]+)?\s*$"
)

UNIT_ALIASES = {
    "ml": "mL", "mL": "mL",
    "l": "L", "L": "L",
    "g": "g",
    "mg": "mg",
    "µl": "µL", "μl": "µL", "uL": "µL",
    "mol/l": "mol/L", "mol·l⁻¹": "mol/L", "mol·l⁻¹": "mol/L", "m": "mol/L",
    "°c": "°C", "c": "°C"
}

def _norm_unit(u: Optional[str]) -> Optional[str]:
    if not u: return None
    u = u.strip()
    u = u.replace("μ", "µ")
    key = u.lower()
    return UNIT_ALIASES.get(key, u)

def parse_amount(s: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not s or not isinstance(s, str): return (None, None)
    m = _AMOUNT_RE.match(s.strip())
    if not m: return (None, None)
    try:
        v = float(m.group("val"))
    except Exception:
        return (None, None)
    unit = _norm_unit(m.group("unit"))
    return (v, unit)

def format_amount(val: Optional[float], unit: Optional[str]) -> Optional[str]:
    if val is None or unit is None: return None
    if abs(val - int(val)) < 1e-9:
        vtxt = str(int(val))
    else:
        vtxt = f"{val:g}"
    return f"{vtxt}{unit}"

def try_aggregate_amount(a: Optional[str], b: Optional[str]) -> Optional[str]:
    v1,u1 = parse_amount(a)
    v2,u2 = parse_amount(b)
    if v1 is None: return b
    if v2 is None: return a
    if u1 and u2 and u1 == u2:
        return format_amount(v1+v2, u1)
    return a


### Extraction Pipeline

class colors:
    YELLOW = "\033[93m"; GREEN = "\033[92m"; RESET = "\033[0m"; RED = "\033[91m"

class ExtractionPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.prompt_template = None
        self.db_path = None
        self.database = {}

    def setup(self) -> bool:
        api_key = self.config.get("api_key")
        if not api_key:
            print(f"{colors.RED}ERROR: missing GOOGLE API key{colors.RESET}")
            return False
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = self.config.get("model_name", "gemini-3.1-flash-lite-preview")

        pf = self.config.get("prompt_file")
        self.prompt_template = self._get_prompt_from_file(pf)

        if not self.prompt_template:
            print(f"{colors.RED}CRITICAL ERROR: No prompt file found.{colors.RESET}")
            print(f"{colors.YELLOW}Check that your Workdir/prompts or Internal package prompts exist.{colors.RESET}")
            return False

        out = self.config.get("output_folder")
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)

        self.db_path = self.config.get("database_file") or "chemical_database.json"
        self._load_database()
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

    def _load_database(self):
        if self.db_path and Path(self.db_path).exists():
            try:
                data = json.loads(Path(self.db_path).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.database = data
                else:
                    self.database = {}
                print(f"Loaded {len(self.database)} chemicals from existing database.")
            except Exception:
                print(f"{colors.YELLOW}Warning: could not read {self.db_path}; starting fresh{colors.RESET}")
                self.database = {}
        else:
            self.database = {}

    def _save_database(self):
        if not self.db_path: return
        Path(self.db_path).write_text(json.dumps(self.database, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Chemical database now contains {len(self.database)} entries. Saved to '{self.db_path}'.")

    def _call_extractor(self, source_text: str) -> Dict:
            prompt = self.prompt_template.replace("{{source_text_here}}", source_text)

            def ask(p):
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=p,
                    config={
                        "temperature": 0,
                        "response_mime_type": "application/json"
                    }
                )
                return (resp.text or "").strip()

            txt = ask(prompt)
            dbg_dir = Path(self.config["output_folder"]) / "_debug"

            try:
                data = _safe_json_loads(txt)
            except Exception as e1:
                # If the first attempt fails (e.g., unescaped quotes), log it and trigger a retry
                dbg_dir.mkdir(parents=True, exist_ok=True)
                (dbg_dir / "extractor_raw_attempt1.txt").write_text(txt, encoding="utf-8")

                retry_prompt = (
                    "Return ONLY valid JSON. Escape all inner quotes. No markdown. "
                    "Respond with a single object containing the key \"reagents\".\n\n"
                    + prompt
                )
                txt2 = ask(retry_prompt)

                try:
                    data = _safe_json_loads(txt2)
                except Exception as e2:
                    (dbg_dir / "extractor_raw_attempt2.txt").write_text(txt2, encoding="utf-8")
                    raise ValueError(f"JSON parsing failed after retry. Attempt 1: {e1}. Attempt 2: {e2}")

            if not isinstance(data, dict) or "reagents" not in data or not isinstance(data["reagents"], list):
                raise ValueError("Extractor returned valid JSON, but invalid structure")

            clean = []
            for r in data["reagents"]:
                nm = (r.get("name") or "").strip()
                amt = r.get("amount")
                clean.append({"name": nm, "amount": amt})
            return {"reagents": clean}

    def _norm_name_key(self, name: str) -> str:
        return (name or "").strip().casefold()

    def _aggregate_reagents(self, items: List[Dict]) -> List[Dict]:
        by_name: Dict[str, Dict] = {}
        for r in items:
            nm = (r.get("name") or "").strip()
            if not nm:
                continue
            key = self._norm_name_key(nm)
            amt = r.get("amount")

            if key not in by_name:
                by_name[key] = {"name": nm, "amount": amt}
            else:
                by_name[key]["amount"] = try_aggregate_amount(by_name[key].get("amount"), amt)
        return list(by_name.values())

    def _strict_pubchem_enrich(self, name: str) -> Optional[int]:
        return None

    def _maybe_enrich_and_update_db(self, reagent: Dict):
        key = self._norm_name_key(reagent["name"])
        existing = self.database.get(key, {})
        cid = existing.get("pubchem_cid")
        if cid is None and self.config.get("enrich_pubchem", False):
            safe_cid = self._strict_pubchem_enrich(reagent["name"])
            if safe_cid is not None:
                existing["pubchem_cid"] = safe_cid
        existing["name"] = reagent["name"]
        existing["amount"] = reagent.get("amount")
        self.database[key] = existing

    def process_file(self, text_filepath: str) -> bool:
        filename = os.path.basename(text_filepath)
        manifest_name = filename.replace(".txt", "_manifest.json")
        out_path = Path(self.config["output_folder"]) / manifest_name

        try:
            source = Path(text_filepath).read_text(encoding="utf-8")
            extracted = self._call_extractor(source)
            reagents = _postprocess_reagents(extracted["reagents"], source)

            for r in reagents:
                self._maybe_enrich_and_update_db(r)

            manifest = {
                "source_file": filename,
                "reagents": reagents
            }
            out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            return True

        except Exception as e:
            tqdm.write(f" - {colors.RED}Extraction error for {filename}: {e}{colors.RESET}")
            return False

    def run(self):
        print(f"\033[92m  >>>  Starting Phase 1: Extraction\033[0m")
        if not self.setup():
            return
        src = self.config.get("source_folder")
        files = sorted([str(p) for p in Path(src).glob("SYN*.txt")])
        print(f"Found {len(files)} files to extract reagents.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_workers", 6)) as ex:
            list(tqdm(ex.map(self.process_file, files), total=len(files), desc="Extracting reagents"))
        self._save_database()
        print(f"Complete. Manifests at '{self.config['output_folder']}'.")
