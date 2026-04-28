import os
import json
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import re
from google import genai


DELIM = '---PROCEDURE-BREAK---'

_ACTION_START = re.compile(
    r"^(Add|Heat|Cool|Stir|Filter|Wash|Dry|Dissolve|Evaporate|Crystallize|Precipitate|Adjust pH|Separate|Decant|Centrifuge|Extract|Transfer|Purge|Grind|Wait)\b",
    re.I
)

_BARE_CHEM = re.compile(r"^[A-Za-z0-9\[\]\(\)·∙\.\-–, ±××⁻⁺µμ°/\\]+$", re.UNICODE)

_ANALYTICAL_PAT = re.compile(
    r"\b(NMR|¹?H-NMR|¹³C-NMR|³¹P-NMR|IR|FT-IR|UV-Vis|MS|GC-MS|HPLC|elemental analysis)\b", re.I
)

_META_PAT = re.compile(r'\b(duplicate|see previous|as above|same as)\b', re.I)
_ANION_ONLY = re.compile(r'^\s*\[[^\]]+\]\s*[⁻⁺0-9]*\s*$', re.UNICODE)

def _ensure_period(s: str) -> str:
    s = s.strip()
    return s if not s or s.endswith((".", "。")) else s + "."

def _looks_like_bare_fragment(s: str) -> bool:
    t = s.strip().rstrip(".").strip()
    if not t:
        return True
    if " " not in t and _BARE_CHEM.match(t):
        return True
    if t in {".", "-", "•"}:
        return True
    return False

def sanitize_action_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        if ln == DELIM:
            cleaned.append(DELIM)
            continue
        if _looks_like_bare_fragment(ln):
            continue
        if not _ACTION_START.match(ln):
            ln2 = re.sub(r"^(Then\s+|And\s+)", "", ln, flags=re.I).strip()
            if _ACTION_START.match(ln2):
                cleaned.append(_ensure_period(ln2))
        else:
            cleaned.append(_ensure_period(ln))
    return "\n".join(cleaned)

def _split_multi_add(line: str) -> List[str]:
    out = []
    s = line.strip().rstrip('.')
    m_stir = re.search(r",?\s*and\s+stir\b(.*)$", s, flags=re.I)
    stir_clause = None
    if m_stir:
        stir_clause = m_stir.group(0)
        s = s[:m_stir.start()].rstrip(" ,")
    else:
        if re.search(r"\bshak(ing|e)\b", s, re.I):
            stir_clause = " and stir"

    m = re.match(r"^(?:then\s+|and\s+)?add\s+(.+)$", s, flags=re.I)
    if m:
        tail = m.group(1)
        parts = re.split(r",\s*followed by\s+|,\s*and\s+|\s*;\s*", tail, flags=re.I)
        for p in parts:
            p = p.strip().strip(",.")
            if p:
                out.append(_ensure_period(f"Add {p}"))
        if stir_clause:
            m2 = re.search(r"\band\s+stir\b(.*)$", stir_clause, flags=re.I)
            detail = m2.group(1).strip() if m2 else ""
            out.append(_ensure_period(f"Stir{detail}"))
        return out

    if stir_clause:
        core = s.strip()
        if core:
            out.append(_ensure_period(core))
        m2 = re.search(r"\band\s+stir\b(.*)$", stir_clause, flags=re.I)
        detail = m2.group(1).strip() if m2 else ""
        out.append(_ensure_period(f"Stir{detail}"))
        return out

    return [_ensure_period(line)]

def _expand_boiling_water(line: str) -> List[str]:
    s = line.strip().rstrip(".")
    m1 = re.match(r"^(dissolve)\s+(.+?)\s+in\s+(\d+(?:\.\d+)?)\s*mL\s+boiling\s+water$", s, flags=re.I)
    if m1:
        what, vol = m1.group(2), m1.group(3)
        return [f"Heat {vol} mL water to boiling.", _ensure_period(f"Dissolve {what} in boiling water")]
    m2 = re.match(r"^(add)\s+(.+?)\s+to\s+(\d+(?:\.\d+)?)\s*mL\s+boiling\s+water$", s, flags=re.I)
    if m2:
        what, vol = m2.group(2), m2.group(3)
        return [f"Heat {vol} mL water to boiling.", _ensure_period(f"Add {what} to boiling water")]
    return [_ensure_period(line)]

_HEAT_WATER_RE = re.compile(r"^Heat\s+(\d+(?:\.\d+)?)\s*mL\s+water\s+to\s+boiling\.\s*$", re.I)
_ADD_BOILING_WATER_RE = re.compile(r"^Add\s+(\d+(?:\.\d+)?)\s*mL\s+of?\s*boiling\s+water\.\s*$", re.I)

def _coalesce_heat_add(lines: List[str]) -> List[str]:
    out = []; prev_heat = None
    for ln in lines:
        mH = _HEAT_WATER_RE.match(ln)
        if mH:
            vol = float(mH.group(1))
            if vol < 5:
                continue
            if prev_heat == ln:
                continue
            out.append(ln); prev_heat = ln
            continue
        prev_heat = None
        mA = _ADD_BOILING_WATER_RE.match(ln)
        if mA:
            vol = float(mA.group(1))
            if vol < 5:
                continue
            vtxt = str(int(vol)) if vol.is_integer() else f"{vol:g}"
            out.append(_ensure_period(f"Add {vtxt} mL water"))
            continue
        out.append(ln)
    return out

def _drop_useless(lines: List[str]) -> List[str]:
    cleaned = []
    for ln in lines:
        t = ln.strip()
        if not t: continue
        if t == ".": continue
        if re.fullmatch(r"Heat\.?", t, flags=re.I):
            continue
        if _ANALYTICAL_PAT.search(t):
            continue
        cleaned.append(ln)
    return cleaned

def normalize_actions(cleaned_text: str) -> str:
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        s = re.sub(r"^\s*Re-?crystal(li|)ze\b", "Recrystallize", s, flags=re.I)
        s = re.sub(r"\bquench\b", "Cool", s, flags=re.I)
        s = re.sub(r"\b(shake|shaking)\b", "Stir", s, flags=re.I)
        s = re.sub(r"\bleave( it to stand)?\b", "Wait", s, flags=re.I)
        s = re.sub(r"\bair[-\s]?dry\b", "Dry in air", s, flags=re.I)
        s = re.sub(r"\bvacuum[-\s]?dry\b", "Dry under vacuum", s, flags=re.I)

        tmp = []
        for part in _split_multi_add(s):
            tmp.append(part)
        exp = []
        for part in tmp:
            exp.extend(_expand_boiling_water(part))
        for part in exp:
            out.append(_ensure_period(re.sub(r"\s+", " ", part).strip()))
    out = [ln for ln in out if ln and ln != "."]
    out = _coalesce_heat_add(out)
    out = _drop_useless(out)

    dedup = []
    for ln in out:
        if not dedup or dedup[-1] != ln:
            dedup.append(ln)
    return "\n".join(dedup)

def ensure_container_solvents(text: str) -> str:
    _ADD_SOLVENT_LINE = re.compile(r"^Add\s+(\d+(?:\.\d+)?)\s*mL\s+([^\.,;]+)\.\s*$", re.I)

    _CONTAINMENT_RE = re.compile(
        r"\b(Add|Transfer|Mix)\b.+?\b(to|into|in)\s+(\d+(?:\.\d+)?)\s*mL\s+([^\.,;]+)\.?\s*$",
        re.I
    )


    def _norm_solv(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()

    blocks = text.split(DELIM)
    fixed = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        present = set()
        out = []
        for ln in lines:
            m_add = _ADD_SOLVENT_LINE.match(ln)
            if m_add:
                vol, solv = m_add.group(1), _norm_solv(m_add.group(2))
                present.add((vol, solv))
                out.append(ln); continue
            m = _CONTAINMENT_RE.search(ln)
            if m:
                vol = m.group(3)
                raw = _norm_solv(m.group(4))
                raw = re.sub(r"^(of\s+)", "", raw)
                solv_phrase = raw
                if (vol, solv_phrase) not in present:
                    out.append(_ensure_period(f"Add {vol} mL {solv_phrase}"))
                    present.add((vol, solv_phrase))
            out.append(ln)
        fixed.append("\n".join(out))
    return f"\n{DELIM}\n".join(s.strip() for s in fixed if s.strip())


_HEAT_GENERIC = re.compile(r"^Heat\b(?!\s*\d)", re.I)
_ADD_VOL_WATER = re.compile(r"^Add\s+(\d+(?:\.\d+)?)\s*mL\s+water\.\s*$", re.I)
_DISSOLVE_IN_VOL_WATER = re.compile(
    r"^Dissolve\s+.+?\s+in\s+(\d+(?:\.\d+)?)\s*mL\s+water\.\s*$", re.I)

_SEPARATE_HINT = re.compile(r"\b(let it|allow it)\s+(separate|settle)\b", re.I)

def _dedupe_procedure(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(ln)
    lines = out; out = []

    prev_specific_heat = False
    for ln in lines:
        if _HEAT_WATER_RE.match(ln):
            prev_specific_heat = True
            out.append(ln)
            continue
        if prev_specific_heat and _HEAT_GENERIC.match(ln):
            prev_specific_heat = False
            continue
        prev_specific_heat = False
        out.append(ln)
    lines = out; out = []

    i = 0
    while i < len(lines):
        m_add = _ADD_VOL_WATER.match(lines[i])
        if m_add and i + 1 < len(lines):
            vol = m_add.group(1)
            m_dis = _DISSOLVE_IN_VOL_WATER.match(lines[i + 1])
            if m_dis and m_dis.group(1) == vol:
                out.append(lines[i + 1])
                i += 2
                continue
        out.append(lines[i]); i += 1
    lines = out; out = []

    for ln in lines:
        if ln.lower().startswith("stir") and _SEPARATE_HINT.search(ln):
            out.append("Stir.")
            out.append("Separate the layers.")
        else:
            out.append(ln)

    return out



### Preprocessing Pipeline

class colors:
    YELLOW = '\033[93m'; GREEN = '\033[92m'; RESET = '\033[0m'; RED = '\033[91m'

class PreprocessingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.prompts = {"count": None, "rewrite": None}
        self.split_mode = self.config.get("split_mode", "delimiter_only")
        self.product_map = self.config.get("product_map", {})
        self.only_files_in_map = self.config.get("only_files_in_map", False)

    def setup(self):
        self.client = genai.Client(api_key=self.config['api_key'])
        self.model_id = self.config.get("model_name", "gemini-3.1-flash-lite-preview")

        pf = self.config.get('prompt_files', {})
        
        self.prompts["count"] = self._get_prompt_from_file(pf.get('count'))
        self.prompts["rewrite"] = self._get_prompt_from_file(pf.get('rewrite'))

        if not self.prompts["count"] or not self.prompts["rewrite"]:
            print(f"{colors.RED}CRITICAL ERROR: Preprocessing prompts missing from disk.{colors.RESET}")
            print(f"{colors.YELLOW}Ensure prompts exist in your Workdir/prompts or package internal folder.{colors.RESET}")
            return False

        out = Path(self.config['output_folder'])
        if out.exists():
            shutil = __import__("shutil")
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
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

    def _call_agent(self, prompt, is_json=False, temperature=0):
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json" if is_json else "text/plain"
            }
        )
        return (resp.text or "").strip()

    def _validated_products(self, source_text, resp_json):
        seen = set(); products = []
        for name in resp_json.get("products", []):
            if isinstance(name, str):
                nm = name.strip()
                if nm and not _ANION_ONLY.match(nm) and nm in source_text and nm not in seen:
                    seen.add(nm); products.append(nm)
        return products[:6]

    def _strip_meta(self, text: str) -> str:
        return "\n".join([ln for ln in text.splitlines() if ln.strip() and not _META_PAT.search(ln)]).strip()

    def _split_and_save(self, content, base_filename, max_parts=None):
        out_dir = Path(self.config['output_folder']); out_dir.mkdir(parents=True, exist_ok=True)
        text = content.strip()

        if self.split_mode == "delimiter_only":
            if DELIM in text:
                parts = [p.strip() for p in text.split(DELIM) if p.strip()]
                if max_parts: parts = parts[:max_parts]
                if not parts:
                    (out_dir / base_filename).write_text(text, encoding="utf-8"); return
                stem = Path(base_filename).stem
                width = max(2, len(str(len(parts))))
                for i, part in enumerate(parts, 1):
                    (out_dir / f"{stem}-{str(i).zfill(width)}.txt").write_text(part, encoding="utf-8")
                return
            (out_dir / base_filename).write_text(text, encoding="utf-8"); return

        if DELIM in text:
            parts = [p.strip() for p in text.split(DELIM) if p.strip()]
            if max_parts is not None and len(parts) > max_parts:
                parts = parts[:max_parts]
            if not parts:
                (out_dir / base_filename).write_text(text, encoding="utf-8"); return
            stem = Path(base_filename).stem
            width = max(2, len(str(len(parts))))
            for i, part in enumerate(parts, 1):
                (out_dir / f"{stem}-{str(i).zfill(width)}.txt").write_text(part, encoding="utf-8")
        else:
            (out_dir / base_filename).write_text(text, encoding="utf-8")

    def _targets_for_file(self, file_stem: str) -> List[str] | None:
        if file_stem in self.product_map:
            return self.product_map[file_stem]
        if "-" in file_stem:
            base = file_stem.split("-", 1)[0]
            if base in self.product_map:
                return self.product_map[base]
        return None

    def process_file(self, filepath):
        filename = os.path.basename(filepath)
        stem = Path(filename).stem
        try:
            if self.only_files_in_map and not self._targets_for_file(stem):
                return

            source_text = Path(filepath).read_text(encoding="utf-8")

            targets = self._targets_for_file(stem)
            if not targets:
                targets = []
                count_prompt = self.prompts['count'].replace("{{source_text_here}}", source_text)
                count_raw = self._call_agent(count_prompt, is_json=True, temperature=0)
                try:
                    count_json = json.loads(count_raw)
                except Exception:
                    count_json = {"products": []}
                targets = self._validated_products(source_text, count_json)

            proc_count = max(1, len(targets)) if targets else 1
            if not targets:
                targets = ["__DUMMY__"]

            rewrite_prompt = (
                self.prompts['rewrite']
                .replace("{{source_text_here}}", source_text)
                .replace("{{product_names}}", json.dumps(targets, ensure_ascii=False))
            )
            cleaned_text = self._call_agent(rewrite_prompt, is_json=False, temperature=0)
            if cleaned_text:
                cleaned_text = self._strip_meta(cleaned_text)
                cleaned_text = sanitize_action_lines(cleaned_text)
                cleaned_text = ensure_container_solvents(cleaned_text)
                cleaned_text = normalize_actions(cleaned_text)
                cleaned_lines = [ln for ln in cleaned_text.splitlines() if ln.strip() and ln.strip() != DELIM]
                cleaned_lines = _dedupe_procedure(cleaned_lines)
                cleaned_text = "\n".join(cleaned_lines)
                max_parts = 1 if proc_count >= 1 else None
                self._split_and_save(cleaned_text, filename, max_parts=max_parts)

        except Exception as e:
            tqdm.write(f"  \033[91mPreprocessing error for {filename} {e}\033[0m")

    def run(self):
        print(f"\033[92m  >>>  Starting Phase 0: Preprocessing\033[0m")
        if not self.setup():
            return
        src = Path(self.config['source_folder'])

        files = sorted([str(p) for p in src.glob("SYN*.txt")])

        if self.only_files_in_map and self.product_map:
            allow = set(self.product_map.keys())
            def allowed(p: Path) -> bool:
                st = p.stem
                if st in allow: return True
                if "-" in st and st.split("-", 1)[0] in allow: return True
                return False
            files = [str(p) for p in map(Path, files) if allowed(Path(p))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['max_workers']) as ex:
            list(tqdm(ex.map(self.process_file, files), total=len(files), desc="Cleaning Text Files"))
        print(f"Preprocessing phase complete. Cleaned texts saved to '{self.config['output_folder']}'.")

