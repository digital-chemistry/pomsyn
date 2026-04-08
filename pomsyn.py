from __future__ import annotations

import csv
import json
import logging
import re

import os
os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker'

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from google import genai
import sys

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="resource_tracker")

from components.text_preprocessing import PreprocessingPipeline
from components.reagent_extraction import ExtractionPipeline
from components.procedure_structuring import StructuringPipeline
from components.data_assembly import AssemblyPipeline


### Config

try:
    API_KEY_FILE = "google_api_key.txt"
    MAX_WORKERS = 5

    DEFAULT_MODEL           = "gemini-3.1-flash-lite-preview"
    MODEL_INPUT             = input(f"\nSet the Model to use (Press Enter for default [{DEFAULT_MODEL}]): ").strip()
    MODEL                   = MODEL_INPUT if MODEL_INPUT else DEFAULT_MODEL

    WORKDIR                 = input("Set Path to Working Directory: ").strip()
    if not WORKDIR:
        print("\n\033[91mERROR: Working Directory cannot be empty.\033[0m")
        print("")
        os._exit(1)
    BASE_DIR                = Path(WORKDIR)

    PROMPTS_DIR             = BASE_DIR / "prompts"
    INTERNAL_PROMPTS        = Path(__file__).parent / "prompts"

    SOURCE_TEXTS_FOLDER     = BASE_DIR / "text_input"
    PHASE0_FOLDER           = BASE_DIR / "phase0_preprocessed_texts"
    PHASE1_FOLDER           = BASE_DIR / "phase1_manifests"
    PHASE2_FOLDER           = BASE_DIR / "phase2_steps_markdown"
    FINAL_OUTPUT_FILENAME   = BASE_DIR / "procedures.json"

    MAP_CSV                 = BASE_DIR / "input_definition.csv"

    RUN_STAMP               = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUNS_ROOT               = BASE_DIR / "runs" 
    THIS_RUN_DIR            = RUNS_ROOT / RUN_STAMP
    LATEST_DIR              = RUNS_ROOT / "latest"
    LOG_DIR                 = RUNS_ROOT / "logs"

    KEEP_LAST_RUNS = 5

except KeyboardInterrupt:
    print("\n\n\033[93m[!] Stopped by user during setup. Exiting...\033[0m")
    print("")
    sys.exit(0)


### Logging

def setup_logging(run_stamp: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"run_{run_stamp}.log"

    ### Root logger
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.INFO)

    ### File
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)

    ### Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(ch)

    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return log_file


### Utils

def load_api_key(path: Path) -> str:
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"API key file is empty at {path}")
    return key

def clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)

def prune_old_runs(root: Path, keep: int) -> None:
    if not root.exists():
        return
    runs = sorted([d for d in root.iterdir() if d.is_dir() and d.name != "latest"])
    old = runs[:-keep]
    for d in old:
        shutil.rmtree(d, ignore_errors=True)

def count_files(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for _ in p.rglob("*") if _.is_file())

def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_map_csv(path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = (row.get("file_id") or row.get("file") or "").strip()
            product = (row.get("product") or row.get("target") or "").strip()
            if not file_id or not product:
                continue
            mapping.setdefault(file_id, []).append(product)
    return mapping

def sort_key_from_procedure_id(proc) -> List[int]:
    pid = proc.get("procedure_information", {}).get("id", "proc-0")
    nums = [int(n) for n in re.findall(r"\d+", pid)]
    return nums if nums else [0]


### Main

def main_entry():
    if not BASE_DIR.exists() or not BASE_DIR.is_dir():
        print(f"\n\033[91mCRITICAL ERROR: The directory '{WORKDIR}' was not found.\033[0m")
        print("")
        sys.exit(1)

    log_file = setup_logging(RUN_STAMP)
    
    try:
        api_key = load_api_key(BASE_DIR / API_KEY_FILE)
        logging.info("API Key loaded successfully.")
        logging.info("Run stamp: %s", RUN_STAMP)
        logging.info("Log file  : %s", log_file)
        logging.info("")
        
        client = genai.Client(api_key=api_key)
        logging.info("Verifying model availability: %s", MODEL)
        
        available_models = [m.name for m in client.models.list()]
        target = MODEL if MODEL.startswith("models/") else f"models/{MODEL}"
        
        if target not in available_models:
            logging.error("\033[91mCRITICAL ERROR: Model '%s' not found or not authorized for this API key.", MODEL)
            logging.info("") 
            
            logging.info("Available models: %s", ", ".join([m.replace("models/", "") for m in available_models]))
            
            logging.info("") 
            sys.exit(1)
        
        logging.info("Model verified and ready.\n")

    except Exception as e:
        logging.error("Initialization failed: %s", e)
        sys.exit(1)

    try:
        ### per-run dir structure
        THIS_RUN_DIR.mkdir(parents=True, exist_ok=True)
        d_preprocess            = THIS_RUN_DIR / "00_text_preprocessing"
        d_reagent_extraction    = THIS_RUN_DIR / "01_reagent_extraction"
        d_proc_struc            = THIS_RUN_DIR / "02_procedure_structuring"
        d_data_assembly         = THIS_RUN_DIR / "03_data_assembly"
        d_final                 = THIS_RUN_DIR / "final"

        ### Clean phase working folders
        for p in (PHASE0_FOLDER, PHASE1_FOLDER, PHASE2_FOLDER):
            clean_dir(p)

        ### input_definition.csv (optional)
        product_map = read_map_csv(MAP_CSV)
        if product_map:
            total_targets = sum(len(v) for v in product_map.values())
            logging.info("Loaded %d target entries for %d file ids from '%s'.",
                         total_targets, len(product_map), MAP_CSV.name)
        else:
            logging.info("No input_definition.csv found or empty; Cleanup will fallback to product detection.")

        ### Phase 0
        preprocessing_config = {
            "api_key": api_key,
            "model_name": MODEL,
            "source_folder": str(SOURCE_TEXTS_FOLDER),
            "output_folder": str(PHASE0_FOLDER),
            "prompt_files": {
                "count": [
                    str(PROMPTS_DIR / "prompt_preprocessing_count.txt"),
                    str(INTERNAL_PROMPTS / "prompt_preprocessing_count.txt")
                ],
                "rewrite": [
                    str(PROMPTS_DIR / "prompt_preprocessing_rewrite.txt"),
                    str(INTERNAL_PROMPTS / "prompt_preprocessing_rewrite.txt")
                ],
            },
            "max_workers": MAX_WORKERS,
            "split_mode": "delimiter_only",
            "product_map": product_map,
            "only_files_in_map": bool(product_map)
        }
        PreprocessingPipeline(preprocessing_config).run()
        copy_tree(PHASE0_FOLDER, d_preprocess)
        logging.info("Phase 0 done. %d files.", count_files(PHASE0_FOLDER))
        logging.info("")

        ### Phase 1
        reagent_extraction_config = {
            "api_key": api_key,
            "model_name": MODEL,
            "source_folder": str(PHASE0_FOLDER),
            "output_folder": str(PHASE1_FOLDER),
            "prompt_file": [str(PROMPTS_DIR / "prompt_reagent_extraction.txt"), str(INTERNAL_PROMPTS / "prompt_reagent_extraction.txt")],
            "database_file": "chemical_database.json",
            "max_workers": MAX_WORKERS,
        }
        ExtractionPipeline(reagent_extraction_config).run()
        copy_tree(PHASE1_FOLDER, d_reagent_extraction)
        db_path = Path("chemical_database.json")
        if db_path.exists():
            shutil.copy2(db_path, d_reagent_extraction / "chemical_database.json")
        logging.info("Phase 1 done. %d files.", count_files(PHASE1_FOLDER))
        logging.info("")

        ### Phase 2
        proc_struc_config = {
            "api_key": api_key,
            "model_name": MODEL,
            "source_folder": str(PHASE0_FOLDER),
            "manifests_folder": str(PHASE1_FOLDER),
            "output_folder": str(PHASE2_FOLDER),
            "prompt_file": [str(PROMPTS_DIR / "prompt_procedure_structuring.txt"), str(INTERNAL_PROMPTS / "prompt_procedure_structuring.txt")],
            "max_workers": MAX_WORKERS,
        }
        StructuringPipeline(proc_struc_config).run()
        copy_tree(PHASE2_FOLDER, d_proc_struc)
        logging.info("Phase 2 done. %d files.", count_files(PHASE2_FOLDER))
        logging.info("")

        ### Phase 3
        data_assembly_config = {
            "api_key": api_key,
            "model_name": MODEL,
            "manifests_folder": str(PHASE1_FOLDER),
            "markdown_folder": str(PHASE2_FOLDER),
            "prompt_file": [str(PROMPTS_DIR / "prompt_data_assembly.txt"), str(INTERNAL_PROMPTS / "prompt_data_assembly.txt")],
            "max_workers": MAX_WORKERS,
        }
        final_procedures = AssemblyPipeline(data_assembly_config).run()
        copy_tree(PHASE2_FOLDER, d_data_assembly)

        ### Final save
        if final_procedures:
            final_procedures.sort(key=sort_key_from_procedure_id)
            out = {"procedures": final_procedures}

            Path(FINAL_OUTPUT_FILENAME).write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            write_json(d_final / "procedures.json", out)
            logging.info("Final dataset with %d procedures saved to '%s'.",
                         len(final_procedures), FINAL_OUTPUT_FILENAME)
        else:
            logging.warning("No procedures assembled. Nothing to write.")

        ### Summaries & manifests
        manifest = {
            "run_stamp": RUN_STAMP,
            "source_folder": str(SOURCE_TEXTS_FOLDER.resolve()),
            "phase0_folder": str(PHASE0_FOLDER.resolve()),
            "phase1_folder": str(PHASE1_FOLDER.resolve()),
            "phase2_folder": str(PHASE2_FOLDER.resolve()),
            "final_output_root": str(d_final.resolve()),
            "final_output_legacy": str(Path(FINAL_OUTPUT_FILENAME).resolve()),
            "log_file": str(Path(log_file).resolve()),
            "counts": {
                "source_files": count_files(SOURCE_TEXTS_FOLDER),
                "phase0_files": count_files(PHASE0_FOLDER),
                "phase1_files": count_files(PHASE1_FOLDER),
                "phase2_files": count_files(PHASE2_FOLDER),
                "final_procedures": len(final_procedures or []),
            },
            "map_csv_used": MAP_CSV.exists(),
        }
        write_json(THIS_RUN_DIR / "run_manifest.json", manifest)
        summary_lines = [
            f"Run: {RUN_STAMP}",
            f"Log: {log_file}",
            f"Source files              : {manifest['counts']['source_files']}",
            f"Preprocessing artifacts   : {manifest['counts']['phase0_files']}",
            f"Extraction artifacts      : {manifest['counts']['phase1_files']}",
            f"Structuring artifacts     : {manifest['counts']['phase2_files']}",
            f"Final procedures          : {manifest['counts']['final_procedures']}",
            f"Final (per-run)           : {d_final / 'procedures.json'}",
            f"Final (legacy)            : {FINAL_OUTPUT_FILENAME}",
        ]
        write_text(THIS_RUN_DIR / "summary.txt", "\n".join(summary_lines))

        if LATEST_DIR.exists():
            shutil.rmtree(LATEST_DIR)
        copy_tree(THIS_RUN_DIR, LATEST_DIR)

        ### Prune old runs
        prune_old_runs(RUNS_ROOT, KEEP_LAST_RUNS)

        logging.info("")
        logging.info("\033[92m  >>>  FULL PIPELINE COMPLETE\033[0m")
        logging.info("Artifacts: %s", THIS_RUN_DIR.resolve())
        logging.info("Latest   : %s", LATEST_DIR.resolve())
        logging.info("Log file : %s", Path(log_file).resolve())
        logging.info("")

    except Exception as e:
        logging.exception("A critical error occurred in the main pipeline: %s", e)
    
    except KeyboardInterrupt:
        print("\n\033[93m[!] Execution stopped by user. Closing immediately...\033[0m")
        print("")
        os._exit(0)


if __name__ == "__main__":
    main_entry()
