# Digitising Polyoxometalate Syntheses — Deposited Corpus

This is the canonical, deposited dataset accompanying the manuscript *"Digitising Polyoxometalate Syntheses"* (Kondinski et al.). It is the exact corpus that every count, table, and figure in the manuscript and its Supporting Information is derived from.

## Files

- **`procedures_405.json`** — the structured corpus (v2, corrected — see Version History below): 405 polyoxometalate synthesis procedures, each with `procedure_information` (ID), `reagents`, and `procedure_steps` (ordered unit operations with reagent usage, time, temperature, each carrying source-text line provenance). Extracted via a 4-phase LLM pipeline (Text Normalisation, Reagent Extraction, Procedure Structuring, Data Assembly) from historical/literature synthesis texts, then independently cross-checked and repaired; extraction prompts are given in the manuscript's Supporting Information.
- **`v1_archived/`** — the original, uncorrected extraction, kept for transparency only. Not what this manuscript describes.
- **`precursor_classification.json`** — the starting addenda-metal source classification for each procedure (named salt, pre-formed precursor, non-classical addenda system, etc.), verified directly from the reagent data in `procedures_405.json`. This is the source for Table S5 in the Supporting Information.
- **`target_compounds_405.json`** — the target/product compound formula for each procedure ID, as reported in the original source text.
- **`procedure_metadata_doi.json`** — the literature DOI(s) for each procedure, matched by the shared procedure numbering to a paper-by-paper bibliography compiled during the original literature download; also carries curated POM-core element lists retained from an earlier internal cross-reference where available (partial coverage — most records fall back to elements derived from the target formula instead).
- **`procedures_schema.json`** — a formal JSON Schema (draft 2020-12) for `procedures_405.json`, validated against all 405 records with zero errors. Step-level fields beyond a small required core are intentionally left open (`additionalProperties: true`), reflecting that the extraction pipeline records whatever conditions a given source text actually states rather than forcing every step to declare every possible field.
- **`LICENSE`** — CC BY 4.0, full text.

## Provenance and known limitations

- All 405 procedure IDs (`SYN1`–`SYN582`, non-contiguous) are consistent across all four files.
- One procedure (`SYN118`) has no reagents recorded in the source text at all; one (`SYN552`)'s addenda source could only be resolved as a pre-formed precursor via out-of-band verification against the primary publisher record, not from the extracted reagent text alone.
- DOI coverage is 405/405 (100%), but curated POM-core element coverage in `procedure_metadata_doi.json` is partial (most records fall back to elements derived from the target formula instead).

## Version history — how this corpus was corrected

This is **v2** of the deposited corpus. The original extraction (v2, archived in `v1_archived/`) recorded each reagent's amount/concentration once per procedure as an aggregated total rather than per step, so a step referencing that reagent did not necessarily carry its own correct sub-quantity, temperature, or timing.

This was caught, not assumed: every one of the 405 procedures was used to regenerate a free-text synthesis paragraph, which was then automatically cross-checked against both the structured JSON and the original extracted source text (reagent identities/quantities, temperature/time/pH/atmosphere, operation order, work-up conditions). That check gave 69 exact-match passes, 181 ambiguous-but-uncontradicted, and 155 procedures with a genuine discrepancy — predominantly the aggregated-quantity issue above, not errors in the underlying extraction itself.

Those 155 (plus the ambiguous 181) were then repaired at the JSON level — attaching each reagent's correct per-step quantity, temperature, and timing directly from the original source text with line-level provenance — and independently re-validated against the same source text until no contradictions remained (0 FAIL; 106 residual flags are literal unit-normalisation differences only, e.g. "2 h" vs. "120 min", not factual errors). Two small schema defects introduced during this repair (a misplaced `target` field in 3 steps; 51 temperature/time values recorded as bare strings instead of the standard `{value, unit}` structure) were found and corrected before this corpus was promoted to canonical.

Promoting this repair changed more than isolated reagent quantities: the addenda-source classification (Table S5) picked up one additional resolved procedure (`SYN552`, 152→153), four competency-question counts shifted (Supporting Information, Section S2), and — because the ML clustering in the manuscript is computed directly from per-step operation sequences, step counts, and temperatures — the clustering was re-run on this corrected corpus rather than left pinned to the superseded v1. The manuscript and Supporting Information reflect this v2 corpus throughout; `v1_archived/` is kept only for transparency and is not what the manuscript's counts, tables, or figures describe.

## Citation

If you use this dataset, please cite the manuscript (full citation in the deposited record's metadata / the published article once assigned a DOI).

## License

CC BY 4.0 -- see `LICENSE` in this deposit for the full text. This matches the manuscript's own open-access license.

## A second, small correction made after initial deposit

Two procedures (`SYN162`, `SYN495`) had their per-step operation label recorded under the key `operation` instead of the schema's `unit_operation` (21 steps total); this meant their unit operations were silently excluded from the unit-operation frequency statistics (Table S3) and the ML clustering features. This was caught by validating the corpus against `procedures_schema.json` above, fixed by renaming the key, and the affected downstream statistics, clustering, and figures were regenerated. The corpus in this deposit already reflects this fix.
