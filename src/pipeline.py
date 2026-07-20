"""Build a documented, non-imputed analysis panel from the original merged CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.config import AUDIT_DIR, PROCESSED_DIR, ensure_output_directories
from src.construction import construct_analysis_panel, treatment_log
from src.data_quality import missingness_report, panel_coverage, validate_unique_panel_keys

def build_panel(input_path: Path, output_name: str="analysis_panel.csv") -> Path:
    ensure_output_directories()
    panel = construct_analysis_panel(pd.read_csv(input_path))
    panel = panel.dropna(subset=["country_code", "year"])
    validate_unique_panel_keys(panel)
    missingness_report(panel).to_csv(AUDIT_DIR / "missingness_by_variable.csv", index=False)
    panel_coverage(panel).to_csv(AUDIT_DIR / "panel_coverage_by_country.csv", index=False)
    treatment_log().to_csv(AUDIT_DIR / "variable_treatment_log.csv", index=False)
    output = PROCESSED_DIR / output_name
    panel.to_csv(output, index=False)
    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a non-imputed, auditable analysis panel.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-name", default="analysis_panel.csv")
    args = parser.parse_args()
    print(f"Wrote {build_panel(args.input, args.output_name)}")
if __name__ == "__main__": main()
