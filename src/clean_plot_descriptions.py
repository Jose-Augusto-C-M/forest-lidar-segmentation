
#!/usr/bin/env python3
import argparse
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import yaml

def _read_table(path: str, sheet: Optional[str]=None) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet)
    elif lower.endswith(".parquet"):
        return pd.read_parquet(path)
    elif lower.endswith(".feather"):
        return pd.read_feather(path)
    else:
        # Let pandas sniff separators; fall back to comma
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")

def _write_table(df: pd.DataFrame, path: str):
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="cleaned")
    elif lower.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif lower.endswith(".feather"):
        df.to_feather(path)
    else:
        df.to_csv(path, index=False)

def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    # basic normalization
    s = str(s)
    s = s.replace("\u00a0", " ")  # non‑breaking space
    s = re.sub(r"[,\u2013\u2014]", " ", s)  # commas/dashes to space
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def apply_rules(text: str, rules: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """
    Returns (cleaned_value, matched_rule_name). If nothing matches, returns (text, None).
    Rules are applied in order.
    """
    for rule in rules:
        rx = rule.get("pattern")
        if not rx:
            continue
        if re.search(rx, text):
            repl = rule.get("replace_with", None)
            if repl is not None:
                return (repl, rule.get("name") or rx)
            else:
                if rule.get("normalize_to"):
                    return (rule["normalize_to"], rule.get("name") or rx)
                return (text, rule.get("name") or rx)
    return (text, None)

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean, normalize, and optionally filter a text column using regex rules (YAML)."
    )
    p.add_argument("input", help="Input table file (.csv, .xlsx, .parquet, .feather)")
    p.add_argument("-o","--output", help="Output file (guessed from extension). Default: <input>.clean.csv")
    p.add_argument("-c","--column", default="Description", help="Column to clean. Default: Description")
    p.add_argument("--sheet", default=None, help="Excel sheet name if reading .xlsx/.xls")
    p.add_argument("--rules", default=None, help="YAML file with rules. If omitted, built-in defaults are used.")
    p.add_argument("--drop-unmatched", action="store_true", help="Drop rows that match no rule.")
    p.add_argument("--keep", action="append", default=[], help="Regex of rows to KEEP after cleaning (apply to cleaned value). May be given multiple times.")
    p.add_argument("--drop", action="append", default=[], help="Regex of rows to DROP after cleaning (apply to cleaned value). May be given multiple times.")
    p.add_argument("--dry-run", action="store_true", help="Do not write output; just print a summary.")
    p.add_argument("--preview", type=int, default=0, help="Print first N rows of original + cleaned column.")
    return p

def load_rules(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        text = '''
- name: plot centre (numbered)
  pattern: "\\bplot\\s*centre\\s*\\d+\\b"
  replace_with: "plot_centre"

- name: centre tree
  pattern: "\\bcentre tree\\b"
  replace_with: "plot_centre"

- name: dead pine
  pattern: "\\bdead\\s+pine(\\b|\\s|\\d)"
  replace_with: "dead_pine"

- name: dead spruce standing
  pattern: "\\bstanding\\s+dead\\s+spruce"
  replace_with: "dead_spruce_standing"

- name: pine
  pattern: "\\bpine\\s*\\d+\\b"
  replace_with: "pine"

- name: spruce
  pattern: "\\bspruce\\s*\\d+\\b"
  replace_with: "spruce"

- name: birch
  pattern: "\\bbirch\\s*\\d+\\b"
  replace_with: "birch"

- name: lying pine
  pattern: "\\blying\\s+pine"
  replace_with: "lying_pine"

- name: lying spruce
  pattern: "\\blying\\s+spruce"
  replace_with: "lying_spruce"

- name: decomposed
  pattern: "\\bdecomposed\\b"
  replace_with: "decomposed"

- name: leaning (any direction/deg)
  pattern: "\\blean(?:ing)?(\\s+[a-z]+)?(\\s+\\d+\\s*deg)?\\b"
  replace_with: "leaning"

- name: two stems
  pattern: "two stems"
  replace_with: "two_stems"

- name: two tops
  pattern: "two tops"
  replace_with: "two_tops"

- name: bark falling
  pattern: "bark is falling off"
  replace_with: "bark_falling_off"

- name: panorama south
  pattern: "panorama south"
  replace_with: "panorama_south"

- name: only living pines
  pattern: "only living pines standing"
  replace_with: "only_living_pines"

- name: living pines few birches
  pattern: "living pines with a few birches"
  replace_with: "living_pines_with_few_birches"

- name: mixed age
  pattern: "\\bmixed age\\b"
  replace_with: "mixed_age"

- name: lonely letters d
  pattern: "^d(\\b|\\s|\\d)"
  replace_with: "dead_note"

- name: lying generic
  pattern: "^lying(\\b|\\s).*"
  replace_with: "lying"

- name: pure spruce
  pattern: "100% spruce"
  replace_with: "spruce_100pct"
'''
        return yaml.safe_load(text)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def filter_series(cleaned: pd.Series, keep: List[str], drop: List[str]) -> pd.Series:
    mask = pd.Series([True]*len(cleaned), index=cleaned.index)
    for rx in drop:
        mask &= ~cleaned.str.contains(rx, regex=True, na=False)
    if keep:
        keepmask = pd.Series([False]*len(cleaned), index=cleaned.index)
        for rx in keep:
            keepmask |= cleaned.str.contains(rx, regex=True, na=False)
        mask &= keepmask
    return mask

def main(argv: Optional[List[str]] = None):
    args = make_argparser().parse_args(argv)
    df = _read_table(args.input, sheet=args.sheet)
    if args.column not in df.columns:
        sys.exit(f"ERROR: Column '{args.column}' not found. Available: {list(df.columns)}")

    rules = load_rules(args.rules)

    original = df[args.column].astype(object).map(normalize_text)
    cleaned_values = []
    matched_rules = []
    for s in original:
        cleaned, rule_name = apply_rules(s, rules)
        cleaned_values.append(cleaned)
        matched_rules.append(rule_name)

    df[args.column + "_clean"] = cleaned_values
    df[args.column + "_rule"] = matched_rules

    if args.drop_unmatched:
        df = df[~df[args.column + "_rule"].isna()]

    if args.keep or args.drop:
        mask = filter_series(df[args.column + "_clean"], args.keep, args.drop)
        df = df[mask]

    stats = (
        df[args.column + "_clean"]
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )

    if args.preview:
        print(df[[args.column, args.column + "_clean", args.column + "_rule"]].head(args.preview).to_string(index=False))

    if args.dry_run:
        print("\n=== Summary (dry run) ===")
        print(stats.to_string(index=False))
        return

    out = args.output or (args.input.rsplit(".", 1)[0] + ".clean.csv")
    _write_table(df, out)

    stats_path = out.rsplit(".", 1)[0] + ".summary.csv"
    stats.to_csv(stats_path, index=False)

    print(f"Wrote cleaned table to: {out}")
    print(f"Wrote summary counts to: {stats_path}")

if __name__ == "__main__":
    main()
