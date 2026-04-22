"""
Table processing utilities for extracting rate data from PDF tables.

This module handles different table structures:
1. Tables with "Current Plan" / "Renewal Plan" columns
2. Tables with direct tier columns (Employee Only, Employee + Spouse, etc.)
3. Summary/composite tables (which should be skipped)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Constants
TIER_ABBREVIATIONS = {
    "EO": "employee_only",
    "ES": "employee_spouse",
    "EC": "employee_child",
    "EF": "employee_family",
}

CATEGORY_KEYWORDS = ["platinum", "gold", "silver", "bronze", "hsa", "ppo", "network", "expanded bronze"]

SKIP_KEYWORDS = ["total", "premium", "sum", "enrolled count", "exposure"]


def normalize_text(text: str) -> str:
    """Normalize whitespace in text (handles newlines, multiple spaces, etc.)."""
    return re.sub(r'\s+', ' ', str(text)).strip().lower() if text else ""


def is_summary_table(table: List[List]) -> bool:
    """
    Check if a table is a summary/composite table that should be skipped.
    
    Summary tables have "Total Monthly Medical Cost" but no tier breakdown.
    """
    table_str = " ".join(" ".join(str(cell) if cell else "" for cell in row) for row in table).lower()
    
    # Require the specific summary phrase; loose combos like ("total" + "cost") are too broad
    # and incorrectly match tables that have a "Total" footer row but still contain tier rates.
    has_total_cost = (
        "total monthly medical cost" in table_str
        or ("composite rate" in table_str and "total" in table_str)
        or ("premium" in table_str and "total" in table_str and "composite" in table_str)
    )

    return has_total_cost


def has_age_band_indicators(table: List[List]) -> bool:
    """
    Check if a table has age-band indicators (age ranges, age-based rates).
    
    Age-banded tables have patterns like:
    - Age ranges: "<20", "20-24", "25-29", "30-34", etc.
    - Single ages: "<15", "15", "16", "17", ..., "64+"
    - Keywords: "age band", "age-based", "aca age"
    """
    # Pattern for age ranges: <20, 20-24, 25-29, 30-34, 80+, etc.
    age_range_pattern = re.compile(
        r'<[0-9]+|[0-9]+\s*-\s*[0-9]+|[0-9]+\+',
        re.IGNORECASE
    )
    
    age_keywords = ["age band", "age-based", "aca age", "age rating"]
    
    table_str = " ".join(" ".join(str(cell) if cell else "" for cell in row) for row in table).lower()
    
    # Check for age keywords
    if any(keyword in table_str for keyword in age_keywords):
        return True
    
    # Check for age range patterns in cells
    for row in table:
        if not row:
            continue
        for cell in row:
            if not cell:
                continue
            cell_str = str(cell)
            # Check for age range patterns
            if age_range_pattern.search(cell_str):
                # Additional validation: make sure it looks like an age range, not just a number
                # Age ranges are typically <20, 20-24, 25-29, 30-34, etc. or single digits/teens
                if re.search(r'<[0-9]{1,2}|[0-9]{1,2}\s*-\s*[0-9]{1,2}|[0-9]{1,2}\+', cell_str):
                    return True
    
    return False


def has_benefit_type_rows(table: List[List]) -> bool:
    """
    Check if a table has benefit type rows (Dental, Vision, etc.) instead of tier rows.
    
    This indicates a plan comparison table where columns = plans, rows = benefit types,
    not a tier rate table where rows = tier types (Employee Only, etc.) or age bands.
    
    IMPORTANT: Age-banded tables should NOT be skipped, so we exclude them here.
    """
    # First check if this is an age-banded table - if so, don't skip it
    if has_age_band_indicators(table):
        return False
    
    # "medical" is intentionally excluded: it appears as a plan type label in rate tables
    # and causes false positives on valid rate tables for medical proposals.
    benefit_keywords = ["dental", "vision", "life", "disability", "admin fee"]
    tier_keywords = ["employee only", "employee + spouse", "employee + child", "employee + family", 
                     "eo", "es", "ec", "ef"]
    
    has_benefit_types = False
    has_tier_types = False
    
    for row in table:
        if not row:
            continue
        row_str = normalize_text(" ".join(str(cell) if cell else "" for cell in row))
        
        # Check for benefit types in first column (common pattern)
        first_cell = normalize_text(str(row[0]) if row[0] else "")
        if any(keyword in first_cell for keyword in benefit_keywords):
            has_benefit_types = True
        if any(keyword in first_cell for keyword in tier_keywords):
            has_tier_types = True
        
        # Also check entire row
        if any(keyword in row_str for keyword in benefit_keywords):
            has_benefit_types = True
        if any(keyword in row_str for keyword in tier_keywords):
            has_tier_types = True
    
    # If we have benefit types but no tier types, this is likely a benefit comparison table
    return has_benefit_types and not has_tier_types


def has_tier_breakdown(table: List[List]) -> bool:
    """
    Check if a table has tier breakdown indicators (EO, ES, EC, EF or full tier names).
    """
    tier_pattern = re.compile(
        r'\b(EO|ES|EC|EF|employee[_\s\n]+only|employee[_\s\n]+\+?[_\s\n]*spouse|'
        r'employee[_\s\n]+\+?[_\s\n]*child|employee[_\s\n]+\+?[_\s\n]*family)\b',
        re.IGNORECASE
    )
    
    for row in table:
        normalized_row = normalize_text(" ".join(str(cell) if cell else "" for cell in row))
        if tier_pattern.search(normalized_row):
            return True
    
    return False


def find_table_columns(table: List[List]) -> Dict[str, Optional[int]]:
    """
    Find column indices for plan ID and rate columns in a table.
    
    Returns a dict with keys:
    - plan_id_col: Column index for Plan ID
    - renewal_col: Column index for Renewal Plan (if exists)
    - current_col: Column index for Current Plan (if exists)
    - employee_only_col: Column index for Employee Only (if exists)
    - employee_spouse_col: Column index for Employee + Spouse (if exists)
    - employee_child_col: Column index for Employee + Child (if exists)
    - employee_family_col: Column index for Employee + Family (if exists)
    - header_row_idx: Row index of the header row
    """
    result = {
        "plan_id_col": None,
        "renewal_col": None,
        "current_col": None,
        "employee_only_col": None,
        "employee_spouse_col": None,
        "employee_child_col": None,
        "employee_family_col": None,
        "header_row_idx": None,
    }
    
    for row_idx, row in enumerate(table):
        if not row or len(row) < 3:
            continue
        
        row_str = normalize_text(" ".join(str(cell) if cell else "" for cell in row))
        
        # Check for "Renewal Plan" / "Current Plan" structure
        if "renewal plan" in row_str or "current plan" in row_str:
            result["header_row_idx"] = row_idx
            for col_idx, cell in enumerate(row):
                cell_str = normalize_text(str(cell))
                if "plan id" in cell_str:
                    result["plan_id_col"] = col_idx
                elif "renewal plan" in cell_str:
                    result["renewal_col"] = col_idx
                elif "current plan" in cell_str:
                    result["current_col"] = col_idx
            break
        
        # Check for direct tier columns
        tier_keywords = ["employee only", "employee + spouse", "employee + child", "employee + family"]
        if any(keyword in row_str for keyword in tier_keywords):
            result["header_row_idx"] = row_idx
            logger.info(f"Found direct tier columns in row {row_idx}")
            
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                cell_str = normalize_text(str(cell))
                
                if "plan id" in cell_str:
                    result["plan_id_col"] = col_idx
                    logger.debug(f"  Found Plan ID column at index {col_idx}")
                elif "employee only" in cell_str or (
                    cell_str.startswith("employee") and "only" in cell_str and
                    "spouse" not in cell_str and "child" not in cell_str and "family" not in cell_str
                ):
                    result["employee_only_col"] = col_idx
                    logger.debug(f"  Found Employee Only column at index {col_idx}")
                elif "employee + spouse" in cell_str or ("employee" in cell_str and "spouse" in cell_str):
                    result["employee_spouse_col"] = col_idx
                    logger.debug(f"  Found Employee + Spouse column at index {col_idx}")
                elif "employee + child" in cell_str or (
                    "employee" in cell_str and "child" in cell_str and "family" not in cell_str
                ):
                    result["employee_child_col"] = col_idx
                    logger.debug(f"  Found Employee + Child column at index {col_idx}")
                elif "employee + family" in cell_str or ("employee" in cell_str and "family" in cell_str):
                    result["employee_family_col"] = col_idx
                    logger.debug(f"  Found Employee + Family column at index {col_idx}")
            break
    
    return result


def extract_rate_from_cell(cell: str, require_dollar: bool = True) -> Optional[float]:
    """
    Extract rate value from a cell string.
    
    Args:
        cell: Cell content as string
        require_dollar: If True, only extract values with "$" prefix. If False, extract any numeric value.
    """
    if not cell:
        return None
    
    # First try with $ prefix
    rate_match = re.search(r'\$([\d,]+\.?\d*)', str(cell))
    if rate_match:
        try:
            rate_value = rate_match.group(1).replace(",", "")
            return float(rate_value)
        except ValueError:
            pass
    
    # If require_dollar is False, try without $ prefix (for dental proposals that don't use $)
    if not require_dollar:
        # Match numeric values (with optional commas and decimals)
        match = re.search(r'^([\d,]+\.?\d*)$', str(cell).strip())
        if match:
            try:
                rate_value = match.group(1).replace(",", "")
                return float(rate_value)
            except ValueError:
                pass
    
    return None


def should_skip_row(row: List, plan_id_col: Optional[int], header_row_idx: Optional[int], row_idx: int) -> bool:
    """Check if a table row should be skipped (category headers, totals, etc.)."""
    if row_idx == header_row_idx or not row:
        return True
    
    # Check if plan ID exists and is valid
    if plan_id_col is not None and plan_id_col < len(row):
        plan_id_cell = str(row[plan_id_col]) if row[plan_id_col] else ""
        plan_id = plan_id_cell.strip()
        
        if not plan_id or "plan id" in plan_id.lower():
            return True
        
        # Skip category rows
        if plan_id.lower() in CATEGORY_KEYWORDS:
            return True
    
    # Check for skip keywords
    row_str = normalize_text(" ".join(str(cell) if cell else "" for cell in row))
    return any(keyword in row_str for keyword in SKIP_KEYWORDS)


def format_table_with_option_columns(
    table: List[List],
    columns: Dict[str, Optional[int]],
    page_num: int,
    table_idx: int,
) -> Optional[str]:
    """
    Format table that has option columns like "Current Plan" and/or "Renewal Plan".
    
    Returns formatted text string or None if no rates found.
    """
    renewal_col = columns["renewal_col"]
    current_col = columns["current_col"]
    header_row_idx = columns["header_row_idx"]
    
    if header_row_idx is None or (renewal_col is None and current_col is None):
        return None
    
    option_cols: List[Tuple[str, int]] = []
    if current_col is not None:
        option_cols.append(("Current", current_col))
    if renewal_col is not None:
        option_cols.append(("Renewal", renewal_col))

    table_text = f"\n--- Table {table_idx} (Page {page_num}) - PLAN OPTIONS ---\n"

    # Try to find plan_id from a row containing "Plan ID"
    plan_id: Optional[str] = None
    for row in table:
        row_str = " ".join(str(cell) if cell else "" for cell in row).lower()
        if "plan id" in row_str:
            for _, col_idx in option_cols:
                if col_idx < len(row):
                    cell_val = str(row[col_idx]).strip() if row[col_idx] else ""
                    if cell_val and "plan id" not in cell_val.lower():
                        plan_id = cell_val
                        break
            if plan_id:
                break

    if plan_id:
        table_text += f"Plan ID: {plan_id}\n"
    
    rates_found = False

    for option_label, col_idx in option_cols:
        option_lines: List[str] = []
        for row_idx, row in enumerate(table):
            if should_skip_row(row, columns["plan_id_col"], header_row_idx, row_idx):
                continue

            if col_idx < len(row):
                cell_text = str(row[col_idx]) if row[col_idx] else ""
                tier_match = re.search(r'\b(EO|ES|EC|EF)\b', cell_text, re.IGNORECASE)
                rate_value = extract_rate_from_cell(cell_text)
                if tier_match and rate_value is not None:
                    tier_abbr = tier_match.group(1).upper()
                    tier_name = TIER_ABBREVIATIONS.get(tier_abbr, tier_abbr.lower())
                    option_lines.append(f"  {tier_name}: ${rate_value:.2f}")
                    rates_found = True

        if option_lines:
            table_text += f"\nOption: {option_label}\n"
            table_text += "\n".join(option_lines) + "\n"
    
    return table_text if rates_found else None


def format_table_with_direct_tier_columns(
    table: List[List],
    columns: Dict[str, Optional[int]],
    page_num: int,
    table_idx: int,
) -> Optional[str]:
    """
    Format table that has direct tier columns (Employee Only, Employee + Spouse, etc.).
    
    Returns formatted text string or None if no rates found.
    """
    header_row_idx = columns["header_row_idx"]
    plan_id_col = columns["plan_id_col"]
    
    if header_row_idx is None:
        return None
    
    # Check if we have any tier columns
    has_tier_cols = any(
        columns.get(key) is not None
        for key in ["employee_only_col", "employee_spouse_col", "employee_child_col", "employee_family_col"]
    )
    
    if not has_tier_cols:
        return None
    
    logger.info(
        f"Page {page_num}, Table {table_idx}: Processing table with direct tier columns "
        f"(EO:{columns['employee_only_col']}, ES:{columns['employee_spouse_col']}, "
        f"EC:{columns['employee_child_col']}, EF:{columns['employee_family_col']})"
    )
    
    table_text = f"\n--- Table {table_idx} (Page {page_num}) - RATES ---\n"
    table_text += "RATES (from direct tier columns):\n"
    
    plans_extracted = 0
    
    for row_idx, row in enumerate(table):
        if should_skip_row(row, plan_id_col, header_row_idx, row_idx):
            continue
        
        # Extract plan ID
        current_plan_id = None
        if plan_id_col is not None and plan_id_col < len(row):
            plan_id_cell = str(row[plan_id_col]) if row[plan_id_col] else ""
            if plan_id_cell.strip() and "plan id" not in plan_id_cell.lower():
                current_plan_id = plan_id_cell.strip()
        
        if not current_plan_id:
            continue
        
        logger.debug(f"Page {page_num}, Table {table_idx}: Processing plan {current_plan_id}")
        
        # Extract rates from tier columns
        plan_rates = []
        tier_mapping = [
            (columns["employee_only_col"], "employee_only"),
            (columns["employee_spouse_col"], "employee_spouse"),
            (columns["employee_child_col"], "employee_child"),
            (columns["employee_family_col"], "employee_family"),
        ]
        
        for col_idx, tier_name in tier_mapping:
            if col_idx is not None and col_idx < len(row):
                rate_value = extract_rate_from_cell(str(row[col_idx]) if row[col_idx] else "")
                if rate_value is not None:
                    plan_rates.append(f"  {tier_name}: ${rate_value:.2f}")
        
        # Only add plan if we found at least one rate
        if plan_rates:
            table_text += f"Plan ID: {current_plan_id}\n"
            table_text += "\n".join(plan_rates) + "\n"
            plans_extracted += 1
            logger.debug(f"  Extracted {len(plan_rates)} rates for plan {current_plan_id}")
    
    if plans_extracted == 0:
        logger.warning(f"Page {page_num}, Table {table_idx}: No plans extracted from direct tier columns table")
        return None
    
    logger.info(f"Page {page_num}, Table {table_idx}: Extracted {plans_extracted} plans from direct tier columns")
    return table_text


def format_table_with_tier_rows(
    table: List[List],
    page_num: int,
    table_idx: int,
) -> Optional[str]:
    """
    Format table where tier names are in the first column and rates are in subsequent columns.
    Each column represents a different plan.
    
    Example structure:
    Row: ["Employee Only", "1", "59.26", "63.29", "62.22", "52.60"]
         [tier names]     [count] [plan1] [plan2] [plan3] [plan4]
    
    Returns formatted text string or None if no rates found.
    """
    # Find the row with tier names in first column
    tier_row_idx = None
    tier_names = []
    
    for row_idx, row in enumerate(table):
        if not row or len(row) < 3:
            continue
        
        first_cell = normalize_text(str(row[0]) if row[0] else "")
        # Check if first cell contains tier names (Employee Only, Employee + Spouse, etc.)
        # The cell may contain multiple tier names separated by newlines
        has_tier_names = (
            "employee only" in first_cell or
            ("employee" in first_cell and ("spouse" in first_cell or "child" in first_cell or "family" in first_cell))
        )
        
        if has_tier_names:
            # This row contains tier names (may be multiline)
            tier_row_idx = row_idx
            # Extract tier names from first column (split by newlines)
            tier_cell = str(row[0]) if row[0] else ""
            tier_lines = [line.strip() for line in tier_cell.split("\n") if line.strip()]
            tier_names = tier_lines
            logger.debug(f"Page {page_num}, Table {table_idx}: Found tier row at index {row_idx} with {len(tier_names)} tier names")
            break
    
    if tier_row_idx is None:
        logger.debug(f"Page {page_num}, Table {table_idx}: No tier row found in tier-row structure detection")
        return None
    
    # Find plan names from header rows (look for "Current", "Renewal", etc. in first few rows)
    plan_names = []
    plan_carriers = []
    
    # Check first few rows for plan names - use general approach without hardcoding keywords
    # Look for header rows that contain potential plan names (non-empty cells that aren't tier/benefit types)
    skip_patterns = ["employee", "tier", "rate", "premium", "total", "dental", "vision", "medical", 
                     "life", "disability", "admin fee", "plan id", "carrier", "network", "deductible"]
    
    for row_idx in range(min(10, len(table))):  # Check more rows to find header
        row = table[row_idx]
        if not row:
            continue
        
        # Look for plan names in this row (skip first column which is usually tier names)
        # Check all columns to handle tables with many plan options
        for col_idx in range(1, min(len(row), 15)):  # Check up to 14 plan columns
            cell = str(row[col_idx]) if col_idx < len(row) and row[col_idx] else ""
            if not cell or not cell.strip():
                continue
                
            cell_lower = normalize_text(cell)
            
            # Skip if this looks like a tier name, benefit type, or other non-plan-name content
            if any(skip in cell_lower for skip in skip_patterns):
                continue
            
            # If cell has reasonable length and doesn't match skip patterns, it might be a plan name
            # Also check if it's not just a number or date
            cell_clean = cell.strip()
            if len(cell_clean) > 2 and not re.match(r'^[\d\s\-\/\.]+$', cell_clean):
                # Extract plan name (may be multiline, take first meaningful part)
                plan_name = cell.split("\n")[0].strip()
                # Clean up common carrier name prefixes (but keep the actual plan name)
                plan_name = re.sub(r'^principal\s+financial\s*', '', plan_name, flags=re.IGNORECASE).strip()
                plan_name = re.sub(r'^principal\s*', '', plan_name, flags=re.IGNORECASE).strip()
                
                if plan_name and len(plan_name) > 2:
                    # Ensure list is long enough (index by col_idx, not append)
                    while len(plan_names) <= col_idx:
                        plan_names.append(None)
                    plan_names[col_idx] = plan_name
                    logger.debug(f"  Found plan name '{plan_name}' at column {col_idx}")
    
    # Also try to find carriers from a row with "Principal" or "Ameritas"
    for row_idx, row in enumerate(table):
        if not row:
            continue
        row_str = " ".join(str(cell) if cell else "" for cell in row).lower()
        if "principal" in row_str or "ameritas" in row_str:
            # Extract carriers from this row
            for col_idx in range(1, min(len(row), 6)):
                cell = str(row[col_idx]) if col_idx < len(row) and row[col_idx] else ""
                cell_lower = cell.lower()
                if "principal" in cell_lower or "ameritas" in cell_lower:
                    # Extract carrier name (may be multiline, take first part)
                    carrier = cell.split("\n")[0].strip()
                    if carrier:
                        # Ensure list is long enough
                        while len(plan_carriers) < col_idx:
                            plan_carriers.append(None)
                        plan_carriers.append(carrier)
                        logger.debug(f"  Found carrier '{carrier}' at column {col_idx}")
            break
    
    # Extract rates from the tier row
    tier_row = table[tier_row_idx]
    if len(tier_row) < 3:
        return None
    
    # Determine start column for rates
    # Check if column 1 looks like employee counts (just numbers) or if it's actually rates
    start_col = 1  # Default: start from column 1
    if len(tier_row) > 1:
        col1_cell = str(tier_row[1]) if tier_row[1] else ""
        # If column 1 contains only numbers (employee counts), skip it and start from column 2
        # Otherwise, column 0 might have tier names + counts combined, so column 1 is the first rate column
        if col1_cell and re.match(r'^[\d\s]+$', col1_cell.strip()) and len(col1_cell.strip()) < 10:
            # Column 1 looks like employee counts (just numbers), skip it
            start_col = 2
        # If column 0 contains both tier names and employee counts (has newlines with numbers), start from column 1
        col0_cell = str(tier_row[0]) if tier_row[0] else ""
        if "\n" in col0_cell and any(re.search(r'\d+', line) for line in col0_cell.split("\n")):
            # Column 0 has tier names + employee counts combined
            start_col = 1
    
    rate_columns = []
    for col_idx in range(start_col, len(tier_row)):
        cell = str(tier_row[col_idx]) if col_idx < len(tier_row) and tier_row[col_idx] else ""
        # Split by newlines to get individual rates
        rates = [line.strip() for line in cell.split("\n") if line.strip()]
        # Include column even if empty - it might have a plan name but no rates yet, or rates might be in a different format
        # We'll filter out columns with no rates later when processing
        if cell.strip() or col_idx < len(plan_names):  # Include if has content or has a plan name
            rate_columns.append((col_idx, rates))
    
    if not rate_columns:
        logger.warning(f"Page {page_num}, Table {table_idx}: Found tier row but no rate columns extracted")
        return None
    
    # Map tier names to standard names
    tier_mapping = {}
    for tier_line in tier_names:
        tier_lower = normalize_text(tier_line)
        if "employee only" in tier_lower or ("employee" in tier_lower and "only" in tier_lower and "spouse" not in tier_lower and "child" not in tier_lower and "family" not in tier_lower):
            tier_mapping[len(tier_mapping)] = "employee_only"
        elif "employee + spouse" in tier_lower or ("employee" in tier_lower and "spouse" in tier_lower):
            tier_mapping[len(tier_mapping)] = "employee_spouse"
        elif "employee + child" in tier_lower or ("employee" in tier_lower and "child" in tier_lower and "family" not in tier_lower):
            tier_mapping[len(tier_mapping)] = "employee_child"
        elif "employee + family" in tier_lower or ("employee" in tier_lower and "family" in tier_lower):
            tier_mapping[len(tier_mapping)] = "employee_family"
    
    if not tier_mapping:
        logger.warning(f"Page {page_num}, Table {table_idx}: Found tier row but couldn't map tier names")
        return None
    
    logger.info(
        f"Page {page_num}, Table {table_idx}: Found tier-row structure with {len(rate_columns)} plan columns, "
        f"tier_row_idx={tier_row_idx}, plan_names={plan_names}, plan_carriers={plan_carriers}"
    )
    
    table_text = f"\n--- Table {table_idx} (Page {page_num}) - PLAN RATES ---\n"
    table_text += f"IMPORTANT: This table contains {len(rate_columns)} separate plan options. Extract ALL of them as separate plans.\n"
    table_text += f"Each plan option has different rates. Do NOT combine or skip any options.\n\n"
    plans_extracted = 0
    
    # Process each plan column
    # Note: plan_idx is 0-based index in rate_columns, but col_idx is the actual column index in table
    # We need to map from rate_columns index to actual column index
    for plan_idx, (col_idx, rates) in enumerate(rate_columns):
        # Get plan name (use plan_names if available, otherwise use column index)
        # plan_names/plan_carriers are indexed by column index, not rate_columns index
        plan_name = None
        # Try to find plan name at this column index
        if col_idx < len(plan_names) and plan_names[col_idx]:
            plan_name = plan_names[col_idx]
        elif col_idx < len(plan_carriers) and plan_carriers[col_idx]:
            plan_name = plan_carriers[col_idx]
        else:
            # Try to find in previous columns (plan names might be offset)
            for check_idx in range(max(0, col_idx - 2), col_idx + 1):
                if check_idx < len(plan_names) and plan_names[check_idx]:
                    plan_name = plan_names[check_idx]
                    break
                elif check_idx < len(plan_carriers) and plan_carriers[check_idx]:
                    plan_name = plan_carriers[check_idx]
                    break
            if not plan_name:
                plan_name = f"Plan {plan_idx + 1}"
        
        # Get carrier if available
        carrier = None
        if col_idx < len(plan_carriers) and plan_carriers[col_idx]:
            carrier = plan_carriers[col_idx]
        
        plan_rates = []
        
        # Match rates to tier names
        for rate_idx, rate_str in enumerate(rates):
            if rate_idx in tier_mapping:
                tier_name = tier_mapping[rate_idx]
                # Extract numeric value (dental proposals often don't use $ prefix)
                rate_value = extract_rate_from_cell(rate_str, require_dollar=False)
                
                if rate_value is not None:
                    plan_rates.append(f"  {tier_name}: ${rate_value:.2f}")
        
        if plan_rates:
            table_text += f"\n--- Plan {plan_idx + 1} (Column {col_idx}) ---\n"
            # Build comprehensive plan name with all available information
            # Include plan name, column position, and any distinguishing features
            plan_name_parts = []
            
            if plan_name and not ("principal" in plan_name.lower() or "ameritas" in plan_name.lower() or "financial" in plan_name.lower()):
                # Use extracted plan name if it's not a carrier name
                plan_name_parts.append(plan_name)
            else:
                # Try to find plan name from text content or use column-based identifier
                plan_name_parts.append(f"Column {col_idx}")
            
            # Add column position for uniqueness
            plan_name_parts.append(f"(Col {col_idx})")
            
            # Combine into final plan name
            actual_plan_name = " ".join(plan_name_parts)
            
            if not plan_name or ("principal" in plan_name.lower() or "ameritas" in plan_name.lower() or "financial" in plan_name.lower()):
                logger.debug(f"  Plan name '{plan_name}' looks like carrier name, using '{actual_plan_name}' for column {col_idx}")
            
            table_text += f"Plan Name: {actual_plan_name}\n"
            table_text += f"Column Position: {col_idx}\n"
            table_text += f"Plan Index: {plan_idx + 1}\n"
            table_text += f"Unique Identifier: Column-{col_idx}-Index-{plan_idx + 1} (use this to ensure uniqueness)\n"
            if carrier:
                table_text += f"Carrier: {carrier}\n"
            table_text += "Rates:\n"
            table_text += "\n".join(plan_rates) + "\n"
            table_text += "\n"  # Add blank line between plans for clarity
            plans_extracted += 1
            logger.debug(f"  Added plan {plan_idx + 1}: {actual_plan_name} (carrier: {carrier}) with {len(plan_rates)} rates")
    
    if plans_extracted == 0:
        logger.warning(f"Page {page_num}, Table {table_idx}: No plans extracted from tier-row structure")
        return None
    
    logger.info(f"Page {page_num}, Table {table_idx}: Extracted {plans_extracted} plans from tier-row structure")
    return table_text


def process_table(table: List[List], page_num: int, table_idx: int) -> Optional[str]:
    """
    Process a single table and return formatted text for LLM extraction.
    
    Returns:
        Formatted table text string, or None if table should be skipped.
    """
    if not table or len(table) == 0:
        return None
    
    # Check if this is a benefit type comparison table (columns = plans, rows = benefit types like Dental/Vision)
    # These tables don't contain tier rates and should be skipped
    if has_benefit_type_rows(table):
        logger.debug(f"Page {page_num}, Table {table_idx}: Skipped (benefit type comparison table, not tier rate table)")
        return None
    
    # Check if this is a summary table that should be skipped
    if is_summary_table(table) and not has_tier_breakdown(table):
        logger.debug(f"Page {page_num}, Table {table_idx}: Skipped (summary/composite table without tier breakdown)")
        return None
    
    # Find column indices
    columns = find_table_columns(table)
    
    # Try to format based on table structure
    # IMPORTANT: Check tier-row structure FIRST, because it might be incorrectly detected as direct tier columns
    tier_row_result = format_table_with_tier_rows(table, page_num, table_idx)
    if tier_row_result:
        logger.info(f"Page {page_num}, Table {table_idx}: Using tier-row structure")
        return tier_row_result

    # Then check for option columns (Current/Renewal)
    if columns.get("current_col") is not None or columns.get("renewal_col") is not None:
        return format_table_with_option_columns(table, columns, page_num, table_idx)
    elif any(
        columns.get(key) is not None
        for key in ["employee_only_col", "employee_spouse_col", "employee_child_col", "employee_family_col"]
    ):
        return format_table_with_direct_tier_columns(table, columns, page_num, table_idx)
    else:
        
        # Fallback: check if it has tier breakdown but unknown structure
        if has_tier_breakdown(table):
            logger.debug(f"Page {page_num}, Table {table_idx}: Unknown structure, showing full table")
            table_text = f"\n--- Table {table_idx} (Page {page_num}) ---\n"
            for row in table:
                row_text = " | ".join(str(cell) if cell else "" for cell in row)
                table_text += row_text + "\n"
            return table_text
        else:
            logger.debug(f"Page {page_num}, Table {table_idx}: Skipped (no tier breakdown found)")
            return None

