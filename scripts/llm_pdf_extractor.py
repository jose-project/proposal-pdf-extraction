import argparse
import asyncio
import json
import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple

import pdfplumber

from scripts.remote_llm import RemoteLLM
from scripts.table_processor import process_table
from scripts.timing import TimingRecorder

# Configure logging
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Extract insurance plan rates from PDF proposal text. Output ONLY valid JSON — no markdown, no prose.

EXTRACT per plan:
- carrier: insurer name
- plan_name: plan label or column header (e.g. "Gold PPO", "Current Plan", "Option 1")
- plan_id: short alphanumeric code adjacent to plan name (e.g. S531BCE, G531PPO, AIBPP615) — null if absent
- rate_structure: see list below
- rates: numeric only (strip "$" and commas; "N/A" → null)

SKIP: totals, composites, enrollment counts, pages with no per-tier EO/ES/EC/EF breakdown.

MULTIPLE PLANS: When side-by-side columns each have their own rates (Current/Renewal, Option A/B, Plan 1/2), return each column as a separate plan object with plan_name = the column header.

COLUMN ABBREVIATIONS: EO = employee_only, ES = employee_spouse, EC = employee_child, EF = employee_family, E+1 = employee_plus_one, E+2 = employee_plus_two_or_more.

RATE STRUCTURES and their rates keys:
- 2_tier: employee_only, employee_family
- 3_tier: employee_only, employee_plus_one, employee_plus_two_or_more
- 4_tier: employee_only, employee_spouse, employee_child, employee_family
- 5_tier: employee_only, employee_spouse, employee_child, employee_two_or_more_children, employee_family
- aca_age / age_band_5 / age_band_10: age-range string keys like "<20", "20-24", "64+"
- 6_tier / 8_tier: like 4_tier but with additional tier keys (e.g. employee_plus_one, employee_two_or_more_children)
- esc_5_year / esc_10_year: {"employee": {age bands}, "spouse": {age bands}, "children": number|null}
- 4_tier_5_year / 4_tier_10_year: {"employee_only": {age bands}, "employee_spouse": {age bands}, "employee_child": {age bands}, "employee_family": {age bands}}
- 3_tier_age_band / 2_tier_age_band: same age-range key style, fewer tier keys

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": string, "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Anthem Gold PPO (G531PPO) with EO $524.18 / ES $1,048.36 / EC $786.27 / EF $1,310.45:
{"carrier":"Anthem","plans":[{"plan_name":"Gold PPO","plan_id":"G531PPO","rate_structure":"4_tier","rates":{"employee_only":524.18,"employee_spouse":1048.36,"employee_child":786.27,"employee_family":1310.45}}]}"""

SYSTEM_PROMPT_ACA_BREAKDOWN = """Extract ACA age-banded insurance plan rates from a healthcare benefits proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: This is a "healthcare & benefits proposal" with a "Medical Employee Costs Breakdown" section. Each plan occupies one page showing:
- Plan header line: "[Number] [Carrier] [PlanID] [Full Plan Name]"  e.g. "1 BlueCross BlueShield of Illinois B5N1BCE Blue Choice Preferred Bronze"
- A "Monthly Age Banded Rates" table with 51 age bands displayed in 3 side-by-side columns
- Age bands run: <=14, 15, 16, 17 … 63, 64+  (read all 3 columns left-to-right, top-to-bottom)

IGNORE card-view summary pages — they show only a "Total Monthly Cost" lump sum, not individual age rates.

EXTRACT per plan:
- carrier: insurer name (e.g. "BlueCross BlueShield of Illinois")
- plan_name: full plan name from the header line (e.g. "BlueCross BlueShield of Illinois B5N1BCE Blue Choice Preferred Bronze")
- plan_id: short alphanumeric code from the header line (e.g. "B5N1BCE", "S534BCE", "G530BCE", "P5M1BCE")
- rate_structure: always "aca_age"
- rates: all 51 age-band values as numbers (strip "$" and commas)

RATE KEYS — use exactly these 51 strings in this order:
"<=14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30",
"31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47",
"48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64+"

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "aca_age", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — plan B5N1BCE with age <=14 $267.77, age 35 $427.74, age 64+ $1,050.09:
{"carrier":"BlueCross BlueShield of Illinois","plans":[{"plan_name":"BlueCross BlueShield of Illinois B5N1BCE Blue Choice Preferred Bronze","plan_id":"B5N1BCE","rate_structure":"aca_age","rates":{"<=14":267.77,"15":291.58,"16":300.68,"17":309.78,"18":319.58,"19":329.38,"20":339.53,"21":350.03,"22":350.03,"23":350.03,"24":350.03,"25":351.43,"26":358.43,"27":366.83,"28":380.48,"29":391.69,"30":397.29,"31":405.69,"32":414.09,"33":419.34,"34":424.94,"35":427.74,"36":430.54,"37":433.34,"38":436.14,"39":441.74,"40":447.34,"41":455.74,"42":463.79,"43":474.99,"44":488.99,"45":505.45,"46":525.05,"47":547.10,"48":572.30,"49":597.15,"50":625.16,"51":652.81,"52":683.26,"53":714.06,"54":747.32,"55":780.57,"56":816.62,"57":853.03,"58":891.88,"59":911.13,"60":949.98,"61":983.59,"62":1005.64,"63":1033.29,"64+":1050.09}}]}"""

SYSTEM_PROMPT_BCBSIL_RENEWAL = """Extract insurance plan rates from a Blue Cross and Blue Shield of Illinois (BCBSIL) Group Renewal Exhibit. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: This is a multi-section renewal packet. ONLY extract from these three appendix sections — ignore all other pages (cover, instructions, plan notes, legal, enrollment):
  • "Appendix - Monthly Medical Premiums"   — Medical plans with Age Rates and Composite Rates tables
  • "Appendix - Monthly Dental Premiums"    — Dental plans with age-band and composite rate tables
  • "Appendix - Monthly Standalone Vision Premiums" — Vision plans with composite rates only

EXTRACT per plan:
- carrier: always "Blue Cross and Blue Shield of Illinois"
- plan_name: full plan name as shown (e.g. "G534BCE Blue PPO", "DILHR30 Blue Dental Choice Select")
- plan_id: alphanumeric code adjacent to plan name (e.g. "G534BCE", "S531PPO", "DILHR30", "DILLM41") — null if absent
- rate_structure: always "4_tier"
- rates: Composite Rates row — EO / ES / EC / EF mapped to keys below

COLUMN ABBREVIATIONS: EO = employee_only, ES = employee_spouse, EC = employee_child, EF = employee_family

IMPORTANT — USE COMPOSITE RATES, NOT AGE RATES:
Each medical and dental plan has two sub-tables:
  1. "Age Rates" table — age-banded (ACA bands) — SKIP THIS
  2. "Composite Rates" table — four tiers (EO / ES / EC / EF) — EXTRACT THIS
For vision plans there is only a composite rate table.

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Medical plan G534BCE with EO $524.18, ES $1,048.36, EC $786.27, EF $1,310.45:
{"carrier":"Blue Cross and Blue Shield of Illinois","plans":[{"plan_name":"G534BCE Blue PPO","plan_id":"G534BCE","rate_structure":"4_tier","rates":{"employee_only":524.18,"employee_spouse":1048.36,"employee_child":786.27,"employee_family":1310.45}}]}

EXAMPLE — Dental plan DILHR30 with EO $32.50, ES $65.00, EC $78.00, EF $104.00:
{"carrier":"Blue Cross and Blue Shield of Illinois","plans":[{"plan_name":"DILHR30 Blue Dental Choice Select","plan_id":"DILHR30","rate_structure":"4_tier","rates":{"employee_only":32.50,"employee_spouse":65.00,"employee_child":78.00,"employee_family":104.00}}]}"""

SYSTEM_PROMPT_BCBSIL_BBF = """Extract insurance plan rates from a Blue Cross and Blue Shield of Illinois Blue Balance Funded (BBF) ASO Proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: This is a BCBSIL "Blue Balance Funded" self-funded proposal. Extract ONLY from "Appendix - Total Monthly Charges" pages — ignore cover, instructions, benefit summary, and census pages. Each appendix page shows one plan with a "4 - Tier for Billing" section containing these rows:
  Monthly Enrollment / Administrative Fees / Individual Stop Loss Premium / Aggregate Stop Loss Premium / Projected Claim Funding / Total Monthly Charges / Monthly Tier Total

EXTRACT PER PLAN — "Total Monthly Charges" row only:
- carrier: always "Blue Cross and Blue Shield of Illinois"
- plan_name: Plan ID shown in the page header before the tier table (e.g. "AIBPP615", "AIBCO609")
- plan_id: same alphanumeric code
- rate_structure: always "4_tier"
- rates: Total Monthly Charges values → employee_only / employee_spouse / employee_child / employee_family

TIER COLUMN ORDER: Employee Only | Employee + Spouse | Employee + Child(ren) | Employee + Family
DO NOT extract: Administrative Fees, Stop Loss Premiums, Projected Claim Funding, Monthly Tier Total, or the "All Tiers" total column.

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Plan AIBPP615, Total Monthly Charges EO $948.00, ES $1,833.56, EC $1,795.73, EF $2,681.32:
{"carrier":"Blue Cross and Blue Shield of Illinois","plans":[{"plan_name":"AIBPP615","plan_id":"AIBPP615","rate_structure":"4_tier","rates":{"employee_only":948.00,"employee_spouse":1833.56,"employee_child":1795.73,"employee_family":2681.32}}]}"""

SYSTEM_PROMPT_AETNA_COST_GRID = """Extract insurance plan rates from an Aetna renewal cost grid proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: This is an Aetna "Renewal FI" proposal with Medical Cost Grid and Dental Cost Grid pages. The column order is always: EE / EE+SP / EE+CH / FAM / Total (skip Total). Numbers in parentheses like (27) (4) (0) (3) are enrollment counts — NOT rates, skip them.

Each plan row: [PlanID] [copay info] $EE $EE+SP $EE+CH $FAM $GroupTotal
Below a plan row there may be a line starting "ID:" with an internal ID — ignore it.
Plans appear under section headers CURRENTPLANS, RENEWINGPLANS, ALTERNATEPLANS — extract all sections.

EXTRACT per plan:
- carrier: "Aetna"
- plan_name: the plan code string (e.g. "KSPPO6500HSA70/50ECYV23", "VolKS8APPOMax1500")
- plan_id: same plan code
- rate_structure: "4_tier"
- rates: EE→employee_only, EE+SP→employee_spouse, EE+CH→employee_child, FAM→employee_family

SKIP: the last (5th) dollar amount on each row (group total), and enrollment counts in parentheses.

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — plan KSPPO6500HSA70/50ECYV23 with EE $488.00, EE+SP $1,360.00, EE+CH $858.00, FAM $1,455.00:
{"carrier":"Aetna","plans":[{"plan_name":"KSPPO6500HSA70/50ECYV23","plan_id":"KSPPO6500HSA70/50ECYV23","rate_structure":"4_tier","rates":{"employee_only":488.00,"employee_spouse":1360.00,"employee_child":858.00,"employee_family":1455.00}}]}"""

SYSTEM_PROMPT_LF_MEDICAL_TIERED = """Extract insurance plan rates from a LifeFirst/United Healthcare "Medical Plan Summary/Rates" proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Each page shows multiple plans. For each plan, rates appear across four lines:
  [Plan Code]  EE: $xxx.xx   $nn,nnn.nn   [deductible/copay columns…]
               SP: $xxx.xx   …
               CH: $xxx.xx   …
               FAM: $xxx.xx  …

The FIRST dollar amount after each tier label (EE: / SP: / CH: / FAM:) is the individual monthly rate — extract this.
The SECOND larger dollar amount on the EE line (e.g. $21,969.45) is the "Total Monthly Required Health Cost" — SKIP.
All deductible, OOP, copay, and prescription amounts are benefit details — SKIP.

EXTRACT per plan:
- carrier: null (not consistently labeled in this format)
- plan_name: plan code (e.g. "HP60002575i8025B", "P2500i100LX26B")
- plan_id: same plan code
- rate_structure: "4_tier"
- rates: EE→employee_only, SP→employee_spouse, CH→employee_child, FAM→employee_family

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — plan HP60002575i8025B: EE $559.73, SP $1,119.47, CH $1,091.48, FAM $1,735.17:
{"carrier":null,"plans":[{"plan_name":"HP60002575i8025B","plan_id":"HP60002575i8025B","rate_structure":"4_tier","rates":{"employee_only":559.73,"employee_spouse":1119.47,"employee_child":1091.48,"employee_family":1735.17}}]}"""

SYSTEM_PROMPT_MEWA_OHIO = """Extract insurance plan rates from an Ohio Chamber Health MEWA (Multiple Employer Welfare Arrangement) proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Separate sections for "Medical Rates", "Dental CoInsurance Plans Rates", and "Vision Rates". In each section a header row lists plan codes, then rate rows follow:
  Employee          24  $xxx.xx  $xxx.xx  $xxx.xx  …   (one $ per plan column)
  Employee+Spouse    2  $xxx.xx  …
  Employee+Child(ren) 1  $xxx.xx  …
  Employee+Family    3  $xxx.xx  …

The NUMBER immediately after the tier label (24, 2, 1, 3) is an enrollment count — NOT a rate, skip it.
Plan codes appear in the header row (e.g. "EQY4w/G15S EQY3w/G15S" medical, "P3303 P3384" dental, "S1006 S1008" vision).

EXTRACT per plan (one plan per column in the header row):
- carrier: "Ohio Chamber Health"
- plan_name: plan code (e.g. "EQY4w/G15S", "P3303", "S1006")
- plan_id: same code
- rate_structure: "4_tier"
- rates: Employee→employee_only, Employee+Spouse→employee_spouse, Employee+Child(ren)→employee_child, Employee+Family→employee_family

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Medical plan EQZBw/G15S: Employee $875.49, Employee+Spouse $1,750.98, Employee+Child(ren) $1,619.66, Employee+Family $2,714.01:
{"carrier":"Ohio Chamber Health","plans":[{"plan_name":"EQZBw/G15S","plan_id":"EQZBw/G15S","rate_structure":"4_tier","rates":{"employee_only":875.49,"employee_spouse":1750.98,"employee_child":1619.66,"employee_family":2714.01}}]}"""

SYSTEM_PROMPT_OPTIMYL_GERBER = """Extract insurance plan rates from an Optimyl Benefits / Gerber Life self-funded proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Rates appear in a "Medical Composite Monthly Cost" table with numbered plan columns (Plan 1, Plan 2, Plan 3, Plan 4):
  Tier               EE's
  Employee            8    $xxx.xx  $xxx.xx  $xxx.xx  $xxx.xx
  Employee + Spouse   2    $xxx.xx  …
  Employee + Child    1    $xxx.xx  …
  Family              0    $xxx.xx  …

The NUMBER after the tier label (8, 2, 1, 0) is an enrollment count — NOT a rate, skip it.

EXTRACT per plan column:
- carrier: "Gerber Life / Optimyl Benefits"
- plan_name: "Plan 1", "Plan 2", "Plan 3", or "Plan 4" (use network/type from Proposal Summary if available, e.g. "Plan 1 PPO")
- plan_id: null
- rate_structure: "4_tier"
- rates: Employee→employee_only, Employee+Spouse→employee_spouse, Employee+Child→employee_child, Family→employee_family

SKIP rows: Total Medical Monthly Cost, Monthly Stop Loss Premium, Monthly Administrative Fees, Monthly Claims Account Funding.

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Plan 1: Employee $559.95, Employee+Spouse $1,511.86, Employee+Child $1,119.89, Family $1,903.82:
{"carrier":"Gerber Life / Optimyl Benefits","plans":[{"plan_name":"Plan 1","plan_id":null,"rate_structure":"4_tier","rates":{"employee_only":559.95,"employee_spouse":1511.86,"employee_child":1119.89,"employee_family":1903.82}}]}"""

SYSTEM_PROMPT_NATIONWIDE_SELF_FUNDED = """Extract insurance plan rates from an Allstate Benefits / Nationwide self-funded medical plan proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Medical rates appear under "Monthly Bill Medical" on the "Stop-Loss Insurance and Financial Details" page. Two plan columns: Plan 1 and Plan 2.

EXTRACT these rows (map to 4_tier keys):
  Employee           → employee_only
  Employee + Spouse  → employee_spouse
  Employee + Child   → employee_child
  Family             → employee_family

IGNORE — these are insurance/admin costs, NOT per-employee rates:
  Stop-loss Premium, Admin/Sales/General Expenses, Claims Account

Dental plans may appear in a separate "Dental Portion of Cost" table with EE/ES/EC/Fam rows — extract as separate plan entries with plan_name indicating the dental plan name.

EXTRACT per plan:
- carrier: "Allstate Benefits"
- plan_name: "Plan 1" or "Plan 2" (or dental plan name if in dental table)
- plan_id: null
- rate_structure: "4_tier"
- rates: as mapped above

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Plan 1: Employee $450.52, Employee+Spouse $1,351.55, Employee+Child $1,126.29, Family $1,711.95:
{"carrier":"Allstate Benefits","plans":[{"plan_name":"Plan 1","plan_id":null,"rate_structure":"4_tier","rates":{"employee_only":450.52,"employee_spouse":1351.55,"employee_child":1126.29,"employee_family":1711.95}}]}"""

SYSTEM_PROMPT_JENSENIT_RFP = """Extract insurance plan option premiums from a broker dental+vision RFP comparison document. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: 3-page side-by-side comparison. Page 1 shows group-level monthly premium totals per column/option:
  Row "Dental"    → monthly dental premium for that option
  Row "Vision"    → monthly vision premium for that option
  Row "TOTAL"     → combined total (skip — it's just dental + vision)
  Row "Admin Fee" → admin fee (skip)

Column headers name the carrier/option (e.g. "Current/Principal", "Renewal/Principal", "Ameritas", "Principal - Vision $130").
Pages 2 and 3 contain benefit feature descriptions — no rate values, ignore them.

EXTRACT per column option:
- carrier: carrier name for that column (e.g. "Principal Financial", "Ameritas")
- plan_name: the column header label (e.g. "Current/Principal", "Renewal/Principal", "Ameritas Dental with Principal Vision $130")
- plan_id: null
- rate_structure: "group_total"
- rates: {"dental_monthly": number, "vision_monthly": number}

Skip columns that have no numeric dental or vision values.

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "group_total", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — "Current/Principal" with Dental $394.03, Vision $42.60:
{"carrier":"Principal Financial","plans":[{"plan_name":"Current/Principal","plan_id":null,"rate_structure":"group_total","rates":{"dental_monthly":394.03,"vision_monthly":42.60}}]}"""

SYSTEM_PROMPT_TRUSTMARK_AETNA = """Extract insurance plan rates from a Trustmark HealthyEdge proposal (Aetna network). Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Each plan has a two-page block. The ODD page has a summary with a composite rate table:
  Family Status  |  Composite Rate  |  Number of EE's  |  Monthly Cost
  EE             |  $xxx.xx         |  nn               |  $nn,nnn.xx
  ES             |  $xxx.xx         |  …
  EC             |  $xxx.xx         |  …
  FF             |  $xxx.xx         |  …       ← FF = Employee + Family

The EVEN page shows cost breakdown (Stop-Loss, Admin, Claim Prefunding) — SKIP THIS PAGE.
Extract ONLY the "Composite Rate" column — NOT "Number of EE's" or "Monthly Cost".

EXTRACT per plan:
- carrier: "Trustmark"
- plan_name: plan label shown at top of the summary page (e.g. "Plan 13", "Plan 14", "Plan 15")
- plan_id: null
- rate_structure: "4_tier"
- rates: EE→employee_only, ES→employee_spouse, EC→employee_child, FF→employee_family

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — Plan 13: EE $645.23, ES $1,435.25, EC $1,078.17, FF $2,414.20:
{"carrier":"Trustmark","plans":[{"plan_name":"Plan 13","plan_id":null,"rate_structure":"4_tier","rates":{"employee_only":645.23,"employee_spouse":1435.25,"employee_child":1078.17,"employee_family":2414.20}}]}"""

SYSTEM_PROMPT_ANTHEM_RENEWAL = """Extract insurance plan rates from an Anthem Blue Cross and Blue Shield fully insured renewal packet. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Two types of rate pages:
  1. "Renewal rate sheet": Each plan has a "Current" row and a "Renewal" row with EO/ES/EC/EF rates plus a group total (last column — SKIP). Extract both rows as separate plan entries.
  2. "Portfolio plans and rates" pages: Wide table with many plans. Rate columns are labeled Employee / Employee + Spouse / Employee + Children / Employee + Family.

EXTRACT per plan per rate period:
- carrier: "Anthem Blue Cross and Blue Shield"
- plan_name: plan description + rate period (e.g. "Blue Access PPO 5000 Current", "Blue Access PPO 5000 Renewal")
- plan_id: short code near plan name (e.g. "9CP7" from "Tiered-9CP7") — null if absent
- rate_structure: "4_tier"
- rates: Employee→employee_only, Employee+Spouse→employee_spouse, Employee+Children→employee_child, Employee+Family→employee_family

SKIP: enrollment counts, premium increase column, enrollment-weighted group total (last $ column).

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — "Blue Access PPO 5000" Current: EO $469.11, ES $1,031.11, EC $791.86, EF $1,448.15:
{"carrier":"Anthem Blue Cross and Blue Shield","plans":[{"plan_name":"Blue Access PPO 5000 Current","plan_id":"9CP7","rate_structure":"4_tier","rates":{"employee_only":469.11,"employee_spouse":1031.11,"employee_child":791.86,"employee_family":1448.15}}]}"""

SYSTEM_PROMPT_SIDECAR_HEALTH = """Extract insurance plan rates from a Sidecar Health large group proposal. Output ONLY valid JSON — no markdown, no prose.

DOCUMENT FORMAT: Plan options appear on two key pages:
  - "Quote proposal" page: one proposed plan with EO/ES/EC/EF rates + Total Monthly Cost (last column — SKIP).
  - "Alternative Plan Options" page: multiple plan rows, each with EO/ES/EC/EF rates + Total Monthly Cost (SKIP).

Row format: [Plan Name]  $EO  $ES  $EC  $EF  $TotalMonthly
Column order: Employee Only | Employee + Spouse | Employee + Child(ren) | Employee + Family | Total Monthly Cost

EXTRACT per plan option:
- carrier: "Sidecar Health"
- plan_name: full plan name (e.g. "Sidecar Health Employer (0/5000)", "Sidecar Health Employer (0/250)")
- plan_id: null
- rate_structure: "4_tier"
- rates: Employee Only→employee_only, Employee+Spouse→employee_spouse, Employee+Child(ren)→employee_child, Employee+Family→employee_family

SKIP: Projected Enrollment row, Broker Service Fees row, Savings comparison table, Total Monthly Cost (5th column).

SCHEMA: {"carrier": string|null, "plans": [{"plan_name": string|null, "plan_id": string|null, "rate_structure": "4_tier", "rates": object}]}
No rates found: {"carrier": null, "plans": []}

EXAMPLE — "Sidecar Health Employer (0/5000)": EO $441.83, ES $971.14, EC $745.81, EF $1,363.93:
{"carrier":"Sidecar Health","plans":[{"plan_name":"Sidecar Health Employer (0/5000)","plan_id":null,"rate_structure":"4_tier","rates":{"employee_only":441.83,"employee_spouse":971.14,"employee_child":745.81,"employee_family":1363.93}}]}"""

USER_PROMPT_TEMPLATE = """File: {pdf_name}, Pages: {page_range}

{page_text}

Return JSON only (no markdown, no explanation)."""

# Pattern to match money values with "$" prefix (e.g., "$1,700", "$524", "$829.64")
# Note: some proposals (e.g., dental) may list rates without "$" in tables; those are handled elsewhere.
MONEY_PATTERN = re.compile(r"\$[\d,]+\.?\d*")


@dataclass
class PlanEntry:
    plan_name: Optional[str]
    plan_id: Optional[str]
    rate_structure: str
    rates: Dict[str, Any]
    source_pages: List[int] = field(default_factory=list)


# Keywords that indicate rate-related content
RATE_KEYWORDS = re.compile(
    r"\b(?:rates?|premium|employee|spouse|child|family|tier|plan|coverage|deductible|"
    r"age[s]?|band[s]?|carrier|option|ppo|hmo|aca)\b",
    re.IGNORECASE,
)


def chunk_pages(
    pdf_path: Path,
    pages_per_chunk: int = 1,
    max_chars: int = 6000,
    filter_empty: bool = True,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Iterable[Tuple[List[int], str]]:
    """
    Yield (page_numbers, text) tuples while limiting prompt size.
    Optimized to skip clearly non-relevant pages early.
    """
    logger.info(f"Starting to chunk pages from PDF: {pdf_path.name}")
    
    with pdfplumber.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)
        logger.info(f"PDF has {total_pages} total pages")
        
        buffer_pages: List[int] = []
        buffer_text: List[str] = []
        processed_pages = 0
        skipped_pages = 0

        for idx, page in enumerate(pdf_doc.pages, start=1):
            # Apply page range filter if provided (1-based indexing)
            if start_page is not None and idx < start_page:
                continue
            if end_page is not None and idx > end_page:
                continue
            
            # Extract tables first (more structured than plain text)
            tables = page.extract_tables()
            page_content_parts = []
            
            # Process each table using the table processor
            if tables:
                for table_idx, table in enumerate(tables):
                    formatted_table = process_table(table, idx, table_idx + 1)
                    if formatted_table:
                        page_content_parts.append(formatted_table)
            
            # Also include plain text for context (headers, footers, etc.)
            text = (page.extract_text() or "").strip()
            if text:
                # Normalize whitespace efficiently
                clean_text = " ".join(text.split())
                if clean_text:
                    page_content_parts.append(f"\n--- Page {idx} Text ---\n{clean_text}")
            
            # Combine table and text content
            combined_content = "\n".join(page_content_parts).strip()
            
            if not combined_content:
                skipped_pages += 1
                logger.debug(f"Page {idx}: Skipped (no content)")
                continue

            # Early filtering: skip pages that clearly don't contain rate data
            from scripts.constants import DEFAULT_MIN_CONTENT_LENGTH
            if filter_empty and len(combined_content) < DEFAULT_MIN_CONTENT_LENGTH:
                skipped_pages += 1
                logger.debug(f"Page {idx}: Skipped (too short: {len(combined_content)} chars)")
                continue

            # Check if current page alone exceeds max_chars
            # If so, yield buffer first (if any), then process this page alone
            if len(combined_content) > max_chars:
                # Current page is too large - yield buffer first if it exists
                if buffer_pages:
                    logger.info(
                        f"Chunk created: pages {buffer_pages[0]}-{buffer_pages[-1]} "
                        f"({len(buffer_pages)} pages, {len(' '.join(buffer_text))} chars)"
                    )
                    yield buffer_pages, "\n".join(buffer_text)
                    processed_pages += len(buffer_pages)
                    buffer_pages = []
                    buffer_text = []
                
                # Process large page alone (will be truncated in process_chunk if needed)
                logger.warning(
                    f"Page {idx} exceeds max_chars ({len(combined_content)} > {max_chars}), "
                    f"processing alone (will be truncated if needed)"
                )
                yield [idx], combined_content
                processed_pages += 1
                continue
            
            candidate_pages = buffer_pages + [idx]
            candidate_text = "\n".join(buffer_text + [combined_content]) if buffer_text else combined_content

            # Yield chunk if limits exceeded
            if (
                len(candidate_pages) > pages_per_chunk
                or len(candidate_text) > max_chars
            ) and buffer_pages:
                logger.info(
                    f"Chunk created: pages {buffer_pages[0]}-{buffer_pages[-1]} "
                    f"({len(buffer_pages)} pages, {len(' '.join(buffer_text))} chars)"
                )
                yield buffer_pages, "\n".join(buffer_text)
                processed_pages += len(buffer_pages)
                buffer_pages = [idx]
                buffer_text = [combined_content]
            else:
                buffer_pages = candidate_pages
                buffer_text = buffer_text + [combined_content] if buffer_text else [combined_content]

        if buffer_pages:
            processed_pages += len(buffer_pages)
            final_text = "\n".join(buffer_text) if buffer_text else ""
            logger.info(
                f"Final chunk created: pages {buffer_pages[0]}-{buffer_pages[-1]} "
                f"({len(buffer_pages)} pages, {len(final_text)} chars)"
            )
            yield buffer_pages, final_text

    logger.info(
        f"Chunking complete: {processed_pages} pages processed, "
        f"{skipped_pages} pages skipped, {total_pages} total pages"
    )


def extract_json(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response with robust error handling.
    Attempts to recover JSON even if wrapped in text.
    """
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip()
    
    # Try direct parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block (might be wrapped in markdown or text)
    # Look for first { and matching }
    start = response.find("{")
    if start == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    end = -1
    for i in range(start, len(response)):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
    
    if end > start:
        try:
            json_str = response[start : end + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


def coerce_rate_value(value: Any) -> Any:
    """
    Coerce a value to a numeric rate value.
    
    Handles:
    - Numbers (int/float) -> float
    - Strings with "$" prefix -> extract numeric part
    - "N/A" or similar -> None
    - Nested structures (dict/list) -> recursive coercion
    
    Args:
        value: Value to coerce (can be any type)
    
    Returns:
        Coerced value (float, None, or recursively coerced structure)
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        # Extract numeric part from "$" amounts (some documents use "$", some don't)
        match = MONEY_PATTERN.search(cleaned)
        if match:
            try:
                # Extract numeric part (remove "$" and convert to float)
                rate_str = match.group(0).replace("$", "").replace(",", "")
                return float(rate_str)
            except ValueError:
                return value
        lowered = cleaned.lower()
        if lowered in {"na", "n/a", "not applicable"}:
            return None
        # Plain numeric string (e.g. "59.26") — no "$" prefix
        try:
            return float(cleaned)
        except ValueError:
            pass
    if isinstance(value, dict):
        return {k: coerce_rate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [coerce_rate_value(v) for v in value]
    return value


def _infer_rate_structure(rates: Dict[str, Any]) -> Optional[str]:
    """Infer rate_structure from the keys of the rates dict when the model omits it."""
    keys = {str(k).lower() for k in rates.keys()}

    # Nested ESC structure: top-level keys are "employee", "spouse", "children"
    if {"employee", "spouse", "children"} <= keys:
        return "esc_5_year"

    # Age-banded: many keys that look like age ranges
    age_pattern = re.compile(r'^[<>]?\d{1,2}(-\d{1,2})?\+?$')
    age_keys = [k for k in keys if age_pattern.match(k)]
    if len(age_keys) >= 5:
        if len(age_keys) <= 14:
            return "age_band_5"
        return "aca_age"

    # Standard tier structures — match known tier key names
    tier_keys = {
        "employee_only", "employee_spouse", "employee_child", "employee_family",
        "employee_plus_one", "employee_plus_two_or_more",
        "employee_two_or_more_children",
    }
    count = len(keys & tier_keys)
    if count == 2:
        return "2_tier"
    if count == 3:
        return "3_tier"
    if count == 4:
        return "4_tier"
    if count == 5:
        return "5_tier"
    if count == 6:
        return "6_tier"
    if count >= 7:
        return "8_tier"

    return None


def normalize_plan(raw: Dict[str, Any], source_pages: List[int]) -> Optional[PlanEntry]:
    """
    Normalize a raw plan dictionary into a PlanEntry object.

    Tries multiple key aliases the model may use for rate_structure and rates,
    infers rate_structure from rates keys when absent, and logs on failure.
    """
    plan = raw.get("plan") or raw

    # Accept alternate key names the 7B model commonly returns
    rate_structure = (
        plan.get("rate_structure")
        or plan.get("structure")
        or plan.get("tier_structure")
        or plan.get("type")
        or plan.get("rate_type")
    )
    rates = (
        plan.get("rates")
        or plan.get("tiers")
        or plan.get("tier_rates")
        or plan.get("rate_values")
    )

    if not isinstance(rates, dict):
        logger.warning(
            f"Dropping plan: rates missing or wrong type "
            f"(keys present: {list(plan.keys())}, raw={raw})"
        )
        return None

    # Infer rate_structure from rates keys if model didn't return it
    if not isinstance(rate_structure, str) or not rate_structure:
        rate_structure = _infer_rate_structure(rates)
        if rate_structure:
            logger.info(
                f"Inferred rate_structure={rate_structure!r} from rates keys: {list(rates.keys())}"
            )
        else:
            logger.warning(
                f"Dropping plan: could not determine rate_structure "
                f"(rates keys={list(rates.keys())}, raw={raw})"
            )
            return None

    normalized_rates = coerce_rate_value(rates)
    if not isinstance(normalized_rates, dict):
        logger.warning(
            f"Dropping plan: rates normalization failed for rate_structure={rate_structure!r}"
        )
        return None

    return PlanEntry(
        plan_name=plan.get("plan_name"),
        plan_id=plan.get("plan_id"),
        rate_structure=rate_structure,
        rates=normalized_rates,
        source_pages=sorted(set(source_pages)),
    )


async def process_chunk(
    llm: RemoteLLM,
    pdf_name: str,
    page_numbers: List[int],
    page_text: str,
    retries: int = 2,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """
    Process a single chunk of pages through the LLM.
    
    Sends the page text to the LLM for extraction, handles retries on JSON parsing
    failures, and returns normalized extraction results.
    
    Args:
        llm: RemoteLLM instance for API calls
        pdf_name: Name of the PDF file (for logging)
        page_numbers: List of page numbers in this chunk
        page_text: Combined text content from pages
        retries: Number of retry attempts on failure (default: 2)
        max_tokens: Maximum tokens for LLM response (default: 4096)
    
    Returns:
        Dictionary with "carrier" and "plans" keys, or empty result on failure
    """
    page_range_str = f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
    logger.info(f"Processing chunk: pages {page_range_str} ({len(page_numbers)} pages, {len(page_text)} chars)")
    
    # Truncate text if too long to avoid token limit issues
    from scripts.constants import DEFAULT_TEXT_TRUNCATE_LIMIT
    original_len = len(page_text)
    if len(page_text) > DEFAULT_TEXT_TRUNCATE_LIMIT:
        page_text = page_text[:DEFAULT_TEXT_TRUNCATE_LIMIT] + "...[truncated]"
        logger.warning(f"Chunk truncated from {original_len} to {len(page_text)} chars")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        pdf_name=pdf_name,
        page_range=", ".join(str(p) for p in page_numbers),
        page_text=page_text,
    )

    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            logger.debug(f"LLM request for pages {page_range_str}, attempt {attempt + 1}/{retries + 1}")
            response = await llm.chat(
                SYSTEM_PROMPT,
                user_prompt,
                max_new_tokens=max_tokens,
            )
            logger.debug(f"LLM response received for pages {page_range_str}: {len(response)} chars")
            
            data = extract_json(response)
            if data is not None and isinstance(data, dict):
                plans_count = len(data.get("plans", []))
                carrier = data.get("carrier")
                retry_note = f" (after {attempt} retr{'y' if attempt == 1 else 'ies'})" if attempt > 0 else ""
                logger.info(
                    f"Chunk processed successfully: pages {page_range_str} - "
                    f"found {plans_count} plans, carrier: {carrier or 'None'}{retry_note}"
                )
                data["_retries"] = attempt
                return data
            
            # If JSON parsing failed, retry with reminder
            last_error = "JSON parse failed"
            logger.warning(f"JSON parsing failed for pages {page_range_str}, attempt {attempt + 1}")
            logger.debug(f"LLM response (first 500 chars): {response[:500] if response else 'None'}")
            attempt += 1
            if attempt <= retries:
                user_prompt += "\n\nCRITICAL: You must respond with ONLY valid JSON. No markdown, no code blocks, no explanations. Just the JSON object starting with { and ending with }."
        except Exception as e:
            last_error = str(e)
            logger.error(f"Error processing chunk pages {page_range_str}, attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt <= retries:
                wait_time = 10 * attempt
                logger.debug(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    # Return empty result on failure
    logger.warning(f"Chunk processing failed after {retries + 1} attempts: pages {page_range_str}")
    if last_error:
        logger.error(f"Last error: {last_error}")
    return {"carrier": None, "plans": [], "_retries": attempt}


async def _stream_chunks_async(
    pdf_path: Path,
    pages_per_chunk: int,
    max_chars: int,
    filter_empty: bool,
    start_page: Optional[int],
    end_page: Optional[int],
) -> AsyncGenerator[Tuple[List[int], str], None]:
    """
    Async generator wrapping the synchronous chunk_pages iterator.

    Runs PDF I/O in a background thread so LLM tasks can be dispatched
    as soon as the first chunk is ready, without blocking the event loop.
    Chunks are fed through an asyncio.Queue (capacity 8) that naturally
    back-pressures the reader if the LLM semaphore is fully saturated.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    _DONE = object()

    def _producer() -> None:
        try:
            for item in chunk_pages(
                pdf_path,
                pages_per_chunk=pages_per_chunk,
                max_chars=max_chars,
                filter_empty=filter_empty,
                start_page=start_page,
                end_page=end_page,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(_DONE), loop).result()

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        thread.join()


async def extract_pdf_with_llm(
    pdf_path: Path,
    output_path: Path,
    pages_per_chunk: int = 1,
    max_chars: int = 6000,
    max_concurrent: int = 3,
    max_tokens: int = 4096,
    filter_empty: bool = False,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract rates from PDF using LLM with optimized parallel processing.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path to write JSON output
        pages_per_chunk: Pages per LLM prompt
        max_chars: Max characters per chunk
        max_concurrent: Max concurrent LLM requests
        max_tokens: Max tokens per LLM response
    """
    recorder = TimingRecorder()

    logger.info(
        f"Starting extraction: {pdf_path.name} "
        f"(chunks: {pages_per_chunk} pages, max_chars: {max_chars}, "
        f"max_concurrent: {max_concurrent}, max_tokens: {max_tokens})"
    )

    llm = RemoteLLM()
    carriers: List[str] = []
    plans: Dict[str, PlanEntry] = {}

    try:
        # Process chunks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(page_numbers: List[int], text: str) -> Dict[str, Any]:
            async with semaphore:
                with recorder.measure("llm_call", accumulate=True):
                    return await process_chunk(
                        llm, pdf_path.name, page_numbers, text, max_tokens=max_tokens
                    )

        # Pipeline: dispatch each LLM task as soon as its chunk is ready so
        # LLM calls overlap with the remaining PDF I/O instead of waiting for
        # the entire document to be read first.
        chunks: List[Tuple[List[int], str]] = []
        tasks: List[asyncio.Task] = []
        t_chunk_start = time.perf_counter()

        async for page_numbers, text in _stream_chunks_async(
            pdf_path,
            pages_per_chunk=pages_per_chunk,
            max_chars=max_chars,
            filter_empty=filter_empty,
            start_page=start_page,
            end_page=end_page,
        ):
            chunks.append((page_numbers, text))
            tasks.append(asyncio.create_task(process_with_limit(page_numbers, text)))

        recorder.record("pdf_load_chunking", time.perf_counter() - t_chunk_start)

        if not chunks:
            logger.warning(f"No chunks created from PDF: {pdf_path.name}")
            output = {
                "file": str(pdf_path),
                "carrier": None,
                "plans": [],
                "timing": recorder.summary(),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            logger.info(f"Extraction complete: 0 plans found, output written to {output_path}")
            recorder.log(logger, label=pdf_path.name)
            return output

        logger.info(f"Started {len(tasks)} pipelined tasks (max_concurrent={max_concurrent})")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"All {len(results)} tasks completed")

        # Process results
        with recorder.measure("merge_normalise"):
            successful_chunks = 0
            failed_chunks = 0
            total_plans_found = 0

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_chunks += 1
                    logger.error(f"Chunk {idx + 1} failed with exception: {result}")
                    continue

                if not isinstance(result, dict):
                    failed_chunks += 1
                    logger.warning(f"Chunk {idx + 1} returned invalid result type: {type(result)}")
                    continue

                successful_chunks += 1
                carrier = result.get("carrier")
                if isinstance(carrier, str) and carrier.strip():
                    carriers.append(carrier.strip())

                chunk_plans = result.get("plans", [])
                total_plans_found += len(chunk_plans)

                for plan_raw in chunk_plans:
                    if not isinstance(plan_raw, dict):
                        continue
                    entry = normalize_plan(plan_raw, chunks[idx][0])
                    if not entry:
                        continue
                    page_key = f"page-{chunks[idx][0][0]}"
                    plan_identifier = entry.plan_name or f"plan-{len(plans)}"
                    plan_id_str = entry.plan_id or ""
                    column_suffix = ""
                    if plan_identifier:
                        col_match = re.search(r'(?:Col|Column)\s+(\d+)', plan_identifier)
                        if col_match:
                            column_suffix = f"-col{col_match.group(1)}"
                        idx_match = re.search(r'Index-(\d+)', plan_identifier)
                        if idx_match:
                            column_suffix += f"-idx{idx_match.group(1)}"
                    key = "::".join(
                        [
                            plan_id_str,
                            plan_identifier + column_suffix,
                            entry.rate_structure or "",
                            page_key,
                        ]
                    )
                    if key not in plans:
                        plans[key] = entry
                    else:
                        plans[key].source_pages = sorted(
                            set(plans[key].source_pages + entry.source_pages)
                        )

        valid_results = [r for r in results if isinstance(r, dict)]
        chunks_with_retries = sum(1 for r in valid_results if r.get("_retries", 0) > 0)
        total_retry_calls = sum(r.get("_retries", 0) for r in valid_results)
        retry_rate = chunks_with_retries / len(chunks) if chunks else 0.0
        logger.info(
            f"Results processing complete: {successful_chunks} successful chunks, "
            f"{failed_chunks} failed chunks, {total_plans_found} total plans found, "
            f"{len(plans)} unique plans after deduplication — "
            f"retry rate: {chunks_with_retries}/{len(chunks)} chunks ({retry_rate:.0%}), "
            f"{total_retry_calls} extra LLM call(s)"
        )

        # Determine most common carrier
        carrier_value = None
        if carriers:
            most_common = Counter(carriers).most_common(1)
            if most_common:
                carrier_value = most_common[0][0]
                logger.info(f"Most common carrier: {carrier_value} (found {most_common[0][1]} times)")

        timing_summary = recorder.summary()
        timing_summary["chunks"] = len(chunks)

        output = {
            "file": str(pdf_path),
            "carrier": carrier_value,
            "plans": [
                {
                    "plan_name": plan.plan_name,
                    "plan_id": plan.plan_id,
                    "rate_structure": plan.rate_structure,
                    "rates": plan.rates,
                    "source_pages": plan.source_pages,
                }
                for plan in plans.values()
            ],
            "timing": timing_summary,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

        recorder.log(logger, label=pdf_path.name)
        logger.info(
            f"Extraction complete: {len(plans)} unique plans extracted, "
            f"carrier: {carrier_value or 'None'}, output written to {output_path}"
        )

        return output

    finally:
        # Cleanup LLM session
        await llm.close()
        logger.debug("LLM session closed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plan rate data from a proposal PDF using Qwen LLM."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    parser.add_argument(
        "--out",
        help="Optional output JSON path (defaults to output/<pdf_name>.json)",
    )
    parser.add_argument(
        "--pages-per-chunk",
        type=int,
        default=2,
        help="Number of pages to group per LLM prompt",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Maximum characters per chunk",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_path = (
        Path(args.out).resolve()
        if args.out
        else Path("output") / f"{pdf_path.stem}.json"
    )

    asyncio.run(
        extract_pdf_with_llm(
            pdf_path,
            output_path,
            pages_per_chunk=args.pages_per_chunk,
            max_chars=args.max_chars,
        )
    )


if __name__ == "__main__":
    main()

