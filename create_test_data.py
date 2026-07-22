"""
Generate synthetic messy test documents for VeriRAG evaluation.
Creates PDFs, images, and text files with intentional challenges.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DOCUMENTS_DIR


def create_test_documents():
    """Create synthetic test documents for VeriRAG evaluation."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Document 1: Employee Handbook 2024
    doc1 = """ACME CORPORATION — EMPLOYEE HANDBOOK 2024
Version 3.2 | Effective: January 1, 2024

CHAPTER 1: LEAVE POLICIES

1.1 Annual Leave
Full-time employees are entitled to 24 days of paid annual leave per year, accrued at 2 days per month. 
Part-time employees receive prorated leave based on their contracted hours. Unused leave up to 5 days
may be carried forward to the next calendar year. Leave exceeding the carry-forward limit will expire on 
March 31st of the following year.

1.2 Sick Leave
Employees are entitled to 12 days of sick leave per year. A medical certificate is required for absences
exceeding 3 consecutive days. Unused sick leave does not carry forward.

1.3 Mental Health Days (NEW IN 2024)
Employees may take up to 5 mental health days per year without requiring a medical certificate. These 
days are intended to support employee wellbeing and work-life balance.

CHAPTER 2: REMOTE WORK POLICY

2.1 Hybrid Work Model
Employees can work remotely up to 3 days per week with manager approval. Fully remote arrangements 
require VP-level approval and must be reviewed quarterly.

2.2 Equipment and Expenses
The company provides a one-time home office setup allowance of $500 for approved remote workers.
Internet reimbursement of $50/month is available for employees working remotely more than 2 days/week.

CHAPTER 3: BENEFITS

3.1 Health Insurance
The company provides comprehensive health insurance covering:
- Medical: 90% coverage for in-network providers
- Dental: 100% coverage (increased from 80% in 2023)
- Vision: Annual eye exam and $200 frames allowance

3.2 Training and Development
- Technical skills training: Budget allocation $50,000/year
- Leadership development program: Budget allocation $30,000/year
- Compliance training: Budget allocation $20,000/year
Total training budget: $100,000/year

CHAPTER 4: TERMINATION

4.1 Notice Period
Employees must provide 30 days written notice before voluntary resignation.
The company may provide pay in lieu of notice at its discretion.
"""

    # Document 2: Employment Contract Template (CONTRADICTS Handbook on notice period)
    doc2 = """ACME CORPORATION — STANDARD EMPLOYMENT CONTRACT TEMPLATE
Legal Document Reference: HR-CONTRACT-2024-v2

Section 8: Termination Provisions

8.1 Notice Period
For regular employees, a minimum notice period of 45 days is required for voluntary resignation.
For senior positions (Director and above), the notice period is 60 days.
The company reserves the right to place the employee on garden leave during the notice period.

8.2 Severance
Employees terminated without cause are entitled to severance pay equal to 2 weeks per year of service,
up to a maximum of 26 weeks.

Section 9: Non-Compete
Following termination, employees agree to a 12-month non-compete clause within a 50-mile radius
of any company office location.
"""

    # Document 3: Vendor Contract
    doc3 = """VENDOR SERVICE AGREEMENT
Between: ACME Corporation ("Client") and TechSupply Inc. ("Vendor")
Contract ID: VS-2024-0847
Effective Date: March 1, 2024

ARTICLE 3: FINANCIAL TERMS

3.1 Contract Value
Total contract value: $500,000 over a 24-month period.
Monthly payments: $20,833.33 due on the 15th of each month.

3.2 Late Payment Penalties
Late payments incur a 1.5% monthly interest charge after a 30-day grace period.
Payments more than 90 days overdue may result in service suspension.

3.3 Liability Cap
The total liability of the Vendor under this agreement shall not exceed $500,000.

ARTICLE 5: DATA HANDLING

5.1 Data Retention
All client data must be retained for a minimum of 5 years following the termination of this agreement,
in compliance with financial regulatory requirements.
"""

    # Document 4: Privacy Policy (CONTRADICTS Vendor Contract on data retention)
    doc4 = """ACME CORPORATION PRIVACY POLICY
Last Updated: February 15, 2024
Document Classification: Public

Section 5: Data Retention

5.1 Retention Period
Personal data collected by ACME Corporation will be retained for a maximum period of 3 years
from the date of last interaction with the data subject. After this period, data will be 
anonymized or securely deleted.

5.2 Exceptions
Certain data may be retained longer if required by specific contractual obligations, provided
that the data subject is informed of the extended retention period.

Section 6: Data Subject Rights
Individuals may request access to, correction of, or deletion of their personal data at any time
by contacting our Data Protection Officer at privacy@acmecorp.com.
"""

    # Document 5: Service Agreement
    doc5 = """SERVICE LEVEL AGREEMENT (SLA)
Between ACME Corporation and CloudHost Partners
Agreement Reference: SLA-2024-112
Maximum Liability: $1,200,000

Service Availability Target: 99.9% uptime
Response Time for Critical Issues: 15 minutes
Resolution Time for Critical Issues: 4 hours

Escalation Matrix:
Level 1: Support Team (0-30 minutes)
Level 2: Engineering Team (30 min - 2 hours)
Level 3: VP of Engineering (2-4 hours)
Level 4: CTO (4+ hours)
"""

    # Document 6: Office Lease
    doc6 = """COMMERCIAL LEASE AGREEMENT
Property: Suite 400, 123 Business Park Drive, Tech City, CA 94000
Tenant: ACME Corporation
Landlord: Premier Properties LLC
Lease Value: $300,000/year
Lease Term: January 1, 2024 — December 31, 2026

Monthly Rent: $25,000
Security Deposit: $50,000
Annual Escalation: 3%
"""

    # Document 7: Expense Reimbursement Policy (for OCR test - will have "messy" formatting)
    doc7 = """EXPENSE REIMBURSEMENT POLICY — ACME CORPORATION
Doc Ref: FIN-POL-2024-003      |      Classification: Internal

>>> SECTION 1: SUBMISSION REQUIREMENTS <<<

All employees must submit expense reports within 14 business days
of incurring the expense. Each report MUST include:

   * Original receipts (photographs accepted)
   * Business justification for each expense
   * Appropriate cost center code
   * Manager pre-approval for expenses exceeding $500

>>> SECTION 2: PROCESSING TIMELINE <<<

Reimbursement is processed within 5 business days of approval.
Payment is made via direct deposit to the employee's registered bank account.

>>> SECTION 3: SPENDING LIMITS <<<

   Meals:           $75/person/day (domestic)   $100/person/day (international)
   Transportation:  Economy class for flights under 6 hours
   Accommodation:   Up to $250/night (domestic)   $350/night (international)
   Client Entertainment: Up to $200/person with Director approval
"""

    # Write all documents as text files
    docs = {
        "employee_handbook_2024.txt": doc1,
        "employment_contract_template.txt": doc2,
        "vendor_contract.txt": doc3,
        "privacy_policy.txt": doc4,
        "service_agreement.txt": doc5,
        "office_lease.txt": doc6,
        "expense_policy.txt": doc7,
    }

    for filename, content in docs.items():
        filepath = DOCUMENTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Created: {filepath}")

    print(f"\nAll {len(docs)} test documents created in {DOCUMENTS_DIR}")
    return list(docs.keys())


if __name__ == "__main__":
    create_test_documents()
