CortexHR Day 20 deliverables
============================
Start with: RELEASE_NOTES.md, docs/reports/CORTEXHR_FINAL_REPORT.md, docs/final_audit/RELEASE_READINESS.md
docs/final_audit/   PROJECT_INVENTORY, MODEL_CARD_VERIFICATION, FINAL_EVALUATION_RESULTS, SERVICE_VERIFICATION, RELEASE_READINESS, MANAGER_RELEASE_DECISION
docs/model_cards/   7 model cards (attrition, resume ranking, resume fairness, attendance, payroll, chatbot/intent, pay band)
docs/architecture/  CORTEXHR_ARCHITECTURE.md (Mermaid)
docs/demo/          DEMO_SCRIPT.md (video NOT recorded)
docs/final_challenge/ + final_challenge/   Final challenge (lives on branch day-20-final-challenge)
docs/evidence/      raw evidence: service verification, monitoring report, test log, rerun outputs
Git release:
The workspace is initialized as a local repository. The audit branch is day-20-final-audit,
the final-challenge branch is day-20-final-challenge, and release v1.0.0 is an annotated local tag.
The repository bundle, when created, can be used for an offline handover:
git clone cortexhr-day20.bundle
git switch day-20-final-audit
git push origin day-20-final-audit day-20-final-challenge v1.0.0

Demo status:
The 5-minute script is complete, but a finished video was not recorded in this environment.
No demo URL is claimed in this release. Add a verified public or private link to this file
and the completion checklist only after the recording has been tested.
