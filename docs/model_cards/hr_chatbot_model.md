# Model Card: HR intent classifier (`hr-intent-v1`) and chatbot

## 1. Name / version
`hr-intent-v1`: TF-IDF (word + char_wb n-grams) + Logistic Regression (balanced, max_iter 2000), confidence threshold 0.64. The shipped artifact was **not available**; it was retrained from the committed `chatbot_nlu_training.csv` (scikit-learn 1.9.1 and again 1.8.0), so the retrained model, not the original binary, was tested.

## 2-4. Overview, use, out of scope
Maps an employee question to an HR intent and answers from FAQ text, or falls back to "contact HR". Not for legal, medical or disciplinary advice; not for making any HR decision. Task 10 adds slot extraction, memory and role-based access on a **mock** HR service.

## 5. Dataset
858 training rows, 132 test rows (no exact or near-duplicate overlap with training). Fictional FAQ content.

## 7. Evaluation (re-run on 132 held-out rows)
Accuracy 0.803; macro-F1 0.7997; weighted-F1 0.7997; 26 misclassified. Mean confidence correct 0.647, wrong 0.293. At threshold 0.64: answered 58 of 132 (44%) with accuracy 0.9655 (2 wrong answers); 74 sent to fallback, **50 of which were actually correct**. Weakest intents: help (F1 0.33), chitchat 0.55, team_members 0.57, thanks 0.57.

## 8. Fairness
Not evaluated (no demographic analysis of language or dialect).

## 9. Limitations
Small data; English only; weak on short or generic messages ("help"); mock HR data; role-based access uses a demo header, not real authentication. Task 10 integration tests fail (3) because the converse patch is not applied to the Task 8 app.

## 10. Ethical considerations
Chat can leak payslip or personal data if authorisation fails; Task 10 is designed to fail closed, but only mock identities were tested.

## 11. Deployment
`POST /v1/chat`; verified locally: "How many casual leaves do I get in a year?" -> `hr_policy_leave` (0.953); notice-period query -> `notice_period_policy` (0.927); "tell me a joke" -> out_of_scope fallback. Not tested remotely.

## 12. Monitoring
None.

## 13. Release recommendation
**INTERNAL DEMONSTRATION ONLY** for the full chatbot (mock service, demo identity); the intent classifier alone is RELEASE WITH WARNINGS.
