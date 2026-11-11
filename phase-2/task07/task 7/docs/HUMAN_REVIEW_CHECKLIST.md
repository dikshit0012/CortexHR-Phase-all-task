# Human-readability review checklist - Task 7 explanations

**Status: PENDING.** No person has reviewed these explanations yet. The automated check in
`reports/task7_human_review_sheet.csv` (column `automated_lint`) only looks for technical words, a "next step"
and length. It is **not** human feedback and must not be reported as such.

## What to give the reviewer
- `reports/task7_human_review_sheet.csv` (20 explanations, no employee ids) - open it in Excel or Google Sheets.
- This checklist and `REVIEWER_FEEDBACK_TEMPLATE.md`.
- Pick someone who knows payroll or HR but did **not** build this project (a friend, colleague or mentor). Do not explain the
  explanations to them first: if they need your help to understand a sentence, the sentence needs fixing.

## For each of the 20 rows the reviewer answers (Y / N, in the sheet)
1. **What looks unusual?** Can you tell, in your own words, what is unusual about this payslip or attendance record?
2. **Why does it matter?** Do you understand why someone should care (for example, an amount that might be affected)?
3. **What should I check next?** Is it clear what action to take?
4. **No technical words?** Is the sentence free of words you would not use at work (model, score, threshold, outlier, ...)?
Then write a better wording if you have one (`suggested_rewording`) and any comments.

## Things to look out for
- Does any sentence sound as if money was **definitely lost** or someone did something wrong? It should only say "may".
- For "looks different from the usual pattern" explanations: is it clear that the reason is **not known**?
- Are amounts shown with a clear meaning (affected, not "lost")?
- Anything confusing, too long, or too alarming?

## After the review (you fill this in - do not guess)
- Reviewer name / role: ______________________   Date: ______________
- Number of rows where all four answers were Y: ____ / 20
- Rows that need rewording: ______________________
- Changes made to `reason_text.py` as a result: ______________________
- Then set the status in the README to "Reviewed by <role> on <date>" and keep the filled sheet as evidence.
