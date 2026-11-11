# Original experiment (v1) - frozen snapshot

These files are the **first** Task 6 results, copied unchanged before the code review that followed.
They were produced with: duplicate rule = "flag the later copy in input-file order", Isolation Forest fitted on
rows in input-file order, and a previous-month lookup that used the previous *row*.

The code review found that those choices made results depend on row order (see `../task6_comparison_report.md`,
section "Review after the original experiment"). The corrected implementation is now the default.
The original behaviour is still reproducible: `python "task 6/scripts/run_task6.py" --original-experiment`.
Do not edit these files; they are the record of what was originally measured.
