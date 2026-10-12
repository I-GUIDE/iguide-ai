# Validator benchmark (fixture subset)

22 scored turns: 12 correct, 7 wrong, 3 honest refusals (0 left out, unscored; 0 relabelled by hand, see LABEL_OVERRIDES; this checkout's scorer disagrees with 0 labels). 21 carry the supervisor state the replay needs; on 4 of them the auditor's own severity was not recorded and is inferred.

Precision: the share of the turns a check fired on that were wrong. Recall: the share of wrong turns it fired on. A catch is coincidental when the finding's cause is a known false-alarm class (it would fire on a correct answer the same way); *earned* leaves those out.

## Stage 46 baseline, reproduced

Over the 13 turns of the runs archived by then, 1 wrong: a banner on 1 wrong and 9 others, precision 10%, recall 100%.

## The banner

| view | turns | wrong | bannered wrong (earned) | bannered not wrong | precision | recall | precision earned | recall earned |
|---|---|---|---|---|---|---|---|---|
| recorded | 22 | 7 | 6 (3) | 12 | 33% | 86% | 17% | 43% |
| recorded_on_replayable | 21 | 7 | 6 (3) | 11 | 35% | 86% | 18% | 43% |
| replay | 21 | 7 | 6 (4) | 8 | 43% | 86% | 29% | 57% |

## Checks as the live turns printed them

| check | kind | on correct | on wrong (coincidental) | on refusal | precision | recall | precision earned | recall earned |
|---|---|---|---|---|---|---|---|---|
| audit | problem | 3 | 1 (1) | 1 | 20% | 14% | 0% | 0% |
| gate:all_nan | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:count_population | problem | 2 | 2 (2) | 0 | 50% | 29% | 0% | 0% |
| gate:coverage | unverifiable | 2 | 1 (1) | 1 | 25% | 14% | 0% | 0% |
| gate:declared_units | note | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:declared_units | problem | 1 | 1 (1) | 0 | 50% | 14% | 0% | 0% |
| gate:finite_value | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:invariant_gate | unverifiable | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:projected_crs | problem | 2 | 1 (0) | 0 | 33% | 14% | 33% | 14% |
| number_scan | note | 1 | 1 (0) | 0 | 50% | 14% | 50% | 14% |
| number_scan | unverifiable | 1 | 1 (0) | 0 | 50% | 14% | 50% | 14% |
| peer | unverifiable | 0 | 2 (0) | 0 | 100% | 29% | 100% | 29% |

## Checks replayed by this checkout, printed findings

| check | kind | on correct | on wrong (coincidental) | on refusal | precision | recall | precision earned | recall earned |
|---|---|---|---|---|---|---|---|---|
| audit | problem | 3 | 0 (0) | 1 | 0% | 0% | 0% | 0% |
| gate:all_nan | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:count_population | problem | 2 | 2 (2) | 0 | 50% | 29% | 0% | 0% |
| gate:declared_units | problem | 1 | 1 (1) | 0 | 50% | 14% | 0% | 0% |
| gate:finite_value | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:projected_crs | problem | 2 | 2 (0) | 0 | 50% | 29% | 50% | 29% |
| peer | unverifiable | 0 | 2 (0) | 0 | 100% | 29% | 100% | 29% |

## Checks replayed by this checkout, every finding

| check | kind | on correct | on wrong (coincidental) | on refusal | precision | recall | precision earned | recall earned |
|---|---|---|---|---|---|---|---|---|
| audit | problem | 3 | 0 (0) | 1 | 0% | 0% | 0% | 0% |
| gate:all_nan | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:count_population | problem | 2 | 2 (2) | 0 | 50% | 29% | 0% | 0% |
| gate:coverage | unverifiable | 4 | 2 (2) | 1 | 29% | 29% | 0% | 0% |
| gate:declared_units | problem | 1 | 1 (1) | 0 | 50% | 14% | 0% | 0% |
| gate:finite_value | problem | 1 | 0 (0) | 0 | 0% | 0% | 0% | 0% |
| gate:projected_crs | problem | 2 | 2 (0) | 0 | 50% | 29% | 50% | 29% |
| number_scan | note | 2 | 3 (0) | 0 | 60% | 43% | 60% | 43% |
| number_scan | unverifiable | 4 | 1 (0) | 1 | 17% | 14% | 17% | 14% |
| peer | unverifiable | 0 | 2 (0) | 0 | 100% | 29% | 100% | 29% |

## False alarms by cause (recorded: printed on a correct answer or a refusal)

| cause | known class | findings | turns |
|---|---|---|---|
| gate could not reach the frames (work inside a function) | yes | 9 | 3 |
| count check: a count of things that are not frame rows (people, pixels, patches) | yes | 4 | 2 |
| audit: the record holds the claim; the auditor marked it absent | yes | 3 | 2 |
| a mean, median or index typed as a count (fractional 'count') | yes | 3 | 1 |
| audit: knowledge or interpretation in no record |  | 3 | 1 |
| all_nan: an optional column empty after a join that matched | yes | 2 | 1 |
| gate finite_value |  | 2 | 1 |
| projected_crs: a degree area used only as a non-zero test | yes | 2 | 1 |
| gate had nothing to check | yes | 1 | 1 |
| unit vocabulary did not know the unit (before stage 43) | yes | 1 | 1 |
| audit: arithmetic on recorded figures called unsupported | yes | 1 | 1 |
| audit: the reason a refusal gives called unsupported | yes | 1 | 1 |
| projected_crs: an exploratory print measured in degrees, not the answer's run | yes | 1 | 1 |
| number scan: statements cut |  | 1 | 1 |
| number scan: a figure resolved to no record |  | 1 | 1 |
| audit: a fact from a tool description called unsupported | yes | 1 | 1 |

## False alarms by cause (replay: printed on a correct answer or a refusal)

| cause | known class | findings | turns |
|---|---|---|---|
| count check: a count of things that are not frame rows (people, pixels, patches) | yes | 4 | 2 |
| audit: the record holds the claim; the auditor marked it absent | yes | 3 | 2 |
| a mean, median or index typed as a count (fractional 'count') | yes | 3 | 1 |
| audit: knowledge or interpretation in no record |  | 3 | 1 |
| all_nan: an optional column empty after a join that matched | yes | 2 | 1 |
| gate finite_value |  | 2 | 1 |
| projected_crs: a degree area used only as a non-zero test | yes | 2 | 1 |
| audit: arithmetic on recorded figures called unsupported | yes | 1 | 1 |
| audit: the reason a refusal gives called unsupported | yes | 1 | 1 |
| projected_crs: an exploratory print measured in degrees, not the answer's run | yes | 1 | 1 |
| audit: a fact from a tool description called unsupported | yes | 1 | 1 |

## The Sources line (replay)

4 turns render a Sources line that leaves out a data source their record holds (2 correct, 2 wrong, 0 refusals; 5 sources in all). Not a banner: the reader is not warned, the provenance is just missing.

| cause | sources |
|---|---|
| Sources line: a code run reread files an earlier call fetched | 4 |
| Sources line: a source whose result no stated figure came from | 1 |
