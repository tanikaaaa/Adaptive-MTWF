# Milestone 3B · TMWF synthetic-to-real fidelity

The best ratio settings reproduced several direction, burst, and duration statistics. Timing discrepancies at two and three tabs and a structural length ceiling at four tabs leave the suitability of these traces for pretraining unestablished. The tested uniform-delay schedule ranked below every ratio condition on the aggregate score for each available tab count.

## Data and comparison protocol

| TBB release file | Raw rows | Nonempty rows | Monotonic rows | Clean rows with a monitored label |
|---|---:|---:|---:|---:|
| `2tabs` | 1,166 | 1,166 | 1,148 | 951 |
| `2tabs-extra` | 10,605 | 10,603 | 10,504 | 8,554 |
| `3tabs` | 1,311 | 1,311 | 1,296 | 1,228 |
| `4tabs` | 1,477 | 1,477 | 1,477 | 1,448 |

The four files in `TMWF/dataset/tbb realworld/` matched their release archive members byte for byte. Each extensionless protocol-4 pickle contains variable-length 1-D `float32` timestamps, parallel `int8` signs, and an `int8` label matrix with one slot per tab. Labels `0–49` represent monitored classes and `50` represents background. The adapter retains the original slots, including repeats. All nonempty traces begin at time zero. Timestamp units are inferred to be seconds from the paper and released delay code. The release contains no direction zeros, and lengths span many values up to 70,059. The original loader applies selection, padding, and truncation at model loading. Capture preprocessing before serialization remains unknown. The observed backward-step counts were 18, 99, 16, and 0, respectively, with no repairs applied. [Real data archive](<../../../TMWF/dataset/tbb realworld.tar.gz>) · [Original real loader](../../../TMWF/code/real_openworld.py#L67) · [Delay code](../../../TMWF/code/MergeSingleTraces_openworld.py#L209) · [TMWF paper, Appendices G and H, pp. 19–20](../../../TMWF/complete_paper_version_of_TMWF.pdf)

The Tor Browser collection stops at four tabs. The five-tab file is in the Chrome release, which has different traffic isolation. Local `generated_appendix_f`, `generated_appendix_h`, and `generated_appendix_h_delay` contain synthetic reproduction data. The primary comparator uses all eligible rows in the ordinary TBB files. `2tabs-extra` is a separate sensitivity comparison. The released model protocol takes the first 1,000 rows of each ordinary file. [Release inventory](../../../TMWF/README.md#L40) · [Reproduction loader](../../../TMWF/code/appendix_h_data.py#L100) · [TMWF paper, Appendix H, p. 20](../../../TMWF/complete_paper_version_of_TMWF.pdf)

The audit synthesized 500 traces in each of 24 conditions using seed `2023`, for 12,000 total. Each of the 2–4-tab source plans samples a real label row from the clean cohort with a monitored label, maps background `50` to `-1`, and draws distinct eligible single-tab training rows for those slots. Five-tab plans use an unstratified draw with a monitored label. Every schedule within a tab count uses the same plans and first 5,000 cells per source. All plans are feasible at 50% overlap, so the comparisons isolate changes in scheduling conditional on feasibility. This restriction rejected 131, 305, 570, and 678 identity draws for 2–5 tabs. The single-tab pool contained 7,945 clean training rows. The current closed-world target requires at least one monitored class and permits background tabs alongside it. An open-world background-only target remains a separate policy decision.

The main comparisons use monotonic, nonempty real rows with at least one monitored label. Raw feature summaries retain the observed order of backward steps and include all nonempty rows. Coverage also includes all nonempty real rows and every synthetic trace. The audit computes trace-level scalar KS statistics without significance tests. The descriptive ranking averages mean KS across four equally weighted domains. Base-2 JS divergence compares the mean 20-bin time-occupancy profiles separately. Bursts are contiguous same-direction runs. Inter-arrival time uses adjacent recorded cells, with no artificial initial zero. [Feature definitions](../../adaptive_mtwf/data/fidelity.py)

## Fidelity results

| Tabs | Lowest aggregate KS | Delay aggregate KS | Duration KS, lowest-score ratio / delay | Lowest time-profile JS |
|---|---:|---:|---:|---:|
| 2 | 20% · 0.119 | 0.167 | 0.049 / 0.331 | 50% · 0.0186 |
| 3 | 30% · 0.150 | 0.228 | 0.072 / 0.538 | 50% · 0.0032 |
| 4 | 20% · 0.123 | 0.259 | 0.100 / 0.888 | 30% · 0.0113 |

- **Closest settings.** The 20% ratio also ranked first against `2tabs-extra` (0.131). No single overlap ratio led across tab counts and metrics. A ratio mixture remains a pretraining hypothesis. The audit compared individual ratios and did not test mixture weights or downstream transfer.
- **Closer features.** Outgoing fraction KS was 0.097–0.110 for the lowest-score ratios, and mean burst length KS was 0.078–0.139. Direction-change rate and several positive-IAT summaries also overlap substantially, with differences quantified in the JSON.
- **Largest gaps.** At two and three tabs, median zero-IAT fractions were 85.7% and 84.9% real versus 89.2% and 89.3% synthetic. Their KS distances were 0.459 and 0.557 for every schedule. For the lowest-score ratios, IAT p95 KS was 0.343 and 0.419, respectively. Four-tab length KS was 0.223 for every schedule. The 4-tab delay median duration was 38.4 s versus 67.0 s in the matched real cohort.
- **Tab count and delay.** The best aggregate ratio score did not degrade monotonically from two to four tabs. Length-tail coverage and delay duration agreement worsened with tab count. The `M_delay` test used uniform 2–9 s start-to-start gaps. It sometimes improved a direction or burst metric, but performed worse overall, so these results establish no need to include that schedule. The published capture procedure motivates the bounds but does not identify the realized onset distribution in the release.
- **Concurrency and ties.** At 50% ratio, nonadjacent overlaps occurred in 44.0%, 78.0%, and 94.8% of 3–5-tab mixtures. The tested delay gave 79.4%, 95.4%, and 98.0%. Its median peak active counts were 3, 3, and 4. These are source-interval overlap counts. Real files contain no per-tab intervals for a direct comparison. Exact cross-source ties were rare, with a maximum condition mean of 0.00053% of cells. Across 768 paired tie-policy checks, timestamp and direction sequences agreed. The timing gap therefore has little support as a tie-policy effect in these conditions.

## Sequence-length evidence

### Percentage of nonempty traces fully retained

| Window | Real 2 | Real 2 extra | Real 3 | Real 4 | Synthetic 2 | Synthetic 3 | Synthetic 4 | Synthetic 5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 53.0 | 55.4 | 26.2 | 8.8 | 46.6 | 17.8 | 6.2 | 2.0 |
| 8,000 | 78.0 | 80.7 | 52.9 | 24.4 | 90.0 | 50.2 | 20.0 | 8.6 |
| 10,000 | 87.6 | 89.4 | 68.3 | 38.6 | 100.0 | 68.4 | 37.0 | 19.4 |
| 12,000 | 92.7 | 94.2 | 79.3 | 51.5 | 100.0 | 87.0 | 62.6 | 33.0 |
| 15,000 | 97.0 | 97.5 | 88.8 | 67.6 | 100.0 | 100.0 | 88.8 | 60.8 |
| 20,000 | 99.6 | 99.0 | 96.6 | 84.8 | 100.0 | 100.0 | 100.0 | 93.2 |

### Real length quantiles, rounded to whole cells

| Setting | p90 | p95 | p97.5 | p99 |
|---|---:|---:|---:|---:|
| 2 tabs | 10,766 | 13,048 | 15,660 | 18,955 |
| 2 tabs extra | 10,226 | 12,529 | 15,012 | 19,911 |
| 3 tabs | 15,397 | 17,942 | 21,290 | 26,404 |
| 4 tabs | 23,641 | 29,846 | 37,700 | 43,977 |

For a 95% full-retention target, 15,000 is the smallest tested window that meets it in both real two-tab files, and 20,000 is the smallest that meets it at three tabs. No tested window reaches 95% at four tabs. At 20,000 cells, among truncated real four-tab traces, p10 and p25 of the fraction retained are 48.3% and 60.8%. The synthetic five-tab result has no TBB comparator. These are coverage measurements for research planning.

## Research decision

The current audit supports ratio synthesis as an experimental baseline. It does not establish that synthetic pretraining transfers to real traces. The strongest evidence for a targeted generator experiment is the fixed length ceiling. Four capped sources provide at most 20,000 cells, while 15.2% of real four-tab traces exceed it. A controlled audit-only ablation at **7,500 cells per source** would raise that ceiling to 30,000, near the real p95. It would measure how much of the length gap is caused by the source cap while keeping source identities, schedules, and comparison metrics fixed. The current 5,000-cell specification still applies. Timing data do not identify a specific change to the delay distribution or burst process. Real per-tab onsets, capture-session dependence, and the packet-to-cell timing procedure remain unverified.

## Reproduction

The complete audit and figures took 20.83 s wall time on the local run. The test suite passed all 57 tests. From `BTP` with NumPy and Matplotlib available:

```sh
cd MTWF
python3 -B -W error -m unittest discover -s tests -v
MPLCONFIGDIR="$(mktemp -d /tmp/mtwf-mpl.XXXXXX)" /usr/bin/time -p \
  python3 -B -m scripts.audit_tmwf_fidelity \
  --real-dir "../TMWF/dataset/tbb realworld" \
  --single-dir "../TMWF/dataset/tbb single tab" --samples 500 --seed 2023
```

[Real data audit](tmwf_real_multitab_audit.json) · [Fidelity and diagnostics](synthetic_multitab_fidelity_audit.json) · [Coverage and retention](sequence_length_coverage.json) · [Distribution plots](tmwf_fidelity_distributions.pdf) · [KS heatmap](tmwf_fidelity_distances.pdf) · [Coverage plot](tmwf_sequence_length_coverage.pdf)
