# Gate 2 post-gate analytical addendum

`MECHANICAL_GATE2_VERDICT = SUPPORTED` remains unchanged. This addendum uses only already-generated
Gate 2 CSVs and does not rerun a reservoir, alter a scenario, ranking, criterion or frozen result.

After centering both slopes within scenario, Pearson correlation is `0.960891`
and Spearman is `0.761729`. The scenario-fixed-effect regression
`measured_slope ~ predicted_slope + scenario intercept` gives coefficient `0.521075`.
Pairwise ordering accuracy across all 100 comparisons is `0.680`;
top-2 match is `0.800`. Mean measured-slope regret of the predicted winner is
`5.81525e-05`. In `0.900` of scenarios, the predicted winner
is statistically indistinguishable from the measured best under the paired bootstrap.

Leave-one-scenario-out centered Pearson correlations range from `0.825795` to
`0.992469`. The originally reported aggregate Spearman includes differences of
scale between processes. The within-scenario evidence is therefore more moderate, as already
indicated by the frozen median within-scenario Spearman of 0.60. This clarification does not alter
the mechanical Gate 2 verdict.
