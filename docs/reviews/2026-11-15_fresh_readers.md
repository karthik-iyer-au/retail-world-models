# Fresh-reader reviews of 15 November 2026, and an outside code and design review

Two readers who had not seen the work read the built PDFs (paper 1 as an IJPE referee, paper 2 as an ML referee);
GPT reviewed the code and the design. Each item: what was found, and what is done about it.
T = text fix, no compute. C = needs compute (Karthik decides). D = done.

## Paper 1 (25 items)
1. D (e3d43ef). Abstract and introduction still say 26- and 52-week forecasts are not tested; 52-week section heading rests on
   comparisons added after the planned test; the tuned 52-week search is not described in 8.7; 52-week windows undated.
2. D (e3d43ef). Count of most accurate 52-week forecasts leaves cheese out and counts re-tuned LightGBM as plain LightGBM.
3. D. "Most accurate choice" ignores the three-model ensemble, lower than the average at item by store on six data sets.
4. D (paragraph and counts, scripts/make_partner_values.py). LightGBM averaged with TimesFM (no world model) ties or beats the average of the two tuned models at item by store
   on four data sets; the paper must say a second model is needed, and where the world model is that second model.
5. D. "Beats every untrained model on every data set but one" is wrong for snacks (TimesFM with plan 0.855 vs 0.893).
6. D (within-data-set contrasts, retailer counts, slow lines reported). Steady-price vs frequent-cut finding is largely a data-set effect; show the contrast within each data set; the
   five-of-nine rule can be met by Dominick's alone; report by retailer; the slow-lines result is not reported.
7. T. Favorita search: hurdle NB kept on a gain below its keep bar; latent check names inconsistent.
8. T. Counts in 15.1 disagree with Table 24 and with themselves.
9. D (table used the largest window; now the mean at the shortest cut, 8.158). Favorita history cut: 8.2 in text, 12.887 in Table 17.
10. C. Test windows are not clean out of sample; the held-out final 24 weeks are never scored; M5's official test
    period is not scored (offered to Karthik 17:40).
11. T. "Other retailers' data" is mostly the same retailer (Dominick's pooled with Dominick's); rounding mismatches.
12. T. "M5's settings carried to Favorita" overclaims (0.447 vs 0.428, no paired test, one transfer).
13. C/T. Only two model classes tuned; TFT is cheap to tune; "cleaned prices" undefined.
14. T. 3.6 and 5.1 contradict on what the carried-state tests show; RQ3 uses a window bootstrap the paper calls
    overconfident.
15. T. Not every analysis added after the results is listed in Section 17 and the abstract.
16. T. Table 25 "new items" overstates kept-out existing items; RQ3 runs on five of nine data sets with no reason.
17. T. Calibration heading contradicts its numbers; M5 is one fit, not five copies.
18. Part done (Figure 9 caption, e3d43ef); open: "LightGBM drifts in total" has no direct measure; Figure 9 caption overclaims (error falls on some categories).
19. C/T. Promotion-regret simulation promised, no results; MinT on the tuned forecasts not tested; Dominick's category
    choice and Favorita top-10 selection need a reason.
20. T. Counts reached by counting rounding ties as wins.
21. Part done (arithmetic, Table 2 start explained); open: search run counts, undefined models. Data paragraph arithmetic (daily vs weekly points), Table 2 training start, search run counts, undefined models.
22. T. "First version / this revision" framing; references to GitHub instead of papers.
23. T. Licence: PolyForm Noncommercial for code and CC BY-NC text conflict with the paper's advice and with Elsevier.
24. T. Layout: URL breaks, letter-spaced URLs, orphan reference, split path, floats far from text, tiny labels.
25. T. Style: hedges, clever headings, code names in tables, inconsistent terms, unexplained jargon.

## Paper 2 (25 items)
1. T. Contradiction: audits "the companion paper's final models" vs "Dominick's final world models are refits".
2. T. Two definitions of the regular price; this drives every C1 failure.
3. C/T. C1 under the model's own plan is inferred for the earlier audit, not rescored.
4. T. C2 dismissed as evidence, then used as evidence.
5. T. C7 pooled-slope rule was added after results but labelled as fixed in advance.
6. T. C7 and C8 have no rule fixed in advance; Table 1 lists only C1 to C6.
7. T. Known-answer analogy does not hold for LightGBM.
8. T. C1 "as amended" refers to no amendment.
9. T. Post-hoc analyses unlabelled (703-configuration test, 14-model analysis, pooled slope in Table 3).
10. T. Mixed aggregation (data-set means vs window shares) all called "% of windows".
11. T. Ranges that cannot match the pass rates (C3, C6): level not stated.
12. T. Sign counts disagree (24 vs 25 of 38).
13. T. Table 7 caption wrong for four rows; zero-width intervals unexplained.
14. T. Interval width: "wide" vs "far too narrow".
15. T. Headlines and captions overclaim (including "causal validity" in the title).
16. T. Ratios quoted that are in no table.
17. T. Per-lever numbers hard to trace.
18. T. Inconsistent definitions (additive vs multiplicative lift, "model version", C4 per item vs per item-week).
19. T. Three vs five copies.
20. T. The October audit takes three pages despite "stays in the record"; two near-identical figures disagree.
21. T. Terms undefined for outside readers.
22. T. Missing and inconsistent references.
23. T. Layout (URL overrun, double periods, hyphen minus, legend overlap).
24. T. Hedges.
25. T. Clever headings; abstract about 450 words.

## Outside code review (GPT, 15 November)
1. D. Missing, NaN or duplicate forecasts read as zero: scoring now fails on incomplete forecasts where the data has
   a row (commits 9ec1cce and after; 1,488 tests pass, 8 new); audit of every scored run: all 346 clean (results/audit/).
2. D. A level with no valid scale scored 0.0: now undefined, with the excluded weight reported.
3. D. Finished-run check ignored forecasts (written last): forecasts first, COMPLETE marker last, check needs
   forecasts.csv.
4. Open. Cache keys without data fingerprint (checkpoint resume; Favorita panel cache).
5. Open. revision_analysis says "latest run if repeated" but takes the earliest: check duplicates in the audit, then
   make code and comment agree.
6. Declined before submission: state_model refactor (risk of changing results); recorded for after.
7. Open. Coverage measurement and a lockfile.

## Outside design review (GPT, 15 November)
Dominick's is six of nine data sets: report by retailer too (T). Settings informed by later windows: stated
limitation. Untouched later window: M5's organisers' test period (C, offered). Accuracy is not proof of response:
check paper 1 wording (T).
