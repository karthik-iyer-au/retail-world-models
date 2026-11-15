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
7. D (the rule was kept: the likelihood round measured its own bar, 0.007, and the gain was 0.015; Table 6 now shows the final settings' score, 0.468, and the caption says each round measures its own bar). Favorita search: hurdle NB kept on a gain below its keep bar; latent check names inconsistent.
8. D (the confirmation rule is first-seed against first-seed, now stated; the 3 loss runs with no matched reference are counted). Counts in 15.1 disagree with Table 24 and with themselves.
9. D (table used the largest window; now the mean at the shortest cut, 8.158). Favorita history cut: 8.2 in text, 12.887 in Table 17.
10. D (16 Nov: held-out final periods scored on eight data sets and M5's organisers' final 28 days; Table and paragraph in RQ results; Favorita held nothing out). Test windows are not clean out of sample; the held-out final 24 weeks are never scored; M5's official test
    period is not scored (offered to Karthik 17:40).
11. D (RQ4 question and text say other data sets; revisit row corrected). "Other retailers' data" is mostly the same retailer (Dominick's pooled with Dominick's); rounding mismatches.
12. D (says M5's settings were less accurate on Favorita than Favorita's own). "M5's settings carried to Favorita" overclaims (0.447 vs 0.428, no paired test, one transfer).
13. D (TFT limitation stated; cleaned prices defined in the data section). Only two model classes tuned; TFT is cheap to tune; "cleaned prices" undefined.
14. D (5.1 agrees with 3.6; Limits says RQ3's kept-out counts rest on the window bootstrap). 3.6 and 5.1 contradict on what the carried-state tests show; RQ3 uses a window bootstrap the paper calls
    overconfident.
15. D (Section 17 lists every added analysis; intro points there). Not every analysis added after the results is listed in Section 17 and the abstract.
16. D (decision table says existing items held out of training, and has a stores row). Table 25 "new items" overstates kept-out existing items; RQ3 runs on five of nine data sets with no reason.
17. D (heading matches the numbers; Limits says both models on M5 are single fits). Calibration heading contradicts its numbers; M5 is one fit, not five copies.
18. D (block claims restated from block scores; Figure 9 caption). "LightGBM drifts in total" has no direct measure; Figure 9 caption overclaims (error falls on some categories).
19. Part done (MinT stated as a limitation); open: regret simulation results, category and store selection reasons. Promotion-regret simulation promised, no results; MinT on the tuned forecasts not tested; Dominick's category
    choice and Favorita top-10 selection need a reason.
20. D (ablation counted on the decimals shown; CRPS ties reported as ties; next-period count over the data sets where it can be read). Counts reached by counting rounding ties as wins.
21. Part done (arithmetic, Table 2 start explained); open: search run counts, undefined models. Data paragraph arithmetic (daily vs weekly points), Table 2 training start, search run counts, undefined models.
22. T. "First version / this revision" framing; references to GitHub instead of papers.
23. D (Karthik 16 Nov: licence kept as it is). Licence: PolyForm Noncommercial for code and CC BY-NC text conflict with the paper's advice and with Elsevier.
24. Part done (references ragged-right, no letter-spaced URLs; code paths break only at / and _; code URL wraps; Figure 5 labels enlarged, 16 Nov); open: floats two to four pages after their first mention (27 full-width tables in two columns; most first mentions are forward references in the overview). Layout: URL breaks, letter-spaced URLs, orphan reference, split path, floats far from text, tiny labels.
25. Part done (clever headings and hedges replaced; plain names in the search table and in Figure 5; abstract says the average is most accurate on the rolling windows and the safe choice on the held-out periods, 16 Nov); open: inconsistent terms. Style: hedges, clever headings, code names in tables, inconsistent terms, unexplained jargon.

## Paper 2 (25 items)
Commits for all Paper 2 items: 0491e42, 2f47003, f50d681, 743afde (PDF 22 pages, abstract 249 words).
1. D. C1 to C6 and C8 read the saved final runs paper 1 reports (results/analysis_tuned/runs.csv lists the same run
   folders); C6 uses a refitted tuned LightGBM and C7 refits the final settings up to each episode. Abstract, 5, 7 and
   11 say this; the stale "refits, measured by C3" paragraph (13 October) is replaced.
2. D. One regular price for the audit (highest of the last 12 training periods, else the planned price; rwm.audit.lifts);
   the model's own rule (state_model: recorded or held price, else the 12-period highest, else mean price, else 1.0)
   is stated in 8 and 11 as different, and as the cause of C1 failures under the audit's plan.
3. D (text). Coordinator's change applied: the earlier audit's own-plan pass counts (6 of 540, 272 of 362) are removed
   from abstract, introduction, Table 5 caption and conclusion; 8 says once that own-plan C1 was measured only for the
   tuned models (7) and not rescored for the earlier versions, and that "all 796 failures have a difference" is weak
   evidence when plans differ in 896 of 933. No rescoring run (C, Karthik decides).
4. D. C2 reported but not used as evidence everywhere (3.2, 7, 8, Table 4 caption); removed from abstract; stays in the
   main-test count as fixed in advance (said in 3.2).
5. D. Pooled C7 slope labelled exploratory and added after the robustness results (3.2, 6.2, 7, 8, Figure 3, 10, 12
   captions, abstract); Table 3 caption says the pooled slope itself was in the plan of 13 October.
6. D. Table 1 now lists C7 and C8 without a pass rule; caption, abstract and Contribution 1 say six checks have one.
7. D. Analogy removed (8); 7 and 9 say the known-answer pattern (only the world model overstated) is not reproduced.
8. D. Appendix B lists all eight dated plan entries, with 17 and 18 October marked as after the results; Table 4
   says own-plan C1 was defined 18 October after the earlier results and fixed for the tuned audit 13 November.
9. D. 703-configuration test (abstract, 7), 14-model analysis (4) labelled; Table 3 caption explains.
10. D. 7 now gives pooled window counts and shares from new macros (C1 5 of 48, C2 39 of 48, C3 41 of 44, C5 36 of
    48, C6 18 of 48; paper2/figs/tuned.py); C4 stated as a mean over windows; table gives per-data-set shares.
11. D. C3 and C6 ranges stated as ranges of per-data-set means.
12. D. 6.3 says 63% of the 38 episodes; 8 explains the 25 (re-estimated over the model's forecast weeks, which start up
    to three weeks before the detected start; results/paper2/summary/zones.csv).
13. D. Table 7 caption gives both reasons for not scoring and explains zero-width intervals (one control, or no
    comparison weeks).
14. D. "wide by construction" and "wide and approximately calibrated" replaced by "too narrow".
15. D. Headings and captions corrected (4.2 heading, Figure 5, abstract, C8 wording, 7 causal sentence). Title left;
    LaTeX comment in main.tex proposes "... Faithfulness, Accuracy and Price Response" for Karthik.
16. D. Table 8 gains promotion and display ratio columns (paper2/figs/known.py), so every quoted ratio is in a table.
17. D. 0.80 named as seed 1's promotion agreement; C6 0.67 and 0.54 assigned to their worlds; "narrowly" removed.
18. D. 3.1 states the network multiplies baseline and lift factor and the explanation reports units (b + Lambda);
    Figure 1 shows "+ lift"; C5 uses "lift factor"; model version defined once (3.3); C4 defined on item-week steps.
19. D. 4.1 no longer says "average of three"; 5 says five copies except M5 (one).
20. D. 8 shortened (Figure 13 dropped, text cut), headed as the record; Figures 10, 11, 12 and Tables 5, 6 captions name
    their audit; 8 says the two audits order the models differently and neither orders them.
21. D. Encoder, transition, readouts, rolled, held and recorded price defined in 3.1 and 8; store switching and
    new-item versions removed; Figure 11f lever names in words; known-answer "true lift not zero" defined; C4 rerun
    sample and its 73% against 49% explained up front.
22. D. Favorita in data availability; "paper 1" replaced by the companion paper citation; Snack crackers throughout;
    Appendix B gives dates.
23. D. xurl added (no overfull box over 1pt); double periods and hyphen minus fixed; equations shortened; repro list
    uses itemize; Figure 11f legend moved outside the axes.
24. D. Hedges removed (9, 6.3, 8).
25. D. Plain headings ("What the known-answer study shows", "Limits of the benchmark", "Single-episode intervals are
    too narrow", Figure 1 "Explanation of a forecast"); abstract 249 words, findings first.
Also found: Favorita's C4 (49%) is scored on a price held at 1 in the data, so it does not measure a price response;
7 now says so.

## Outside code review (GPT, 15 November)
1. D. Missing, NaN or duplicate forecasts read as zero: scoring now fails on incomplete forecasts where the data has
   a row (commits 9ec1cce and after; 1,488 tests pass, 8 new); audit of every scored run: all 346 clean (results/audit/).
2. D. A level with no valid scale scored 0.0: now undefined, with the excluded weight reported.
3. D. Finished-run check ignored forecasts (written last): forecasts first, COMPLETE marker last, check needs
   forecasts.csv.
4. D (16 Nov). Partway saves keyed by a fingerprint of the data as loaded (also in the manifest); Favorita cache keyed by the raw files' sizes; tests/test_cache_keys.py.
5. D (16 Nov). No configuration was repeated in the folders analysed (final, cleanrev, ftransfer), so no result changes; the code now takes the latest repeat, as its comment says; tests/test_run_selection.py.
6. Declined before submission: state_model refactor (risk of changing results); recorded for after.
7. D (16 Nov). Coverage measured: 88% of 4,250 statements in src/rwm (1,506 tests); lowest are the optional comparison-model wrappers (TFT 25%, Chronos 41%, Timestral 73%), reconciliation (31%) and the download command (50%), which need packages or network the test machine lacks. Lockfile: requirements-lock.txt, frozen from the cloud image the runs use (scripts/freeze_env.py).

## Outside design review (GPT, 15 November)
Dominick's is six of nine data sets: report by retailer too (T). Settings informed by later windows: stated
limitation. Untouched later window: M5's organisers' test period (C, offered). Accuracy is not proof of response:
check paper 1 wording (T).
