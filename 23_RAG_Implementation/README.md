# 23. Evidence retrieval implementation

| File | Purpose |
|---|---|
| build_corpus.py | Builds out/evidence_corpus.tsv (30,158 documents with provenance) from Lyra 2026 tracks, ClinVar (2024-09-13), Findlay 2018, PDB-derived residue context, domains and verified literature records |
| evidence_retriever.py | Scenario 1 structure-aware retriever (8 Å heavy-atom neighbourhood plus ±3 sequence window, 1/(1+d) weights) with fold-specific allowed residues; `leakage_check` |
| tests_leakage.py | Automated leakage tests T1–T4, including a positive control. Must pass before Model D runs. Log: out/leakage_tests.log |
| scenario2_interpretation.py | Scenario 2 BM25 / TF-IDF / hybrid (RRF) / hybrid with variant filter retrieval, deterministic cited synthesis, automatic verification |
| out/ | Corpus, summaries, per-query metrics, synthesis examples |

Model D is trained by `20_GNN_Model_Source/run_experiments.py` (models gnn_D_evidence, gnn_D_nograph, lgbm_seqevo_ev),
which calls the retriever per outer fold and asserts zero leakage violations.
Reports: 24_RAG_Leakage_Control_Report.md and 25_RAG_Evaluation.md.
