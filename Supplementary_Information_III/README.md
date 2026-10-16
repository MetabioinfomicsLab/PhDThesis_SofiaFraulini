# Chapter III supplementary materials

This directory contains the supplementary materials associated with Chapter III, **Machine learning-aided prediction of microbial functions critical for anaerobic digestion performance**.

## Contents

### `Supplementary_Information_III.pdf`

This file contains:

- Supplementary Methods III;
- Figures III-S1–III-S5 and their captions;
- captions for Tables III-S1–III-S4.

The supplementary methods describe the sequencing and genomic reconstruction of syntrophic propionate- and butyrate-oxidising microorganisms and the refinement of the negative classes used to construct selected function-specific training datasets.

### `Supplementary_Tables_Chapter_III.xlsx`

This file contains the supplementary tables associated with Chapter III. The information is organised into the following worksheets:

- **Table S1:** positive reference genomes used to construct the nine function-specific training datasets. For each genome, the worksheet reports the associated function, identifier used for model development, organism name, taxonomic order, genome source, assembly level, NCBI link and, where available, completeness and contamination.
- **Table S2:** summary of the training-set composition and selected model for each function, including the numbers of positive and negative genomes, algorithms evaluated, selected estimator, model-selection cross-validation MCC and F1, and number of KOs with a non-zero contribution.
- **Table S2 notes:** explanatory notes concerning the cross-validation summaries and the definition of the reported feature counts.
- **Table S2 CV details:** fold-specific results for LR, RF and SVM, including selected hyperparameters and training and validation MCC and F1 values.
- **Table S2 KO features:** ranked KO features retained by the selected model for each function.
- **Table S3:** metadata and feedstock classification for the 314 samples obtained from the Biogas Microbiome database. The worksheet includes the original reactor and feedstock metadata, the assigned `Feedstock_source` group and non-mutually exclusive Boolean variables describing the presence of specific substrate components.
- **Table S4:** datasets used as anaerobic digestion case studies, including the experimental systems, samples analysed, database identifiers, comparisons, functions investigated and purpose of each analysis.
