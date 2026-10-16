# Chapter IV supplementary materials

This directory contains the supplementary materials for Chapter IV, **Modelling microbial genome–environment associations across ecological scales**.

## Contents

- `Supplementary_Information_IV.pdf`: supplementary methods, method tables IV-SM1–IV-SM6 and supplementary figures IV-S1–IV-S5.
- `Table_IV_S1.xlsx`: genome metadata, quality, environmental labels and taxonomic information for the MAGs with OTU assignments considered during construction of the macro-environment dataset.

## Supplementary figures

- **Figure IV-S1:** distribution and overlap of macro-environmental labels among species-level representative genomes.
- **Figure IV-S2:** Moran’s I across Evo 2 embedding representations and neighbourhood sizes.
- **Figure IV-S3:** exploratory evaluation of the 151 multi-environment representatives excluded from single-label model development.
- **Figure IV-S4:** comparison of logistic regression and random forest performance across the sub-environment model-development stages.
- **Figure IV-S5:** class-specific prediction outcomes for the selected KO-based logistic regression model on the held-out test set and the additional evaluation dataset.

## Table IV-S1

Table IV-S1 contains 25,233 MAGs assigned to OTUs as described in the Methods: 22,983 from the Genomes from Earth’s Microbiomes (GEM) catalogue and 2,250 from the anaerobic digestion (AD) genome collection. AD genomes without an OTU assignment are not included. Completeness and contamination correspond to values recalculated using CheckM2.

The `DATABASE` column indicates the source collection. The `RETAINED` column identifies the 10,912 species-level representatives retained after clustering, for which environmental annotations and taxonomy are reported. The `MACRO_MODEL_STATUS` column distinguishes the 10,761 representatives associated with a single macro-environment and included in model development from the 151 representatives associated with multiple macro-environments and excluded from single-label macro-environment modelling.

When a retained representative is associated with multiple environments, the labels are reported in the same cell and separated by ` | `. Blank project, study or biosample fields indicate that the corresponding information was unavailable or not applicable to that genome collection.
