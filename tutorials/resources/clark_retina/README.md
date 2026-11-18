# Clark et al. embryonic mouse retina

Six original 10x v2 libraries from GSE118614 span E12, E14, E16 and E18. The authors aligned these files with Cell Ranger 1.2.1 against mm10 v1.2.0. The frozen run manifest records each original ENA BAM URL, byte count and MD5. The .bam.1 suffix belongs to the deposited filename. Each URL is a complete coordinate-sorted BAM. The index is optional for this sequential velocyto analysis.

## From submitted BAMs to six looms

With the standalone ZIP, extract it, open a terminal in the extracted directory and run the commands below. In a scCS source checkout, first change into `tutorials/resources/clark_retina`. Use the Linux filesystem if working in WSL.

```bash
export RETINA_WORK="$HOME/projects/clark-retina"
mkdir -p "$RETINA_WORK/ref"
bash retina.sh barcodes
bash retina.sh download E16
```

E16 is the smallest pilot. If you already have the submitted BAMs, put them in `$RETINA_WORK/bam/` as E12.bam, E14_rep1.bam, E14_rep2.bam, E16.bam, E18_rep2.bam and E18_rep3.bam. The download command verifies existing files. The six BAMs total 67.62 GB decimal; allow additional disk for the Cell Ranger reference archive, barcode-sorted BAMs, sorting temporaries and looms. Run one library at a time.

Use the GTF already present in your original **refdata-cellranger-mm10-1.2.0** reference if you have it. Otherwise fetch the archived 10x reference and extract its GTF. Your GRCm39-2024-A reference uses different genomic coordinates.

```bash
bash retina.sh reference
# Alternatively export RETINA_GTF=/path/to/refdata-cellranger-mm10-1.2.0/genes/genes.gtf

# In the environment containing velocyto and samtools
bash retina.sh velocity E16
bash retina.sh download
bash retina.sh velocity
```

The whitelist is built from the authors' annotated cells for each individual library. In the 2021 annotation update the exact label column is `umap2_CellType`. GEO's E12 corresponds to `sample=E12_rep1`. Different libraries can use the same 16-base barcode. The published cell key is the library plus the barcode. The scripts preserve the 10x `-1` suffix in whitelist files and join the velocyto IDs within each library. Missing annotated cells and unexpected loom barcodes are reported by the notebook. If a run stops during barcode sorting, inspect any `cellsorted_*.bam` with `samtools quickcheck` before restarting. velocyto can reuse that intermediate.

## Local analysis

In the standalone kit, keep the notebook alongside the TSVs and script. In the source checkout open `tutorials/25_clark_fetal_retina_multiscorer.ipynb`. From the shell with `RETINA_WORK` exported, launch Jupyter using your scCS tutorial environment (scCS, scVelo 0.3.4, Scanpy, AnnData, loompy, Matplotlib and Jupyter). If starting from a fresh environment, install the checkout using `python -m pip install -c constraints-tutorials.txt -e '.[tutorials]' loompy` from its root. E16 alone tests loom production. Run the four-age notebook after all six looms exist. The notebook writes quantitative tables, a fitted velocity graph and editable PDF figures to `$RETINA_WORK/results/`. It analyzes the shared developmental reference and repeats scoring with transitions restricted by age.

## Provenance

`cells.tsv.gz` contains the library ID, author sample name, age, 10x barcode and published cell type for the six selected libraries from the authors' 2021 CSV. Counts in the guide describe those annotations before velocyto filtering. Full sequencing files stay on your computer.

[Clark et al., Neuron, PMID 31128945](https://pubmed.ncbi.nlm.nih.gov/31128945/) · [2021 annotations, including horizontal cells](https://github.com/gofflab/developing_mouse_retina_scRNASeq#cellular-phenotype-data) · [GEO study](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118614) · [ENA study](https://www.ebi.ac.uk/ena/browser/view/PRJNA486209) · [velocyto CLI](https://velocyto.org/velocyto.py/tutorial/cli.html)

SHA-256 of the original 2021 annotation CSV used for the supplied subset: `a69f70ee5ec94c7ff2278bd4f2e37423f9deb3fbf99127d8663ee0ddd0ac288f`.
