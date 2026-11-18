#!/usr/bin/env bash
# From a checkout or the standalone kit; RETINA_WORK is shared with the notebook.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${RETINA_WORK:=$HOME/projects/clark-retina}"
action=${1:-help}
shift || true
if [[ "$action" == reference ]]; then
  ref="$RETINA_WORK/ref"
  mkdir -p "$ref"
  if [[ ! -s "$ref/genes.gtf" ]]; then
    archive="$ref/mm10-1.2.0.tar.gz"
    if [[ ! -f "$archive" ]]; then
      curl --fail --location --retry 3 --continue-at - \
        https://cf.10xgenomics.com/supp/cell-exp/refdata-cellranger-mm10-1.2.0.tar.gz \
        -o "$archive.part"
      mv "$archive.part" "$archive"
    fi
    listing=$(tar -tzf "$archive")
    mapfile -t genes < <(awk '/\/genes\/genes\.gtf(\.gz)?$/' <<< "$listing")
    [[ "${#genes[@]}" -eq 1 ]] || { echo "Expected one genes.gtf in the reference." >&2; exit 2; }
    if [[ "${genes[0]}" == *.gz ]]; then
      tar -xzOf "$archive" "${genes[0]}" | gzip -dc > "$ref/genes.gtf.tmp"
    else
      tar -xzOf "$archive" "${genes[0]}" > "$ref/genes.gtf.tmp"
    fi
    test -s "$ref/genes.gtf.tmp"
    mv "$ref/genes.gtf.tmp" "$ref/genes.gtf"
  fi
  echo "$ref/genes.gtf"
  exit 0
fi
case "$action" in
  download|barcodes|velocity) ;;
  *) echo "Usage: bash retina.sh {reference|download|barcodes|velocity} [E16 ...]" >&2; exit 2 ;;
esac
mkdir -p "$RETINA_WORK"/{bam,barcodes,looms}
if (( $# )); then
  samples=("$@")
else
  mapfile -t samples < <(awk -F '\t' 'NR>1 {print $1}' "$here/samples.tsv")
fi
for sample in "${samples[@]}"; do
  line=$(awk -F '\t' -v s="$sample" 'NR>1 && $1==s' "$here/samples.tsv")
  [[ -n "$line" ]] || { echo "Unknown library: $sample" >&2; exit 2; }
  IFS=$'\t' read -r name gsm run author age count url md5 size <<< "$line"
  bam="$RETINA_WORK/bam/$sample.bam"
  barcodes="$RETINA_WORK/barcodes/$sample.tsv"
  case "$action" in
    download)
      if [[ ! -f "$bam" ]]; then
        curl --fail --location --retry 3 --continue-at - "$url" -o "$bam.part"
        printf '%s  %s\n' "$md5" "$bam.part" | md5sum --check -
        mv "$bam.part" "$bam"
      else
        printf '%s  %s\n' "$md5" "$bam" | md5sum --check -
      fi ;;
    barcodes)
      gzip -dc "$here/cells.tsv.gz" | awk -F '\t' -v s="$sample" 'NR>1 && $1==s {print $4}' > "$barcodes.tmp"
      [[ "$(wc -l < "$barcodes.tmp")" -eq "$count" ]]
      mv "$barcodes.tmp" "$barcodes" ;;
    velocity)
      : "${RETINA_GTF:=$RETINA_WORK/ref/genes.gtf}"
      [[ -s "$RETINA_GTF" && -s "$barcodes" ]] || { echo "Check the GTF and run barcodes first." >&2; exit 2; }
      samtools quickcheck -v "$bam"
      sorted="$RETINA_WORK/bam/cellsorted_$sample.bam"
      if [[ -e "$sorted" ]]; then samtools quickcheck -v "$sorted"; fi
      if [[ -e "$RETINA_WORK/looms/$sample.loom" ]]; then
        echo "Already present: $sample.loom"; continue
      fi
      # Use a separate output directory so a failed run cannot appear complete.
      tmp=$(mktemp -d "$RETINA_WORK/looms/$sample.XXXXXXXX")
      velocyto run -e "$sample" -b "$barcodes" -o "$tmp" \
        --samtools-threads "${SORT_THREADS:-4}" --samtools-memory "${SORT_MB_PER_THREAD:-1000}" \
        "$bam" "$RETINA_GTF"
      [[ -s "$tmp/$sample.loom" ]]
      mv "$tmp/$sample.loom" "$RETINA_WORK/looms/$sample.loom"
      rmdir "$tmp" 2>/dev/null || true ;;
  esac
done
