# rc2 validation evidence

The JSON/CSV files and logs record the checks performed for this source candidate.
Notebook outputs are embedded in tutorials/ and copied byte-for-byte to docs/tutorials/.

To replay the independent full-graph probability and ordering-flux checks, unpack
the previously provided scCS_Audit_Evidence.zip, then run:

```bash
python release_checks/replay_graphs.py --input-dir /path/to/evidence/replay --output-dir release_checks
```

The replay inputs preserve complete cell graphs and a compact expression panel.
They validate scoring on audited graphs; the fresh fitting logs separately record
new public-data preprocessing. Schwann kinetic fitting and velocity calculation
were separated by a lossless H5AD checkpoint to reduce peak process memory.
All biological notebooks then reused the matching shared preprocessing fits.
