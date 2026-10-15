# Non-regenerable figure assets

## `main_workflow_rev2.pdf` — Main Fig. 1

The workflow schematic is laid out by hand in vector-editing software rather than produced by a plotting
script. The adopted export is the version wired into the manuscript, so it is versioned here as the
authoritative asset rather than regenerated.

SHA256: `b87bc6cb34f01c69315fb38dd81af97cf012520f13a670ccf4d73b475f995373`

Layout notes for anyone editing it: panel **c** deliberately shows both the temporal contrastive loss and
the ED-teacher → pocket-student constraint loss; the schematic's `T` runs opposite to the sampling time
`t` in Methods (`T = 1 - t`). Baked-in labels ``Amplification Pipeline'' and ``Constraint Transfer'' are
quoted in the manuscript caption, so renaming them in the art requires updating the caption too.

## Figures that *are* regenerated from data

Everything else is produced by the scripts listed in `reproduction/scripts/README.md`. If a figure can be
regenerated, do not add a hand-exported copy here — it will drift from the data.
