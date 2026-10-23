# Scripts

`table1.py` computes the statistics of Table 1 from evaluation outputs written by
`drl_pipeline/evaluate.py`:

```bash
python scripts/table1.py drl_pipeline/outputs/evaluation/*
```

For each evaluation directory it averages over t⁺ ∈ [500, 3000] of every trajectory

- DR = 1 − ⟨∂u/∂y|_wall⟩ / 7.9, in percent;
- E_act = ½⟨|v_w|³⟩ / u_τ³, multiplied by 10 as in the paper;

and reports the mean and standard deviation over trajectories and R̄ = DR − E_act of the means.
