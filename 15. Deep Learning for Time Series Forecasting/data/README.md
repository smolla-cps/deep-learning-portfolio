# Data

`synthetic_hourly_demand.csv` is a reproducible hourly forecasting dataset created for this portfolio section.

Columns:

- `timestamp`
- `demand`
- `temperature`
- `humidity`

The target demand contains:

- trend,
- daily seasonality,
- weekly seasonality,
- exogenous temperature/humidity relationships,
- random noise.

The dataset is intentionally small enough for notebook-based model comparison while still containing several temporal structures that different neural architectures can learn.

The same methodology can later be transferred to real energy, manufacturing, healthcare, demand, weather, or sensor forecasting data.
