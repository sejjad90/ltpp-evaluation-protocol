"""Annual event-specific ESAL exposure through the last completed calendar year."""
import numpy as np
import pandas as pd

KEY = ['STATE_CODE', 'SHRP_ID', 'CONSTRUCTION_NO']

def cumulative_loading(visits, traffic):
    """Return thousand ESALs and an event/year provenance table.

    Sum ANNUAL_ESAL_TREND for this construction event from OPEN_DATE's year
    through survey year minus one. The event-specific construction-year record
    is used as supplied, without invented within-year prorating. Missing annual
    records/values make the cumulative exposure unknown, rather than zero.
    Surveys in the construction year have an empty completed-year interval (0).
    No traffic record for the survey year or a later year contributes.
    """
    annual = traffic.groupby(KEY + ['YEAR'], as_index=False)['ANNUAL_ESAL_TREND'].mean()
    groups = {k: g.set_index('YEAR')['ANNUAL_ESAL_TREND'] for k,g in annual.groupby(KEY)}
    out = pd.Series(np.nan, index=visits.index, dtype=float)
    ledger = []
    for k,g in visits.groupby(KEY):
        origin = g.OPEN_DATE.iloc[0]
        if pd.isna(origin): continue
        first = int(origin.year)
        a = groups.get(k, pd.Series(dtype=float))
        for year, rows in g.groupby(g.VISIT_DATE.dt.year):
            years = list(range(first, int(year)))
            values = a.reindex(years)
            known = int(values.notna().sum())
            value = float(values.sum()/1000) if known == len(years) else np.nan
            out.loc[rows.index] = value
            ledger.append(dict(zip(KEY,k)) | {
                'construction_year':first,'survey_year':int(year),
                'last_traffic_year':int(year)-1,'expected_annual_records':len(years),
                'known_annual_records':known,'cum_kesal':value})
    return out, pd.DataFrame(ledger)
