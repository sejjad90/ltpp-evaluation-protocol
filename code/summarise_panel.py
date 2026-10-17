"""Emit auditable CSV diagnostics from the assembled parquet and raw SDR tables."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import pyodbc
from splits import load, feature_matrix, splits_temporal, calibration_split

ROOT=Path(__file__).resolve().parent
if ROOT.name=='code':ROOT=ROOT.parent

def raw(module,db,table,columns='*'):
 p=Path(os.environ.get('LTPP_RAW_DIR',ROOT/'extracted'))/f'SDR39 - {module}'/f'{db}.accdb'
 with pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+str(p)) as cn:
  d=pd.read_sql(f'SELECT {columns} FROM [{table}]',cn)
 for c in ['SHRP_ID','VWS_ID']:
  if c in d:d[c]=d[c].astype(str).str.strip()
 return d

def main():
 os.chdir(ROOT)
 p=pd.read_parquet('panel/ltpp_panel.parquet');m=load();X,_=feature_matrix(m)
 e=p.loc[~p.index.isin(m.index)];sec=p.drop_duplicates('section_id')
 tr,te=splits_temporal(m,cutoff=2013,n_splits=3,seed=0)[0];cal,fit=calibration_split(tr)
 d={'n_panel':len(p),'n_sections':p.section_id.nunique(),'n_units':p.unit_id.nunique(),
    'n_model':len(m),'n_model_sec':m.section_id.nunique(),'n_features':X.shape[1],
    'date_min':str(p.VISIT_DATE.min().date()),'date_max':str(p.VISIT_DATE.max().date()),
    'clim_dist':sec.clim_dist_km.median(),'clim_onsite':100*sec.clim_onsite.mean(),
    'clim_p90':sec.clim_dist_km.quantile(.90),'clim_p95':sec.clim_dist_km.quantile(.95),
    'clim_max':sec.clim_dist_km.max(),'pre_oct_pct':100*(p.VISIT_DATE.dt.month<10).mean(),
    'n_excl_first':int((p.visit_seq==0).sum()),'n_excluded':len(e),
    'n_excl_other':int((e.visit_seq>0).sum()),'n_states':p.STATE_CODE.nunique(),
    'inc_state_gap':(m.STATE_CODE.value_counts(normalize=True)-e.STATE_CODE.value_counts(normalize=True)).abs().max(),
    'cost_ntrain':len(fit),'cost_ncal':len(cal),'cost_ntest':len(te),
    'cum_kesal_missing':p.cum_kesal.isna().sum()}
 # Save each jurisdiction's diagnostic, allowing the loader to select the
 # weakest geographic score without embedding a hard-coded jurisdiction.
 for state,g in m.groupby('STATE_CODE'):
  spans=g.groupby('section_id').VISIT_DATE.agg(['min','max'])
  z=f'state_{state}_'
  d.update({z+'rows':len(g),z+'sections':g.section_id.nunique(),
    z+'rigid_pct':100*(g.thk_PC.fillna(0)>0).mean(),
    z+'span':((spans['max']-spans['min']).dt.days/365.25).median(),
    z+'temp':g.groupby('section_id').t12_mean_temp.median().median(),
    z+'precip':g.groupby('section_id').t12_precip.median().median(),
    z+'freeze':g.groupby('section_id').t12_freeze_thaw.median().median()})
 pd.DataFrame([{'key':k,'value':v,'source':'panel/ltpp_panel.parquet; splits.py'} for k,v in d.items()]).to_csv('panel/panel_summary.csv',index=False)
 link=raw('CLIMATE_SUMMARY_DATA','Climate_Summary_Data','CLM_SITE_VWS_LINK','STATE_CODE, SHRP_ID, VWS_ID')
 K=['STATE_CODE','SHRP_ID','CONSTRUCTION_NO']
 iri=raw('MONITORING','Monitoring','ANALYSIS_IRI','STATE_CODE, SHRP_ID, CONSTRUCTION_NO, VISIT_DATE, MRI')
 iri=iri[iri.MRI.notna()&(iri.MRI>0)].groupby(K+['VISIT_DATE'],as_index=False).MRI.mean()
 age=raw('INVENTORY','Inventory','INV_AGE','STATE_CODE, SHRP_ID, CONSTRUCTION_NO, TRAFFIC_OPEN_DATE, CONSTRUCTION_DATE')
 age['open']=pd.to_datetime(age.TRAFFIC_OPEN_DATE).fillna(pd.to_datetime(age.CONSTRUCTION_DATE))
 lay=raw('INVENTORY','Inventory','INV_LAYER','STATE_CODE, SHRP_ID, CONSTRUCTION_NO, MEAN_THICKNESS')
 lay=lay[lay.MEAN_THICKNESS.notna()][K].drop_duplicates()
 restricted=iri.merge(age[K+['open']],on=K).merge(lay,on=K)
 restricted=restricted[pd.to_datetime(restricted.VISIT_DATE)>=restricted.open]
 vals={'vws_sections':len(link[['STATE_CODE','SHRP_ID']].drop_duplicates()),'vws_stations':link.VWS_ID.nunique(),
       'restricted_rows':len(restricted),'restricted_sections':restricted.groupby(['STATE_CODE','SHRP_ID']).ngroups}
 pd.DataFrame([{'key':k,'value':v,'source':'SDR39 CLM_SITE_VWS_LINK / ANALYSIS_IRI + INV_AGE + nonmissing INV_LAYER, nonnegative age'} for k,v in vals.items()]).to_csv('panel/raw_summary.csv',index=False)
 print(pd.Series(vals).to_string());print('Saved panel_summary.csv and raw_summary.csv')
 from summarise_splits import main as split_summary
 split_summary()

if __name__=='__main__':main()
