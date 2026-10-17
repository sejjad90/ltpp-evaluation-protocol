"""Write aggregate split sizes/overlaps; does not export membership manifests."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from splits import load,PROTOCOLS,splits_temporal,calibration_split

def main():
 df=load();rows=[]
 digest=hashlib.sha256(Path('panel/ltpp_panel.parquet').read_bytes()).hexdigest()
 def add(experiment,protocol,seed,cutoff,fold,tr,te,cal=None,fit=None):
  if fit is None:fit=tr
  if cal is None:cal=np.array([],dtype=int)
  sets=[set(df.iloc[a].section_id) for a in [fit,cal,te]]
  r=dict(experiment=experiment,protocol=protocol,seed=seed,cutoff=cutoff,fold=fold,
         n_train_partition=len(tr),n_fit=len(fit),n_cal=len(cal),n_test=len(te),
         fit_cal_section_overlap=len(sets[0]&sets[1]),
         fit_test_section_overlap=len(sets[0]&sets[2]),
         cal_test_section_overlap=len(sets[1]&sets[2]),panel_sha256=digest)
  for label,idx in [('train_partition',tr),('fit',fit),('cal',cal),('test',te)]:
   g=df.iloc[idx]
   r[label+'_sections']=g.section_id.nunique();r[label+'_events']=g.unit_id.nunique()
   r[label+'_year_min']=g.VISIT_DATE.dt.year.min() if len(g) else np.nan
   r[label+'_year_max']=g.VISIT_DATE.dt.year.max() if len(g) else np.nan
  rows.append(r)
 for protocol,fn in PROTOCOLS.items():
  for fold,(tr,te) in enumerate(fn(df)):
   cal,fit=calibration_split(tr,seed=0)
   add('principal',protocol,0,2015 if protocol.endswith('forward-in-time') else np.nan,fold,tr,te,cal,fit)
 for seed in [0,1,2]:
  for cut in [2010,2013,2016]:
   for fold,(tr,te) in enumerate(splits_temporal(df,cutoff=cut,n_splits=3,seed=seed)):
    cal,fit=calibration_split(tr,seed=seed)
    add('repeated grid','section-disjoint + forward-in-time',seed,cut,fold,tr,te,cal,fit)
 out=pd.DataFrame(rows)
 # Check actual principal fit logs when available; never invent missing records.
 for name in ['principal_tree_folds.csv','principal_tabpfn_folds.csv']:
  p=Path('panel')/name
  if p.exists():
   measured=pd.read_csv(p)
   assert not measured.duplicated(['model','protocol','fold']).any(),name
   assert len(measured)==15*measured.model.nunique(),name
   a=measured.merge(out[out.experiment=='principal'][['protocol','fold','n_fit','n_cal','n_test']],on=['protocol','fold'],how='left',validate='many_to_one',suffixes=('_measured','_reconstructed'))
   assert len(a)==len(measured) and a.n_test_reconstructed.notna().all(),name
   for field in ['n_fit','n_cal','n_test']:assert (a[field+'_measured']==a[field+'_reconstructed']).all()
 for name in ['robustness_trees.csv','robustness_tabpfn.csv']:
  p=Path('panel')/name
  if p.exists():
   measured=pd.read_csv(p)
   assert not measured.duplicated(['model','seed','cutoff','fold']).any(),name
   assert len(measured)==27*measured.model.nunique(),name
   a=measured.merge(out[out.experiment=='repeated grid'][['seed','cutoff','fold','n_test']],on=['seed','cutoff','fold'],how='left',validate='many_to_one',suffixes=('_measured','_reconstructed'))
   assert len(a)==len(measured) and a.n_test_reconstructed.notna().all(),name
   assert (a.n_test_measured==a.n_test_reconstructed).all(),name
 assert len(out)==42 and not out.duplicated(['experiment','protocol','seed','cutoff','fold']).any()
 out.to_csv('panel/split_summary.csv',index=False)
 print(f'Saved {len(out)} principal/grid split summaries; exact membership manifests remain absent.')
if __name__=='__main__':main()
