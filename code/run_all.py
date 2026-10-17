"""Run analyses from raw SDR 39, or render the supplied measured summaries."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'code'
PANEL=ROOT/'panel'
STAGES=[
 ('build_panel.py',[],['ltpp_panel.parquet'],False),
 ('run_comparators.py',[],['comparator_results.csv','principal_tree_folds.csv'],False),
 ('run_tabpfn.py',[],['tabpfn3_results.csv','principal_tabpfn_folds.csv'],True),
 ('run_robustness.py',['--models','tree','--out','panel/robustness_trees.csv'],['robustness_trees.csv'],False),
 ('run_robustness.py',['--models','tabpfn','--out','panel/robustness_tabpfn.csv'],['robustness_tabpfn.csv'],True),
 ('run_decomposition.py',[],['decomposition_folds.csv','decomposition_summary.csv'],False),
 ('run_tabpfn_designs.py',[],['decomposition_tabpfn.csv','review_designs_tabpfn.csv'],True),
 ('run_review_experiments.py',[],['review_designs.csv','calibration_curve.csv','residuals.csv','feature_importance.csv'],False),
 ('run_ablation.py',[],['ablation_initial_mri.csv'],False),
 ('run_conformal_methods.py',[],['conformal_methods.csv'],False),
 ('run_cost_zones.py',[],['computational_cost.csv','heterogeneity.csv'],False),
 ('run_tuning.py',[],['tuning_nested.csv','tuning_sensitivity.csv'],False),
 ('run_inclusion.py',[],['inclusion_numeric.csv','inclusion_structural.csv','inclusion_flow.csv'],False),
 ('summarise_panel.py',[],['panel_summary.csv','raw_summary.csv','split_summary.csv'],False),
]

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--trees-only',action='store_true',help='fit only tree learners; omit reference comparisons')
 ap.add_argument('--results-only',action='store_true',help='stage shipped summaries and render figures without raw data or refitting')
 ap.add_argument('--force',action='store_true',help='rerun all fits, ignoring existing working outputs')
 ap.add_argument('--fetch',action='store_true',help='download and extract SDR 39 first')
 ap.add_argument('--raw-dir',type=Path,help='directory containing extracted SDR39 module folders')
 a=ap.parse_args()
 if a.results_only and a.force:ap.error('--force and --results-only are incompatible')
 PANEL.mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
 env={**os.environ,'PYTHONIOENCODING':'utf-8','LTPP_TREES_ONLY':str(int(a.trees_only))}
 if a.raw_dir:env['LTPP_RAW_DIR']=str(a.raw_dir.resolve())
 def run(script,args=()):
  print('[run]',script,*args,flush=True);start=time.monotonic()
  subprocess.run([sys.executable,'-u',str(CODE/script),*args],cwd=ROOT,env=env,check=True)
  print('[done]',script,f'{time.monotonic()-start:.1f}s',flush=True)
 if a.fetch:run('fetch_sdr39.py')
 if a.results_only:
  for p in (ROOT/'results').glob('*'):
   if p.is_file() and p.suffix in ('.csv','.json'):shutil.copy2(p,PANEL/p.name)
  print('Staged shipped summaries. No model fits performed.',flush=True)
 else:
  raw=Path(env.get('LTPP_RAW_DIR',str(ROOT/'extracted')))
  if not raw.is_dir():raise SystemExit('Raw modules absent. Run with --fetch, or use --raw-dir EXTRACTED. See README.')
  for script,args,markers,tab in STAGES:
   if a.trees_only and tab:print('[skip]',script,'TabPFN stage',flush=True);continue
   if not a.force and all((PANEL/p).exists() for p in markers):print('[have]',script,flush=True);continue
   run(script,[*args,*(['--trees-only'] if a.trees_only and script=='run_cost_zones.py' else [])])
  run('record_environment.py')
 run('merge_tabpfn_results.py')
 if not a.trees_only and (PANEL/'robustness_tabpfn.csv').exists():run('run_cluster_stats.py')
 else:print('[skip] TabPFN reference statistics and paired histogram are unavailable in tree-only mode.',flush=True)
 for script in ['make_fig1.py','make_figures.py','make_figures2.py','make_figures3.py']:run(script)
 if not a.trees_only:run('paper_numbers.py')
 print('Complete. Analysis outputs: panel/; rendered figures: figures/. Document builders are not included.',flush=True)

if __name__=='__main__':main()
