"""Record package versions and CPU/thread policy for a recomputation."""
import hashlib, importlib.metadata, json, os, platform, subprocess
from pathlib import Path

def main():
 d={'python':platform.python_version(),'platform':platform.platform(),
    'cpu_logical':os.cpu_count(),'cpu_model':platform.processor(),
    'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
    'thread_policy':'RF n_jobs=-1; other estimator thread arguments unset; library defaults apply'}
 d['configuration_policy']='get_params() records constructor arguments. CatBoost returns explicit settings rather than all resolved defaults; omitted settings use the pinned library defaults.'
 d['categorical_policy']='Numeric codes are supplied to all learners. Tree categorical options are not enabled; TabPFN categorical_features_indices=None retains automatic fitting-data inference.'
 if os.name=='nt':
  result=subprocess.run(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],capture_output=True,text=True,check=True)
  d['cpu_model']=result.stdout.strip()
 for name in ['numpy','pandas','scikit-learn','lightgbm','xgboost','catboost','scipy','torch','tabpfn','shap','pyarrow','psutil','pyodbc','matplotlib','python-docx']:
  try:d[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:d[name]='not installed'
 import psutil
 d['ram_gb']=round(psutil.virtual_memory().total/1e9,1)
 try:
  import torch
  d['torch_distribution_metadata']=d['torch']
  d['torch']=str(torch.__version__)
  d['torch_cuda_runtime']=torch.version.cuda
  d['cuda']=torch.cuda.is_available()
 except ImportError:d['cuda']=False
 # Inspect existing cached checkpoint files without downloading or modifying weights.
 candidates=list((Path(os.environ['APPDATA'])/'tabpfn').rglob('*.ckpt')) if os.environ.get('APPDATA') else []
 candidates+=list((Path.home()/'.cache/tabpfn').rglob('*.ckpt'))
 d['cached_checkpoints']=[{'filename':p.name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in set(candidates)]
 try:
  from tabpfn.model_loading import resolve_model_version,resolve_model_path
  version=resolve_model_version(None)
  paths,_,_,_=resolve_model_path(None,'regressor',version=version.value)
  d['default_regressor_checkpoint']={
    'resolved_version':version.value,'constructor_model_path':'auto',
    'files':[{'filename':p.name,'cached':p.exists(),
              'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if p.exists() else None} for p in paths]}
 except ImportError:d['default_regressor_checkpoint']='TabPFN unavailable in tree-only environment'
 d['historical_checkpoint']='No checksum was recorded for the audited historical run; current cache identity cannot establish past identity.'
 Path('panel/environment.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
 from run_comparators import models
 config={name:model.get_params() for name,model in models().items()}
 try:
  from tabpfn import TabPFNRegressor
  config['TabPFN-3']=TabPFNRegressor(n_estimators=8,auto_scale_n_estimators=False,random_state=0,ignore_pretraining_limits=True).get_params()
 except ImportError:config['TabPFN-3']={'unavailable':'package not installed; tree-only execution'}
 Path('panel/model_configurations.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
 print('Saved environment.json and model_configurations.json')
if __name__=='__main__':main()
