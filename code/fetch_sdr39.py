"""Download and extract SDR 39 module ZIPs to repository-relative directories."""
from pathlib import Path
import json, re, urllib.request, urllib.parse, zipfile
ROOT=Path(__file__).resolve().parent
if ROOT.name=='code':ROOT=ROOT.parent
IDS=[3290,3271,3272,3239,3273,3229,3284,3302,3291,3277,3275,3276,3274,3232,3226,3244,3245,3238,3263,3256]
def get(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=180)
def extract_archive(path,destination):
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path) as archive:
  for member in archive.infolist():
   if not (destination/member.filename).resolve().is_relative_to(destination.resolve()):raise ValueError('Archive path outside extracted/')
  archive.extractall(destination)
def main():
 z=ROOT/'zips';out=ROOT/'extracted';z.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
 for i in IDS:
  with get(f'https://infopave.fhwa.dot.gov/DownloadTracker/SDR/Volume/{i}?viewDownload=inline') as r:body=r.read().decode('utf-8')
  match=re.search(r'"FilePath"\s*:\s*("(?:[^"\\]|\\.)*")',body)
  if not match:raise RuntimeError(f'FHWA did not supply a file URL for module {i}')
  url=json.loads(match[1]).strip();p=z/Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
  if not p.exists():
   print('Download',p.name,flush=True)
   partial=p.with_suffix(p.suffix+'.part')
   with get(url.replace(' ','%20')) as r,partial.open('wb') as f:
    import shutil;shutil.copyfileobj(r,f)
   if not zipfile.is_zipfile(partial):raise RuntimeError(f'Download is not a ZIP: {p.name}')
   partial.replace(p)
  extract_archive(p,out)
  print('Extracted',p.name,flush=True)
if __name__=='__main__':main()
