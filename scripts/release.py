"""Build from a committed, detached archive; never deploys models implicitly."""
import argparse,hashlib,json,os,pathlib,subprocess,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def run(args,**kw):return subprocess.run(args,check=True,**kw)
def sha_tree(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
def main():
 meta=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:deploy/hub/components.json'],text=True));p=argparse.ArgumentParser();p.add_argument('mode',choices=['build']);p.add_argument('--component',choices=list(meta['components']),required=True);p.add_argument('--work-dir',type=pathlib.Path,required=True);p.add_argument('--launcher',required=True);a=p.parse_args()
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();work=a.work_dir.resolve();work.mkdir(parents=True,exist_ok=True);source=work/'source'
 if source.exists():
  assert (source/'.source-commit').read_text().strip()==commit,'snapshot belongs to another commit'
  original=json.loads((work/'snapshot-sha256.json').read_text());assert sha_tree(source)==original,'frozen input changed'
 else:
  archive=work/'source.tar';run(['git','-C',str(ROOT),'archive','--format=tar','--output',str(archive),commit]);source.mkdir()
  with tarfile.open(archive) as t:t.extractall(source,filter='data')
  (source/'.source-commit').write_text(commit+'\n');original=sha_tree(source);(work/'snapshot-sha256.json').write_text(json.dumps(original,indent=2))
  for f in source.rglob('*'):
   if f.is_file():f.chmod(0o555 if os.access(f,os.X_OK) else 0o444)
  for d in sorted([x for x in source.rglob('*') if x.is_dir()],reverse=True):d.chmod(0o555)
  source.chmod(0o555)
 env=os.environ.copy();env['ANSIBLE_PIPELINING']='1'
 args=[a.launcher,str(source/'deploy/hub/build.yml'),'-i',str(source/'deploy/hub/inventory.yml'),'-e',json.dumps({'provider_project':meta['project'],'provider_components':list(meta['components']),'component':a.component,'source_path':str(source),'evidence_path':str(work)})]
 run(args,env=env);assert sha_tree(source)==original,'source snapshot mutated during build'
 print(json.dumps({'sourceCommit':commit,'component':a.component,'evidence':str(work)}))
if __name__=='__main__':main()
