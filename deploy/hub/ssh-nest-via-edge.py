#!/usr/bin/env python3
"""Reuse the established Edge->Nest trust without copying a private key."""
import os,shlex,sys
args=sys.argv[1:]
known={'ttd-nest','root@ttd-nest','ttd-nest.ttd','root@ttd-nest.ttd','192.168.100.14','root@192.168.100.14'}
target=next((i for i,a in enumerate(args) if a in known),None)
if target is None:
    if args==['-V']:os.execvp('ssh',['ssh','-V'])
    raise SystemExit('This transport is restricted to root@ttd-nest')
remote=['ssh','-T','-o','BatchMode=yes','-o','ControlMaster=no','root@ttd-nest',*args[target+1:]]
os.execvp('ssh',['ssh','-T','-o','BatchMode=yes','root@ttd-edge',shlex.join(remote)])
