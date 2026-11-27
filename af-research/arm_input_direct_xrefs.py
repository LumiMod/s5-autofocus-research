#!/usr/bin/env python3
"""Exact direct-branch encoding search for two already identified ARM entries.

This scans one pinned component for branch candidates, not a whole-firmware
linear disassembly. Hits are decoded with Capstone and bounded caller context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from capstone import Cs,CS_ARCH_ARM,CS_MODE_ARM,CS_MODE_THUMB

from arm_input_boundary import BASE,inspect_block
from extract import require,write_json
from research_paths import add_firmware_arguments,resolve_source

TARGETS=(0x067e03ec,0x067e04ee)


def scan(program):
    half=np.frombuffer(program,dtype='<u2')
    words=np.frombuffer(program,dtype='<u4')
    thumb=np.flatnonzero(((half[:-1]&0xf800)==0xf000)&((half[1:]&0xc000)==0xc000))
    arm=np.flatnonzero(((words&0x0f000000)==0x0b000000)|((words&0xfe000000)==0xfa000000))
    hits=[]
    for mode,indices,stride in [(CS_MODE_THUMB,thumb,2),(CS_MODE_ARM,arm,4)]:
        decoder=Cs(CS_ARCH_ARM,mode);decoder.detail=True
        for index in indices:
            offset=int(index)*stride
            ins=next(decoder.disasm(program[offset:offset+4],BASE+offset,count=1),None)
            if not ins or not ins.operands or ins.operands[0].type!=2:continue
            branch=(ins.mnemonic in ('bl','blx','b.w') if mode==CS_MODE_THUMB else ins.mnemonic.startswith(('bl','blx')))
            if branch and (ins.operands[0].imm&~1) in TARGETS:
                hits.append({'va':hex(ins.address),'file_offset':offset,'bytes_hex':ins.bytes.hex(),
                             'isa':'thumb' if mode==CS_MODE_THUMB else 'arm','mnemonic':ins.mnemonic,
                             'target':hex(ins.operands[0].imm),'operands':ins.op_str})
    pointers={}
    for target in TARGETS:
        for value in (target,target|1):
            pattern=struct.pack('<I',value);offset=program.find(pattern);found=[]
            while offset>=0:
                found.append({'file_offset':offset,'direct_load_va':hex(BASE+offset)})
                offset=program.find(pattern,offset+1)
            pointers[hex(value)]=found
    return {'thumb_branch_candidate_count':len(thumb),'arm_branch_candidate_count':len(arm),
            'direct_branch_hits':hits,'exact_pointer_word_occurrences':pointers}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/recognition-diagnosis/arm-input-direct-callers.json')
    args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite evidence')
    program=(resolve_source(args)/'03_program.bin').read_bytes()
    digest=hashlib.sha256(program).hexdigest()
    previous=json.loads((Path(__file__).parent/'evidence/2026-11-27/runtime-investigation.json').read_text())
    require(digest==previous['component_digests']['03_program.bin'],'Source identity changed')
    result=scan(program)
    require(result['direct_branch_hits']==[{'va':'0x67e0524','file_offset':0x560524,
            'bytes_hex':'fff762ff','isa':'thumb','mnemonic':'bl','target':'0x67e03ec','operands':'#0x67e03ec'}],
            'Direct-call evidence changed')
    result.update(source_file='03_program.bin',source_sha256=digest,source_bytes=len(program),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  targets=[hex(t) for t in TARGETS],
                  caller_block=inspect_block(program,'descriptor_reuse_wrapper',0x067e04ee,0x067e0530,'thumb'),
                  conclusion={
                    'direct_caller_of_067e03ec':'Only0x067e0524 in this component, inside wrapper0x067e04ee.',
                    'descriptor_production':'Wrapper saves its entry r0 in r5, compares/copies the descriptor, then passes the same r5 pointer as r0. It does not populate pixel-format/width/height/stride/ROI/mean/scale fields.',
                    'next_boundary':'Caller/registration of wrapper0x067e04ee is unresolved within03_program.bin: no direct branch or exact pointer-word match to its ARM/Thumb address.',
                    'scope_limit':'Other components, indirect calls, computed pointers and original hardware activity were not searched or executed.'})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps({'output':str(args.output),'direct_hits':result['direct_branch_hits'],
                      'pointer_counts':{k:len(v) for k,v in result['exact_pointer_word_occurrences'].items()}},indent=2))


if __name__=='__main__':main()
