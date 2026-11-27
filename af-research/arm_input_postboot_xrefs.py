#!/usr/bin/env python3
"""Bounded exact callers in two mapped postboot components and one font resource."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from capstone import Cs,CS_ARCH_ARM,CS_MODE_ARM,CS_MODE_THUMB

from arm_input_boundary import BASE
from extract import require,write_json
from research_paths import add_firmware_arguments,resolve_source

TARGETS=(0x067e03ec,0x067e04ee)


def pointer_hits(data):
    result={}
    for value in [target|bit for target in TARGETS for bit in (0,1)]:
        found=[];offset=data.find(struct.pack('<I',value))
        while offset>=0:
            found.append(offset);offset=data.find(struct.pack('<I',value),offset+1)
        result[hex(value)]=found
    return result


def branch_hits(data,base):
    half=np.frombuffer(data,dtype='<u2');words=np.frombuffer(data,dtype='<u4')
    thumb=np.flatnonzero(((half[:-1]&0xf800)==0xf000)&((half[1:]&0xc000)==0xc000))
    arm=np.flatnonzero(((words&0x0f000000)==0x0b000000)|((words&0xfe000000)==0xfa000000))
    hits=[]
    for mode,indices,stride in [(CS_MODE_THUMB,thumb,2),(CS_MODE_ARM,arm,4)]:
        decoder=Cs(CS_ARCH_ARM,mode);decoder.detail=True
        for index in indices:
            offset=int(index)*stride
            ins=next(decoder.disasm(data[offset:offset+4],base+offset,count=1),None)
            if not ins or not ins.operands or ins.operands[0].type!=2:continue
            direct=(ins.mnemonic in ('bl','blx','b.w') if mode==CS_MODE_THUMB else ins.mnemonic.startswith(('bl','blx')))
            if direct and (ins.operands[0].imm&~1) in TARGETS:
                lo=max(0,offset-64);hi=min(len(data),offset+64)
                context=[{'va':hex(i.address),'bytes_hex':i.bytes.hex(),'mnemonic':i.mnemonic,'operands':i.op_str}
                         for i in decoder.disasm(data[lo:hi],base+lo)]
                hits.append({'va':hex(ins.address),'file_offset':offset,'bytes_hex':ins.bytes.hex(),
                             'isa':'thumb' if mode==CS_MODE_THUMB else 'arm','mnemonic':ins.mnemonic,
                             'target':hex(ins.operands[0].imm),'context_start_va':hex(base+lo),
                             'context_raw_hex':data[lo:hi].hex(),'linear_context_candidate':context})
    return {'thumb_candidates':len(thumb),'arm_candidates':len(arm),'direct_branch_hits':hits}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/recognition-diagnosis/arm-input-postboot-callers.json')
    args=parser.parse_args();require(not args.output.exists(),'Refusing to overwrite evidence')
    source=resolve_source(args)
    program=(source/'03_program.bin').read_bytes()
    metadata=json.loads((source/'verification.json').read_text())
    records={r['index']:r for r in metadata['components']}
    require(hashlib.sha256(program).hexdigest()==records[3]['sha256'],'Program identity changed')
    components=[]
    for index in (5,7,20):
        record=records[index];data=(source/record['file']).read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        require(digest==record['sha256'] and len(data)==record['bytes'],'Component identity changed')
        row={'file':record['file'],'source_sha256':digest,'bytes':len(data),'pointer_word_offsets':pointer_hits(data)}
        if index in (5,7):
            table=0x06f52bb0+(index-5)*16
            storage,destination,header,size=struct.unpack_from('<4I',program,table-BASE)
            require(header==destination and size==len(data) and data[:4]==b'LOAD','Unverified load geometry')
            row.update(load_base=hex(destination),load_table_va=hex(table),load_record_raw_hex=program[table-BASE:table-BASE+16].hex(),
                       storage_source=hex(storage),**branch_hits(data,destination))
        else:
            require(data[:9]==b'FDL Font\0','Expected storage font identity changed')
            require(struct.unpack_from('<I',program,0x06867e5c-BASE)[0]==0x41260000,'Font storage literal changed')
            row.update(load_base=None,storage_source='0x41260000',branch_scan_skipped=True,
                       reason='Existing loader evidence identifies an FDL Font storage resource, without a permanent ARM executable load base; no base is guessed.')
        components.append(row)
    report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'program_sha256':hashlib.sha256(program).hexdigest(),'targets':[hex(t) for t in TARGETS],
            'scope':'Only05_postboot1,07_postboot3 and20_osdover; no inference, training, QEMU or firmware execution.',
            'components':components,
            'conclusion':{'direct_branch_hit_count':sum(len(c.get('direct_branch_hits',[])) for c in components),
                          'pointer_word_hit_count':sum(sum(len(v) for v in c['pointer_word_offsets'].values()) for c in components),
                          'format_fields_confirmed':False,
                          'remaining_boundary':'Interpret exact hits only after validating nearby instruction boundaries; negative results do not exclude indirect/computed pointers or other components.'}}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report)
    print(json.dumps({'output':str(args.output),'conclusion':report['conclusion'],
                      'hits':{c['file']:[{k:v for k,v in h.items() if k not in ('context_raw_hex','linear_context_candidate')}
                                      for h in c.get('direct_branch_hits',[])] for c in components}},indent=2))


if __name__=='__main__':main()
