#!/usr/bin/env python3
"""Bounded static ARM consumer-chain evidence; no execution or XM4 decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import capstone
from capstone import Cs, CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB

from extract import require, write_json
from research_paths import add_firmware_arguments, resolve_source

BASE = 0x06280000
BLOCKS = [
    ('registration_caller_and_callback',0x06797638,0x067976be,'thumb'),
    ('registration_fixed_memory_descriptor',0x067e034c,0x067e036a,'thumb'),
    ('registration_command0_wrapper',0x067d1770,0x067d17aa,'thumb'),
    ('command_ring_copy_and_enqueue',0x067d170c,0x067d1770,'thumb'),
    ('neighbor_command4_wrapper_not_a_verified_inference_caller',0x067d1830,0x067d1872,'thumb'),
    ('CDNN_model_loader',0x067e0702,0x067e07c4,'thumb'),
    ('model_pointer_getters',0x067e0822,0x067e0854,'thumb'),
    ('opaque_descriptor_submit',0x067e03ec,0x067e04ba,'thumb'),
    ('descriptor_comparison_and_copy',0x067e0530,0x067e0594,'thumb'),
    ('driver_submit_wrapper',0x06776f78,0x06776fae,'thumb'),
    ('driver_fixed_RAM_object_getter',0x065c6914,0x065c6918,'thumb'),
    ('driver_constructor_caller',0x065c697c,0x065c69e2,'thumb'),
    ('driver_constructor_fixed_RAM_function_table',0x067b64ee,0x067b64fe,'thumb'),
    ('driver_pointer_store_and_indirect_dispatch',0x065c69e2,0x065c6a58,'thumb'),
    ('memcpy_thumb_interworking',0x066c2598,0x066c259c,'thumb'),
    ('memcpy_ARM_body',0x066c259c,0x066c2618,'arm'),
]
EXPECTED = {
    0x067d1790:'f0f604ef',  # copy64 descriptor
    0x067d1732:'f0f634ef',  # copy92 envelope
    0x067e046e:'96f783fd',  # opaque descriptor submit
    0x06776fa8:'4ff61bfd',  # wrapper -> driver
    0x065c6a32:'8847',      # BLX runtime table slot
    0x067e056c:'b4ee600a',  # float equality, not subtraction
    0x067b64f6:'0448',      # fixed RAM function table literal
}


def inspect_block(program,name,start,end,mode):
    blob = program[start-BASE:end-BASE]
    decoder = Cs(CS_ARCH_ARM,CS_MODE_THUMB if mode=='thumb' else CS_MODE_ARM)
    decoder.detail = True
    rows=[]
    for instruction in decoder.disasm(blob,start):
        row={'va':hex(instruction.address),'file_offset':instruction.address-BASE,
             'bytes_hex':instruction.bytes.hex(),'mnemonic':instruction.mnemonic,
             'operands':instruction.op_str}
        ops=instruction.operands
        if instruction.mnemonic.startswith('ldr') and len(ops)>1 and ops[1].type==3 and decoder.reg_name(ops[1].mem.base)=='pc':
            literal=((instruction.address+(4 if mode=='thumb' else 8))&~3)+ops[1].mem.disp
            if 0<=literal-BASE<=len(program)-4:
                row['pc_literal']={'va':hex(literal),'file_offset':literal-BASE,
                                   'u32':hex(struct.unpack_from('<I',program,literal-BASE)[0])}
        rows.append(row)
    return {'name':name,'start_va':hex(start),'end_va_exclusive':hex(end),'file_offset':start-BASE,
            'isa':mode,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
            'raw_hex':blob.hex(),'decoded_bytes':sum(len(bytes.fromhex(r['bytes_hex'])) for r in rows),
            'instructions':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/recognition-diagnosis/arm-input-boundary.json')
    args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite an evidence report')
    program=(resolve_source(args)/'03_program.bin').read_bytes()
    previous=json.loads((Path(__file__).parent/'evidence/2026-11-27/runtime-investigation.json').read_text())
    digest=hashlib.sha256(program).hexdigest()
    require(digest==previous['component_digests']['03_program.bin'],'Program differs from previous runtime investigation')
    for address,expected in EXPECTED.items():
        require(program[address-BASE:address-BASE+len(expected)//2].hex()==expected,'Instruction bytes changed at '+hex(address))
    blocks=[inspect_block(program,*b) for b in BLOCKS]
    require(all(b['decoded_bytes']==b['bytes'] for b in blocks),'Incomplete instruction coverage inside selected function ranges')
    report={
        'source_file':'03_program.bin','source_sha256':digest,'load_base':hex(BASE),
        'component_bytes':len(program),'direct_mapping_end_exclusive':hex(BASE+len(program)),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'capstone_version':capstone.__version__,
        'scope':{'kind':'static only','selected_blocks':len(blocks),'selected_code_bytes':sum(b['bytes'] for b in blocks),
                 'whole_firmware_disassembly':False,'target_execution':False,'model_training':False},
        'observations':[
            {'id':'ARM-REG-64','addresses':['0x0679764c','0x0679766a','0x06797686','0x06797692','0x067976ae'],
             'fact':'Caller assembles a64-byte registration descriptor containing two model pointers and0x400000 lengths. Six initial words come from0x067e034c, which writes fixed mapping addresses/lengths.',
             'fixed_words':['0x7b140000','0x83e90000','0x10000','0x7b100000','0x83ea0000','0x40000'],
             'limit':'These are initialization/memory-descriptor values, not observed camera pixel samples or proof of an input tensor format.'},
            {'id':'ARM-REG-COPY','addresses':['0x067d1770','0x067d178a','0x067d1790','0x067d179e','0x066c259c'],
             'fact':'Command0 wrapper copies exactly0x40 bytes unchanged into a0x5c-byte command at+4, appends two callback/context words, and calls0x067d170c. ARM memcpy body contains loads/stores and count arithmetic.',
             'limit':'No pixel conversion, mean subtraction or normalization is performed in this selected wrapper or copy helper.'},
            {'id':'ARM-RING-92','addresses':['0x067d171e','0x067d1732','0x067d1746','0x067d1758','0x067d175e'],
             'fact':'Submitter copies the0x5c-byte command to a5-slot ring at0x09111464 and passes its address to0x06738de4 through a handle loaded from0x06f498f4.',
             'limit':'The queue consumer and its dispatch to CDNN/XM4 were not identified by this bounded trace; this is not an observed DSP invocation.'},
            {'id':'ARM-OPAQUE-DESC','addresses':['0x067e042c','0x067e0432','0x067e0438','0x067e0458','0x067e046e','0x06776f78','0x06776fa8'],
             'fact':'Nearby submit path checks descriptor halfwords+0/+2 and word+0x20 for nonzero, then forwards the original descriptor pointer unchanged as r2 through the driver wrapper.',
             'limit':'Field types/names and linkage from this nearby path to a specific CDNN input tensor remain unproven; width,height,Y/UV,ROI cannot be assigned from these checks alone.'},
            {'id':'ARM-INDIRECT-BOUNDARY','addresses':['0x065c6a0c','0x065c6a10','0x065c6a14','0x065c6a2c','0x065c6a30','0x065c6a32'],
             'fact':'Driver stores the original descriptor pointer at object+0x10, callback at+0x14 and selector at+0x0c, then calls the runtime function-table slot+8 using BLX r1.',
             'limit':'The function-table entry value is not recovered here. No claim that this slot executes on XM4, or that downstream preprocessing is absent.'},
            {'id':'ARM-TABLE-BASE','addresses':['0x065c6914','0x065c6984','0x065c6996','0x067b64f6','0x067b64f8'],
             'fact':'The immediate constructor path uses fixed RAM object0x08aa7ee4 and writes function-table base0x08aa7efc into its first word. Thus the submit slot+8 reads RAM word0x08aa7f04 on this successful initialization path.',
             'limit':'The table lies outside this component direct-load range; no initialized RAM content is available. Slot contents and downstream callee remain unresolved without widening scope.'},
            {'id':'ARM-FLOAT-FIELD','addresses':['0x067e0530','0x067e0564','0x067e0568','0x067e056c','0x067e0576','0x067e0586'],
             'fact':'Descriptor comparison checks six16-bit fields(+0..+0xa), a float32 field at+0x0c by VCMP.F32, and a word at+0x20; adjacent copy routine preserves0x100 descriptor bytes.',
             'limit':'A float32 field is established, but no value or mean/scale/threshold semantics. Float equality is not evidence for numerical preprocessing.'},
            {'id':'ARM-MODEL-LOAD','addresses':['0x067e0702','0x067e0770','0x067e0788','0x067e07b4'],
             'fact':'Loader verifies CDNN magic, declared bound and model class counts6/2 while moving model/runtime components.',
             'limit':'This path describes model loading, not camera-frame format or preprocessing.'}],
        'answers':{'input_pixel_format':'unresolved','mean_subtraction':'unresolved',
                   'normalization_scale':'unresolved','ROI_crop_and_resize':'unresolved',
                   'confirmed_boundary':'Descriptor assembly/copy/queue and a separate nearby opaque driver handoff, ending at the function pointer held in RAM[0x08aa7f04].'},
        'bounded_negative_findings':[
            'No per-pixel processing or mean/scale arithmetic in the selected registration/copy functions.',
            'Local candidate geometry functions found by adjacency were not connected to the confirmed registration chain and are not treated as CDNN preprocessing.',
            'No connected device state, memory-object contents, original tensors or XM4 code execution were available.'],
        'blocks':blocks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps({'output':str(args.output),'selected_code_bytes':report['scope']['selected_code_bytes'],
                      'answers':report['answers']},indent=2))


if __name__=='__main__':main()
