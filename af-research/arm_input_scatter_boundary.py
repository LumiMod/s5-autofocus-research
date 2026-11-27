#!/usr/bin/env python3
"""Extend the prior ARM boundary by reading original scatter-initialized data.

No QEMU start, firmware execution, RAM mutation, image inference or training.
Historical QEMU reports, when supplied, are corroboration of byte copying only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from arm_input_boundary import BASE, inspect_block
from extract import require, write_json
from research_paths import add_firmware_arguments, resolve_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--historical-scatter-report',type=Path)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'artifacts/recognition-diagnosis/arm-input-scatter-boundary.json')
    args=parser.parse_args()
    require(not args.output.exists(),'Refusing to overwrite evidence')
    program=(resolve_source(args)/'03_program.bin').read_bytes()
    digest=hashlib.sha256(program).hexdigest()
    previous=json.loads((Path(__file__).parent/'evidence/2026-11-27/runtime-investigation.json').read_text())
    require(digest==previous['component_digests']['03_program.bin'],'Program identity differs from prior runtime evidence')
    scatter=[]
    for va in range(0x062800fc,0x0628015c,16):
        source,destination,size,handler=struct.unpack_from('<4I',program,va-BASE)
        require(handler in (0x062800c4,0x062800e0),'Unexpected scatter handler')
        scatter.append({'table_va':hex(va),'raw_hex':program[va-BASE:va-BASE+16].hex(),
                        'source_va':source,'destination_va':destination,'bytes':size,
                        'operation':'copy' if handler==0x062800c4 else 'zero'})
    table=0x08aa7efc
    matches=[r for r in scatter if r['destination_va']<=table and table+28<=r['destination_va']+r['bytes']]
    require(len(matches)==1 and matches[0]['operation']=='copy','Function table not covered uniquely by a copy')
    region=matches[0]
    offset=region['source_va']-BASE+table-region['destination_va']
    require(0<=offset<=len(program)-28,'Scatter table source not in pinned program component')
    raw=program[offset:offset+28]
    entries=list(struct.unpack('<7I',raw))
    require(entries==[0x067b6341,0x067b6409,0x067b6457,0x067b64ad,0x067b64c1,0x067b64d3,0x067b64e5],
            'Function-table bytes changed')
    copy_blob=program[region['source_va']-BASE:region['source_va']-BASE+region['bytes']]
    copy_digest=hashlib.sha256(copy_blob).hexdigest()
    historical=None
    if args.historical_scatter_report:
        history=json.loads(args.historical_scatter_report.read_text())
        case=next(c for c in history['cases'] if c['kind']=='all_modules')
        check=next(c for c in case['checks'] if int(c['destination'],16)==region['destination_va'])
        require(check['expected_sha256']==copy_digest and check['destination_sha256']==copy_digest,
                'Historical whole-region bytes do not match this source')
        historical={'path':str(args.historical_scatter_report),
                    'report_sha256':hashlib.sha256(args.historical_scatter_report.read_bytes()).hexdigest(),
                    'recorded_copy_hash_matches_original_source':True,
                    'evidence_type':'Existing QEMU scatter-routine check with synthetic CPU/stack state; not physical-camera inference.'}
    ranges=[('original_scatter_dispatch',0x06280090,0x062800bc,'thumb'),
            ('original_scatter_copy_handler',0x062800c4,0x062800de,'thumb'),
            ('resolved_submit_function',0x067b6456,0x067b64ac,'thumb'),
            ('descriptor_pointer_MMIO_write',0x067c14b6,0x067c14bc,'thumb'),
            ('control_bit_helpers',0x067c1538,0x067c1596,'thumb'),
            ('submit_gate_and_status_poll',0x067c15c6,0x067c1604,'thumb')]
    blocks=[inspect_block(program,*r) for r in ranges]
    require(all(b['decoded_bytes']==b['bytes'] for b in blocks),'Selected function range not completely decoded')
    report={
        'source_file':'03_program.bin','source_sha256':digest,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256':{'arm_input_boundary.py':hashlib.sha256(Path(__file__).with_name('arm_input_boundary.py').read_bytes()).hexdigest()},
        'supersedes_limitations_of':'arm-input-boundary.json: function-table slot contents are now recovered statically through scatter; preprocessing semantics still unresolved.',
        'scope':{'static_original_firmware':True,'qemu_started':False,'training':False,
                 'selected_code_bytes':sum(b['bytes'] for b in blocks),'physical_camera_execution':False},
        'scatter_records':scatter,
        'table_mapping':{'destination_ram_va':hex(table),'scatter_record':region,
                         'formula':'file_offset = source_va - program_load_base + table_ram_va - scatter_destination',
                         'original_source_va':hex(BASE+offset),'program_file_offset':hex(offset),
                         'raw_table_hex':raw.hex(),'whole_copy_source_sha256':copy_digest,
                         'slot_entries':[{'offset':i*4,'ram_va':hex(table+i*4),'thumb_pointer':hex(v),
                                          'code_va':hex(v&~1)} for i,v in enumerate(entries)]},
        'historical_emulation_corroboration':historical,
        'observations':[
            {'id':'SCATTER-TABLE','fact':'The original first scatter copy relocates0x54fa4 bytes from0x070879b8 to0x08a67000. This includes the seven-word function table at0x08aa7efc.',
             'evidence':['table0x062800fc','copy_handler0x062800c4','03_program.bin+0xe488b4']},
            {'id':'RESOLVED-SLOT','fact':'RAM0x08aa7f04 receives original Thumb pointer0x067b6457. Entry0x067b6456 loads object+0x10 (the descriptor saved by the traced wrapper) and passes it to0x067c14b6.',
             'evidence':['0x067b645c:2869','0x067b645e:0bf02af8']},
            {'id':'MMIO-DESCRIPTOR','fact':'0x067c14b6 loads base0x7c01f000;0x067c14b8 stores the unchanged descriptor pointer to base+0x70, address0x7c01f070.',
             'evidence':['0x067c14b6:5c49','literal0x067c1628=0x7c01f000','0x067c14b8:0867'],
             'limit':'The register role is inferred as descriptor submission from this pointer flow; precise peripheral/XM4 ownership and hardware payload semantics are not established here.'},
            {'id':'MMIO-CONTROL','fact':'The resolved function controls0x7c01f018 bit0, polls0x7c01f020 bits0x10/0x08 with a bounded retry, then changes control bits at0x7c01f1cc,0x7c01f140,0x7c01f1c0,0x7c01f1c4 and0x7c01f100.',
             'evidence':['0x067b6464','0x067b6484','0x067c15c6','0x067c15e4','0x067c1538','0x067c155a','0x067c156a'],
             'limit':'These are register operations, not ARM pixel reads or known mean/scale operations.'}],
        'answer':{'resolved_call_target':'0x067b6456 Thumb via original pointer0x067b6457',
                  'new_boundary':'Original descriptor pointer written unchanged to MMIO0x7c01f070.',
                  'input_format':'unresolved','mean_subtraction':'unresolved','scale':'unresolved','ROI_crop_resize':'unresolved'},
        'limitations':['Recovered initial table contents from original scatter source, not captured live RAM; later mutations were not exhaustively searched.',
                       'Existing QEMU scatter equality only corroborates byte copying; no emulator peripheral response is used as camera truth.',
                       'No per-pixel arithmetic in the resolved submit/control functions does not mean preprocessing is absent upstream or inside hardware/DSP.',
                       'The nearby submission path still lacks a complete traced link to the CDNN inference queue consumer.'],
        'blocks':blocks}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,report)
    print(json.dumps({'output':str(args.output),'table_source_offset':hex(offset),'target':hex(entries[2]),
                      'boundary':report['answer']['new_boundary'],'selected_code_bytes':report['scope']['selected_code_bytes']},indent=2))


if __name__=='__main__':main()
