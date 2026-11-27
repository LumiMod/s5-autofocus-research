#!/usr/bin/env python3
"""Parse the fixed result boundary consumed by S5 ARM0x063bd2c2.

This is a boundary parser, not a DSP proposal decoder or a class-name contract.
Raw records retain all six words; observed ARM behavior uses low16 coordinate
and confidence bits. Do not infer head/visible-face labels from numeric class.
"""
import argparse
import json
from pathlib import Path
import struct


def parse_input(blob):
    if len(blob) != 16+60*24:
        raise ValueError('Expected exactly16 header bytes and60 records of24 bytes')
    records=[]
    for i in range(60):
        words=struct.unpack_from('<6I',blob,16+i*24)
        cls,x,y,w,h,score=words
        x,y,w,h,score=(v&0xffff for v in (x,y,w,h,score))
        records.append(dict(slot=i,raw_u32_words=list(words),numeric_class=cls,
            arm_numeric_class=cls if cls<=6 else 0,
            xywh_low16=[x,y,w,h],xyxy_arm_u16=[x,y,(x+w)&0xffff,(y+h)&0xffff],
            score_low16=score,arm_score_f32=struct.unpack('<f',struct.pack('<f',score/65535.))[0]))
    return dict(header_hex=blob[:16].hex(),header_semantics='unresolved',records=records,
                class_names='unbound numeric IDs; no face/head declaration',source_contract='ARM0x063bd2c2 fixed60-slot reader')


def expected_output(blob,fill=0xa5):
    output=bytearray([fill])*(60*16)
    for row in parse_input(blob)['records']:
        offset=row['slot']*16
        struct.pack_into('<5H',output,offset,row['arm_numeric_class'],*row['xyxy_arm_u16'])
        struct.pack_into('<f',output,offset+12,row['arm_score_f32'])
    return bytes(output)


def parse_output(blob):
    if len(blob)!=60*16:
        raise ValueError('Expected60 records of16 bytes')
    return [dict(slot=i,numeric_class=struct.unpack_from('<H',blob,i*16)[0],
        xyxy_u16=list(struct.unpack_from('<4H',blob,i*16+2)),
        untouched_gap_hex=blob[i*16+10:i*16+12].hex(),score_f32=struct.unpack_from('<f',blob,i*16+12)[0]) for i in range(60)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--converted',action='store_true',help='Parse960-byte ARM-converted records instead of1456-byte raw result boundary')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite output')
    value=parse_output(args.input.read_bytes()) if args.converted else parse_input(args.input.read_bytes())
    args.output.write_text(json.dumps(value,indent=2)+'\n')
