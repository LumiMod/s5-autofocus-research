#!/usr/bin/env python3
"""Replay one original S5 ARM result-conversion function in local QEMU.

Synthetic records only; no DSP/network, camera, image, training, or peripherals.
The original instruction bytes and their literal pool keep their original VA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

BASE = Path(__file__).resolve().parent
CONTRACT = BASE/'artifacts/recognition-ranking-repair/firmware-contract'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_section(blob):
    header = struct.unpack_from('<16sHHIIIIIHHHHHH', blob)
    assert blob[:7] == b'\x7fELF\x01\x01\x01', 'Expected little-endian ELF32'
    start, size, count, string_index = header[6], header[11], header[12], header[13]
    sections = [struct.unpack_from('<10I', blob, start+i*size) for i in range(count)]
    strings = sections[string_index]
    names = blob[strings[4]:strings[4]+strings[5]]
    for index, section in enumerate(sections):
        name = names[section[0]:].split(b'\0', 1)[0]
        if name == b'.text':
            assert not any(s[1] in (4, 9) and s[7] == index and s[5] for s in sections), 'Unresolved text relocation'
            return blob[section[4]:section[4]+section[5]]
    raise ValueError('Missing text section')


def run(args):
    assert not args.output.exists(), 'Preserve previous replay attempts'
    original = args.firmware_root/'03_program.bin'
    verification = json.loads((args.firmware_root/'verification.json').read_text())
    expected_original_sha = next(r['sha256'] for r in verification['components'] if r['file'] == original.name)
    assert sha(original) == expected_original_sha, 'Original firmware hash changed'
    raw = (CONTRACT/'parser-synthetic-input.bin').read_bytes()
    expected = (CONTRACT/'parser-synthetic-converted.bin').read_bytes()
    assert len(raw) == 1456 and len(expected) == 960, 'Wrong fixture record counts'
    args.output.mkdir(parents=True)
    (args.output/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
    # Include function, TBB table and the original float65535 literal.
    program = original.read_bytes()
    (args.output/'converter.bin').write_bytes(program[0x063bd000-0x06280000:0x063bd700-0x06280000])
    (args.output/'input.bin').write_bytes(raw)
    (args.output/'output-initial.bin').write_bytes(bytes([0xa5])*976)
    (args.output/'expected-with-canary.bin').write_bytes(expected+bytes([0xa5])*16)
    assembly = '''.syntax unified
.arch armv7-a
.fpu vfpv3
.arm
.text
.global _start
_start:
    mrc p15, 0, r0, c1, c0, 2
    orr r0, r0, #0x00f00000
    mcr p15, 0, r0, c1, c0, 2
    isb
    mov r0, #0x40000000
    vmsr fpexc, r0
    ldr sp, =0x401f0000
    ldr r0, =0x40100000
    ldr r1, =0x40110000
    ldr r3, =0x063bd2c3
    blx r3
    ldr r0, =0x40110000
    ldr r1, =0x40120000
    ldr r2, =976
check_loop:
    ldrb r3, [r0], #1
    ldrb r4, [r1], #1
    cmp r3, r4
    bne failure
    subs r2, r2, #1
    bne check_loop
    mov r0, #4
    adr r1, pass_message
    svc #0x123456
    mov r0, #0x20
    adr r1, pass_exit
    svc #0x123456
failure:
    mov r0, #4
    adr r1, fail_message
    svc #0x123456
    mov r0, #0x20
    adr r1, fail_exit
    svc #0x123456
    b failure
.align 2
pass_exit: .word 0x20026, 0
fail_exit: .word 0x20026, 1
pass_message: .asciz "S5_CONVERTER_PASS_960_BYTES_AND_16_CANARY\\n"
fail_message: .asciz "S5_CONVERTER_MISMATCH\\n"
.align 2
.ltorg
'''
    asm = args.output/'harness.S'
    asm.write_text(assembly)
    obj = args.output/'harness.o'
    compile_command = ['/usr/bin/clang', '--target=armv7-none-eabi', '-c', str(asm), '-o', str(obj)]
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
    (args.output/'harness.bin').write_bytes(text_section(obj.read_bytes()))
    command = [str(args.qemu), '-M', 'virt', '-cpu', 'cortex-a15', '-m', '64M',
        '-display', 'none', '-serial', 'none', '-monitor', 'none',
        '-semihosting-config', 'enable=on,target=native']
    for name, address, cpu in [('converter.bin', '0x063bd000', False),
        ('input.bin', '0x40100000', False), ('output-initial.bin', '0x40110000', False),
        ('expected-with-canary.bin', '0x40120000', False), ('harness.bin', '0x41000000', True)]:
        command += ['-device', 'loader,file='+str(args.output/name)+',addr='+address+',force-raw=on'+(',cpu-num=0' if cpu else '')]
    plan = dict(original_program_sha256=expected_original_sha, qemu_sha256=sha(args.qemu),
        fixture_input_sha256=sha(CONTRACT/'parser-synthetic-input.bin'),
        fixture_expected_sha256=sha(CONTRACT/'parser-synthetic-converted.bin'),
        source_sha256=sha(Path(__file__)), compile_command=compile_command, qemu_command=command,
        original_entry='0x063bd2c3 (Thumb)', optimizer_steps=0, network_or_DSP_execution=False,
        model_images_read=False, test_access=False, synthetic_records=60)
    (args.output/'frozen-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        output = result.stdout+result.stderr
        passed = result.returncode == 0 and 'S5_CONVERTER_PASS_960_BYTES_AND_16_CANARY' in output
        report = dict(passed=passed, returncode=result.returncode, output=output,
            actual_original_ARM_instruction_execution=(True if passed or 'S5_CONVERTER_MISMATCH' in output
                                                       else 'not confirmed'),
            complete_960_output_bytes_and_16_tail_canary_equal_to_parser_fixture=passed)
    except subprocess.TimeoutExpired as error:
        report = dict(passed=False, timeout_seconds=15, output=str(error.stdout)+str(error.stderr),
                      actual_original_ARM_instruction_execution='not confirmed')
    assert sha(original) == expected_original_sha, 'Original firmware modified'
    report.update(plan_sha256=sha(args.output/'frozen-plan.json'), source_files={
        p.name:sha(p) for p in args.output.iterdir() if p.is_file()},
        source_firmware_unchanged=True, original_model_semantics_confirmed=False,
        network_or_DSP_execution=False, optimizer_steps=0, test_access=False,
        limitation='Original ARM converter only, with synthetic raw records. No network inference or face/head semantics established.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed','output','network_or_DSP_execution')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware-root', type=Path, required=True)
    parser.add_argument('--qemu', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
