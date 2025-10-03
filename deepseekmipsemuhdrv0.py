#!/usr/bin/env python3
"""
MIPSEMU1.0 — Darkness Revived (Fast3D Edition)
N64 Emulator Skeleton with Minimal MIPS R4300i Core & Boot-Stub Loader
Python 3.10+ | Tkinter GUI

Notes
-----
- This is a learning-focused skeleton that "recognizes" N64 ROMs by:
  1) Detecting and normalizing ROM byte-order (.z64/.n64/.v64).
  2) Parsing and displaying header fields (name, game ID, region, CRC1/CRC2).
  3) Copying the boot stub (IPL3) at 0x40..0xFFF to SP DMEM 0xA4000040 and
     starting execution there with a tiny MIPS interpreter.
- The CPU implements a safe subset of the R4300i ISA commonly touched by
  early boot code. Unsupported/unknown opcodes are treated as NOPs.
- Memory-mapped IO is largely stubbed and non-functional; this is not
  a full emulator. It's enough to load and step without blowing up.
- Endianness is fully normalized to big-endian (.z64 layout) internally.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os, struct, time, threading
from pathlib import Path
from datetime import datetime

# --------------------------- Utilities ---------------------------

def u32(x): return x & 0xFFFFFFFF
def s32(x): x &= 0xFFFFFFFF; return x if x < 0x80000000 else x - 0x100000000
def sign16(x): x &= 0xFFFF; return x if x < 0x8000 else x - 0x10000
def sext16(x): return u32(sign16(x))

def bits(val, lo, hi):
    """Inclusive bit slice [lo, hi] (0 = LSB)."""
    mask = (1 << (hi - lo + 1)) - 1
    return (val >> lo) & mask

# --------------------------- ROM Header ---------------------------

class ROMHeader:
    """N64 ROM Header Parser (normalized to big-endian view)."""
    def __init__(self, data_be: bytes):
        self.raw_data = data_be[:0x40]
        self.parse()

    def parse(self):
        magic = struct.unpack(">I", self.raw_data[0:4])[0]
        self.endian = "big" if magic == 0x80371240 else "unknown"
        self.clock_rate = struct.unpack(">I", self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack(">I", self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack(">I", self.raw_data[0x0C:0x10])[0]
        self.crc1 = struct.unpack(">I", self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack(">I", self.raw_data[0x14:0x18])[0]
        self.name = self.raw_data[0x20:0x34].decode("ascii", errors="ignore").strip("\x00")
        self.game_id = self.raw_data[0x3B:0x3F].decode("ascii", errors="ignore")
        self.cart_id = chr(self.raw_data[0x3F])
        self.region = self.cart_id  # rough

# --------------------------- Byte-Order ---------------------------

def normalize_rom_to_z64_be(rom: bytes) -> bytes:
    """
    Normalize ROM to standard big-endian .z64 byte order based on magic.
    """
    if len(rom) < 4:
        return rom
    b0, b1, b2, b3 = rom[0], rom[1], rom[2], rom[3]
    magic = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    if magic == 0x80371240:  # .z64 big endian
        return rom
    elif magic == 0x40123780:  # .n64 little endian -> swap 32-bit
        out = bytearray(len(rom))
        for i in range(0, len(rom), 4):
            out[i:i+4] = rom[i:i+4][::-1]
        return bytes(out)
    elif magic == 0x37804012:  # .v64 byteswapped -> swap every 16-bit pair
        out = bytearray(len(rom))
        for i in range(0, len(rom), 2):
            if i+1 < len(rom):
                out[i], out[i+1] = rom[i+1], rom[i]
            else:
                out[i] = rom[i]
        return bytes(out)
    else:
        # Unknown magic: try best-effort (assume it's already BE)
        return rom

# --------------------------- Memory ---------------------------

class Memory:
    """
    Very small subset of N64 memory map.
      - RDRAM:       0x00000000 - 0x007FFFFF (8MB)
      - SP DMEM:     0x04000000 - 0x04000FFF (4KB)  (alias KSEG1: 0xA4000000 - ...)
      - SP IMEM:     0x04001000 - 0x04001FFF (4KB)  (alias KSEG1: 0xA4001000 - ...)
      - Cartridge:   0x10000000 - 0x1FBFFFFF (ROM, read-only)
    Everything else is unmapped and returns 0 on reads; writes are ignored.
    """
    def __init__(self):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)
        self.rom_be = None
        self.rom_size = 0

    def load_rom(self, rom_data_be: bytes):
        self.rom_be = rom_data_be
        self.rom_size = len(rom_data_be)

    # --- Addressing helpers ---

    @staticmethod
    def virt_to_phys(addr: int) -> int:
        """
        Map KSEG0/KSEG1/USEG virtual to 'physical-like' 0x00000000.. range.
        For our tiny model, mask to 0x1FFFFFFF to fold segments.
        """
        return addr & 0x1FFFFFFF

    def _read(self, phys: int, size: int) -> int:
        # SP DMEM/IMEM
        if 0x04000000 <= phys <= 0x04000FFF and size == 1:
            return self.sp_dmem[phys - 0x04000000]
        if 0x04001000 <= phys <= 0x04001FFF and size == 1:
            return self.sp_imem[phys - 0x04001000]
        # RDRAM
        if 0x00000000 <= phys <= 0x007FFFFF and size == 1:
            return self.rdram[phys]
        # Cartridge ROM (read-only)
        if 0x10000000 <= phys <= 0x1FBFFFFF and size == 1 and self.rom_be:
            off = phys - 0x10000000
            if 0 <= off < self.rom_size:
                return self.rom_be[off]
        # Unmapped
        return 0

    def _write(self, phys: int, val: int, size: int):
        b = val & 0xFF
        # SP DMEM/IMEM
        if 0x04000000 <= phys <= 0x04000FFF and size == 1:
            self.sp_dmem[phys - 0x04000000] = b; return
        if 0x04001000 <= phys <= 0x04001FFF and size == 1:
            self.sp_imem[phys - 0x04001000] = b; return
        # RDRAM
        if 0x00000000 <= phys <= 0x007FFFFF and size == 1:
            self.rdram[phys] = b; return
        # Cartridge ROM & others are read-only or ignored

    # --- Public byte/half/word ops ---

    def read_u8(self, addr: int) -> int:
        return self._read(self.virt_to_phys(addr), 1)

    def read_u16(self, addr: int) -> int:
        b0 = self.read_u8(addr)
        b1 = self.read_u8(addr + 1)
        return (b0 << 8) | b1

    def read_u32(self, addr: int) -> int:
        b0 = self.read_u8(addr)
        b1 = self.read_u8(addr + 1)
        b2 = self.read_u8(addr + 2)
        b3 = self.read_u8(addr + 3)
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    def write_u8(self, addr: int, val: int):
        self._write(self.virt_to_phys(addr), val, 1)

    def write_u16(self, addr: int, val: int):
        self.write_u8(addr, (val >> 8) & 0xFF)
        self.write_u8(addr + 1, val & 0xFF)

    def write_u32(self, addr: int, val: int):
        val &= 0xFFFFFFFF
        self.write_u8(addr, (val >> 24) & 0xFF)
        self.write_u8(addr + 1, (val >> 16) & 0xFF)
        self.write_u8(addr + 2, (val >> 8) & 0xFF)
        self.write_u8(addr + 3, val & 0xFF)

    # --- Boot stub loader (IPL3) ---

    def load_boot_stub_to_sp_dmem(self):
        """
        Copy 0xFC0 bytes from ROM[0x40..0xFFF] to SP DMEM[0x40..0xFFF].
        This mirrors the cartridge boot stub convention. Safe if ROM is small.
        """
        if not self.rom_be or self.rom_size < 0x1000:
            return False
        src = self.rom_be[0x40:0x1000]
        end = min(0x1000, 0x40 + len(src))
        # Clear DMEM
        for i in range(0x1000):
            self.sp_dmem[i] = 0
        # Copy at offset 0x40
        self.sp_dmem[0x40:end] = src[:(end - 0x40)]
        return True

# --------------------------- CPU Core (minimal) ---------------------------

class MIPSCPU:
    """
    Extremely small MIPS R4300i core just to step the N64 boot stub.
    - Implements a handful of integer, branch, and load/store ops.
    - Branch delay slots are honored.
    - COP0 reads/writes are stubbed via a small register file.
    Unsupported opcodes become NOP to keep things moving.
    """
    def __init__(self, memory: Memory, logger=None):
        self.mem = memory
        self.reg = [0] * 32
        self.hi = 0
        self.lo = 0
        self.pc = 0xA4000040  # SP DMEM + 0x40 (KSEG1 alias)
        self.cp0 = [0] * 32
        self.running = False
        self.instructions = 0
        self.logger = logger

    def reset(self):
        self.reg = [0] * 32
        self.hi = self.lo = 0
        self.cp0 = [0] * 32
        # A common initial $sp for the boot environment; arbitrary but harmless here.
        self.reg[29] = 0xA4001FF0
        self.pc = 0xA4000040
        self.instructions = 0

    # --- register helpers ---
    def _read_reg(self, i): return 0 if i == 0 else u32(self.reg[i])
    def _write_reg(self, i, val):
        if i != 0:
            self.reg[i] = u32(val)

    # --- instruction fetch ---
    def _fetch(self, addr):
        return self.mem.read_u32(addr)

    # --- execution ---
    def step(self):
        if not self.running:
            return
        pc = self.pc
        instr = self._fetch(pc)

        # Defaults
        next_pc = u32(pc + 4)
        branch_taken = False
        branch_target = 0

        op = bits(instr, 26, 31)
        rs = bits(instr, 21, 25)
        rt = bits(instr, 16, 20)
        rd = bits(instr, 11, 15)
        sa = bits(instr, 6, 10)
        fn = bits(instr, 0, 5)
        imm = bits(instr, 0, 15)
        simm = sext16(imm)
        target = bits(instr, 0, 25)

        def do_load(addr, size):
            if size == 1: return self.mem.read_u8(addr)
            if size == 2: return self.mem.read_u16(addr)
            return self.mem.read_u32(addr)

        def do_store(addr, val, size):
            if size == 1: self.mem.write_u8(addr, val)
            elif size == 2: self.mem.write_u16(addr, val)
            else: self.mem.write_u32(addr, val)

        # --- Decode ---
        if op == 0x00:  # SPECIAL
            if fn == 0x00:  # SLL
                self._write_reg(rd, (self._read_reg(rt) << sa) & 0xFFFFFFFF)
            elif fn == 0x02:  # SRL
                self._write_reg(rd, (self._read_reg(rt) >> sa) & 0xFFFFFFFF)
            elif fn == 0x03:  # SRA
                self._write_reg(rd, u32(s32(self._read_reg(rt)) >> sa))
            elif fn == 0x04:  # SLLV
                self._write_reg(rd, u32(self._read_reg(rt) << (self._read_reg(rs) & 31)))
            elif fn == 0x06:  # SRLV
                self._write_reg(rd, u32(self._read_reg(rt) >> (self._read_reg(rs) & 31)))
            elif fn == 0x07:  # SRAV
                self._write_reg(rd, u32(s32(self._read_reg(rt)) >> (self._read_reg(rs) & 31)))
            elif fn == 0x08:  # JR
                branch_taken = True
                branch_target = u32(self._read_reg(rs))
            elif fn == 0x09:  # JALR
                self._write_reg(rd if rd != 0 else 31, next_pc)
                branch_taken = True
                branch_target = u32(self._read_reg(rs))
            elif fn == 0x10:  # MFHI
                self._write_reg(rd, self.hi)
            elif fn == 0x12:  # MFLO
                self._write_reg(rd, self.lo)
            elif fn == 0x11:  # MTHI
                self.hi = self._read_reg(rs)
            elif fn == 0x13:  # MTLO
                self.lo = self._read_reg(rs)
            elif fn == 0x18:  # MULT
                self._do_mult(self._read_reg(rs), self._read_reg(rt), signed=True)
            elif fn == 0x19:  # MULTU
                self._do_mult(self._read_reg(rs), self._read_reg(rt), signed=False)
            elif fn == 0x1A:  # DIV
                self._do_div(self._read_reg(rs), self._read_reg(rt), signed=True)
            elif fn == 0x1B:  # DIVU
                self._do_div(self._read_reg(rs), self._read_reg(rt), signed=False)
            elif fn == 0x21:  # ADDU
                self._write_reg(rd, self._read_reg(rs) + self._read_reg(rt))
            elif fn == 0x23:  # SUBU
                self._write_reg(rd, self._read_reg(rs) - self._read_reg(rt))
            elif fn == 0x24:  # AND
                self._write_reg(rd, self._read_reg(rs) & self._read_reg(rt))
            elif fn == 0x25:  # OR
                self._write_reg(rd, self._read_reg(rs) | self._read_reg(rt))
            elif fn == 0x26:  # XOR
                self._write_reg(rd, self._read_reg(rs) ^ self._read_reg(rt))
            elif fn == 0x27:  # NOR
                self._write_reg(rd, u32(~(self._read_reg(rs) | self._read_reg(rt))))
            elif fn == 0x2A:  # SLT
                self._write_reg(rd, 1 if s32(self._read_reg(rs)) < s32(self._read_reg(rt)) else 0)
            elif fn == 0x2B:  # SLTU
                self._write_reg(rd, 1 if self._read_reg(rs) < self._read_reg(rt) else 0)
            else:
                # Unimplemented SPECIAL -> NOP
                pass

        elif op == 0x01:  # REGIMM
            rtcode = rt
            cmp = s32(self._read_reg(rs))
            if rtcode == 0x00:  # BLTZ
                if cmp < 0: branch_taken = True; branch_target = u32(next_pc + (simm << 2))
            elif rtcode == 0x01:  # BGEZ
                if cmp >= 0: branch_taken = True; branch_target = u32(next_pc + (simm << 2))
            elif rtcode == 0x10:  # BLTZAL
                if cmp < 0:
                    self._write_reg(31, next_pc)
                    branch_taken = True; branch_target = u32(next_pc + (simm << 2))
            elif rtcode == 0x11:  # BGEZAL
                if cmp >= 0:
                    self._write_reg(31, next_pc)
                    branch_taken = True; branch_target = u32(next_pc + (simm << 2))
            else:
                pass

        elif op == 0x02:  # J
            branch_taken = True
            branch_target = u32((next_pc & 0xF0000000) | (target << 2))

        elif op == 0x03:  # JAL
            self._write_reg(31, next_pc)
            branch_taken = True
            branch_target = u32((next_pc & 0xF0000000) | (target << 2))

        elif op == 0x04:  # BEQ
            if self._read_reg(rs) == self._read_reg(rt):
                branch_taken = True; branch_target = u32(next_pc + (simm << 2))

        elif op == 0x05:  # BNE
            if self._read_reg(rs) != self._read_reg(rt):
                branch_taken = True; branch_target = u32(next_pc + (simm << 2))

        elif op == 0x06:  # BLEZ
            if s32(self._read_reg(rs)) <= 0:
                branch_taken = True; branch_target = u32(next_pc + (simm << 2))

        elif op == 0x07:  # BGTZ
            if s32(self._read_reg(rs)) > 0:
                branch_taken = True; branch_target = u32(next_pc + (simm << 2))

        elif op == 0x08:  # ADDI
            self._write_reg(rt, u32(self._read_reg(rs) + simm))

        elif op == 0x09:  # ADDIU
            self._write_reg(rt, u32(self._read_reg(rs) + simm))

        elif op == 0x0A:  # SLTI
            self._write_reg(rt, 1 if s32(self._read_reg(rs)) < simm else 0)

        elif op == 0x0B:  # SLTIU
            self._write_reg(rt, 1 if self._read_reg(rs) < u32(simm) else 0)

        elif op == 0x0C:  # ANDI
            self._write_reg(rt, self._read_reg(rs) & (imm))

        elif op == 0x0D:  # ORI
            self._write_reg(rt, self._read_reg(rs) | (imm))

        elif op == 0x0E:  # XORI
            self._write_reg(rt, self._read_reg(rs) ^ (imm))

        elif op == 0x0F:  # LUI
            self._write_reg(rt, (imm << 16))

        elif op == 0x10:  # COP0
            rs_co = bits(instr, 21, 25)
            if rs_co == 0x00:  # MFC0
                self._write_reg(rt, self.cp0[rd])
            elif rs_co == 0x04:  # MTC0
                self.cp0[rd] = self._read_reg(rt)
            else:
                # TLB ops or others -> NOP
                pass

        elif op == 0x20:  # LB
            addr = u32(self._read_reg(rs) + simm)
            val = self.mem.read_u8(addr)
            self._write_reg(rt, u32(sign16(val if val < 0x80 else val - 0x100)))
        elif op == 0x21:  # LH
            addr = u32(self._read_reg(rs) + simm)
            val = self.mem.read_u16(addr)
            self._write_reg(rt, u32(sign16(val)))
        elif op == 0x23:  # LW
            addr = u32(self._read_reg(rs) + simm)
            self._write_reg(rt, self.mem.read_u32(addr))
        elif op == 0x24:  # LBU
            addr = u32(self._read_reg(rs) + simm)
            self._write_reg(rt, self.mem.read_u8(addr))
        elif op == 0x25:  # LHU
            addr = u32(self._read_reg(rs) + simm)
            self._write_reg(rt, self.mem.read_u16(addr))
        elif op == 0x28:  # SB
            addr = u32(self._read_reg(rs) + simm)
            self.mem.write_u8(addr, self._read_reg(rt))
        elif op == 0x29:  # SH
            addr = u32(self._read_reg(rs) + simm)
            self.mem.write_u16(addr, self._read_reg(rt))
        elif op == 0x2B:  # SW
            addr = u32(self._read_reg(rs) + simm)
            self.mem.write_u32(addr, self._read_reg(rt))

        elif op == 0x2F:  # CACHE (stub: ignore)
            pass

        else:
            # Unimplemented top-level opcode -> NOP
            pass

        # Execute branch delay slot
        if branch_taken:
            delay_instr = self._fetch(next_pc)
            self._exec_delay_slot(delay_instr)
            self.pc = u32(branch_target)
        else:
            self.pc = next_pc

        self.instructions += 1

    def _exec_delay_slot(self, instr):
        # Minimal: reuse decoder for a subset; ignore nested branches inside delay slot for simplicity.
        op = bits(instr, 26, 31)
        rs = bits(instr, 21, 25)
        rt = bits(instr, 16, 20)
        rd = bits(instr, 11, 15)
        sa = bits(instr, 6, 10)
        fn = bits(instr, 0, 5)
        imm = bits(instr, 0, 15)
        simm = sext16(imm)

        if op == 0x00 and fn == 0x21:   # ADDU
            self._write_reg(rd, self._read_reg(rs) + self._read_reg(rt))
        elif op == 0x00 and fn == 0x25: # OR
            self._write_reg(rd, self._read_reg(rs) | self._read_reg(rt))
        elif op == 0x0D:                # ORI
            self._write_reg(rt, self._read_reg(rs) | imm)
        elif op == 0x0F:                # LUI
            self._write_reg(rt, (imm << 16))
        elif op == 0x23:                # LW
            addr = u32(self._read_reg(rs) + simm)
            self._write_reg(rt, self.mem.read_u32(addr))
        elif op == 0x2B:                # SW
            addr = u32(self._read_reg(rs) + simm)
            self.mem.write_u32(addr, self._read_reg(rt))
        else:
            # Treat any other delay-slot op as NOP for safety
            pass

    def _do_mult(self, a, b, signed=True):
        if signed:
            res = s32(a) * s32(b)
        else:
            res = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)
        self.hi = u32((res >> 32) & 0xFFFFFFFF)
        self.lo = u32(res & 0xFFFFFFFF)

    def _do_div(self, a, b, signed=True):
        if b == 0:
            # MIPS leaves HI/LO undefined; keep previous values.
            return
        if signed:
            self.lo = u32(int(s32(a) / s32(b)))
            self.hi = u32(int(s32(a) % s32(b)))
        else:
            self.lo = u32((a & 0xFFFFFFFF) // (b & 0xFFFFFFFF))
            self.hi = u32((a & 0xFFFFFFFF) % (b & 0xFFFFFFFF))

# --------------------------- Graphics (stub) ---------------------------

class Fast3DStub:
    """Placeholder Fast3D parser (pretends to read triangles)."""
    def __init__(self):
        self.triangles = []
    def step(self):
        # Demo: one red triangle
        self.triangles = [[(300, 200), (400, 400), (200, 400)]]

class VideoInterface:
    def __init__(self, canvas):
        self.canvas = canvas
        self.frame_count = 0
        self.mode = "fast3d"
        self.fast3d = Fast3DStub()

    def render_frame(self, cpu_state, rom_info=None):
        self.canvas.delete("all")
        screen_x, screen_y = 192, 114
        self.canvas.create_rectangle(screen_x, screen_y, screen_x + 640, screen_y + 480,
                                     fill="#000000", outline="#00ff88", width=2)

        if self.mode == "fast3d":
            self.fast3d.step()
            for tri in self.fast3d.triangles:
                self.canvas.create_polygon(tri, fill="#ff0000", outline="white")

            self.canvas.create_text(512, 150, text="Fast3D Triangle Stub",
                                    font=("Arial", 18), fill="#00ff88")

        # Overlay CPU + ROM info
        text_lines = [
            f"PC: 0x{cpu_state['pc']:08X}    Instr#: {cpu_state['instructions']}",
            "Regs: " + " ".join([f"r{i}={cpu_state['regs'][i]:08X}" for i in range(8)]),
        ]
        if rom_info:
            text_lines += [
                f"ROM: {rom_info.get('name','?')}  ID: {rom_info.get('game_id','??')}  Region: {rom_info.get('region','?')}",
                f"CRC1: {rom_info.get('crc1','0'):08X}  CRC2: {rom_info.get('crc2','0'):08X}"
            ]

        self.canvas.create_text(16, 16, text="\n".join(text_lines),
                                font=("Consolas", 10), anchor="nw", fill="#d0ffd0")
        self.frame_count += 1

# --------------------------- Emulator Shell ---------------------------

class MIPSEMU1:
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU1.0 — Fast3D Edition")
        self.memory = Memory()
        self.cpu = MIPSCPU(self.memory, logger=self)
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.video = VideoInterface(self.canvas)
        self.running = False
        self.rom_header = None

        # Controls
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X)
        tk.Button(ctrl, text="Open ROM", command=self.cmd_open_rom).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Start", command=self.start).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Step 1", command=self.step_once).pack(side=tk.LEFT, padx=2)

        # Log
        self.log_text = scrolledtext.ScrolledText(root, height=8, bg="#0a0a0a", fg="#00ff00")
        self.log_text.pack(fill=tk.BOTH, expand=False)
        self.log("MIPSEMU1.0 initialized")

    # ----- Logging -----
    def log(self, msg):
        t = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{t}] {msg}\n"); self.log_text.see(tk.END)

    # ----- ROM -----
    def load_rom(self, path):
        data = Path(path).read_bytes()
        be = normalize_rom_to_z64_be(data)
        self.memory.load_rom(be)
        self.memory.load_boot_stub_to_sp_dmem()

        self.rom_header = ROMHeader(be)
        self.cpu.reset()
        self.log(f"ROM loaded: {self.rom_header.name} ({self.rom_header.game_id}) region={self.rom_header.region}")
        self.log(f"CRC1={self.rom_header.crc1:08X} CRC2={self.rom_header.crc2:08X}")
        self.log("Boot stub copied to SP DMEM @ 0xA4000040. PC set to 0xA4000040.")

    def cmd_open_rom(self):
        fn = filedialog.askopenfilename(title="Open N64 ROM",
                                        filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64 *.bin *.rom"), ("All files", "*.*")])
        if fn:
            try:
                self.load_rom(fn)
            except Exception as e:
                self.log(f"Error loading ROM: {e}")

    # ----- Run/Stop/Step -----
    def start(self):
        if not self.memory.rom_be:
            self.log("No ROM loaded."); return
        self.running = True
        self.cpu.running = True
        threading.Thread(target=self.emu_loop, daemon=True).start()
        self.render_loop()
        self.log("Emulation started")

    def stop(self):
        self.running = False
        self.cpu.running = False
        self.log("Emulation stopped")

    def step_once(self):
        if not self.memory.rom_be:
            self.log("No ROM loaded."); return
        self.cpu.running = True
        self.cpu.step()
        self.cpu.running = False
        self.render_once()

    def emu_loop(self):
        try:
            while self.running and self.cpu.running:
                self.cpu.step()
                # ~1.5 MHz-ish toy speed, not real-time
                time.sleep(0.0006)
        except Exception as e:
            self.log(f"CPU Exception at PC=0x{self.cpu.pc:08X}: {e}")
            self.cpu.running = False
            self.running = False

    def render_once(self):
        state = {'pc': self.cpu.pc, 'instructions': self.cpu.instructions, 'regs': self.cpu.reg[:8]}
        rom_info = None
        if self.rom_header:
            rom_info = {
                'name': self.rom_header.name,
                'game_id': self.rom_header.game_id,
                'region': self.rom_header.region,
                'crc1': self.rom_header.crc1,
                'crc2': self.rom_header.crc2,
            }
        self.video.render_frame(state, rom_info)

    def render_loop(self):
        if self.running:
            self.render_once()
            # ~60Hz UI refresh
            self.root.after(16, self.render_loop)

# --------------------------- Main ---------------------------

def main():
    root = tk.Tk(); root.geometry("1024x768")
    app = MIPSEMU1(root)
    root.mainloop()

if __name__ == "__main__":
    main()
