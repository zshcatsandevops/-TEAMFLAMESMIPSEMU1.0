#!/usr/bin/env python3
"""
MIPSEMU2.1 — Darkness Revived II (RCP-Stubs)
Upgraded N64 Emulator Skeleton with Expanded MIPS R4300i Core + RCP MMIO stubs
Python 3.10+ | Tkinter GUI

Notes
-----
- Builds on the MIPSEMU2.0 skeleton (R4300i, Tk UI).
- Adds:
  • IMEM window (0x04001000-0x04001FFF) mapping in memory.
  • RCP register maps (SP/DP/VI/AI/PI/SI) with safe read/write behavior.
  • Minimal PI DMA (ROM → RDRAM) and SP DMA (RDRAM ↔ DMEM/IMEM).
  • A very small VI path that treats VI_ORIGIN/VI_WIDTH as a 16‑bpp RGBA5551
    framebuffer pointer and draws a preview onto the Tk canvas (stub only).
  • A few additional MIPS opcodes (MULT, DIV, SLT, SLTI, ADD, SUB, shifts by reg).
- Still NOT a full emulator. Missing: TLB/exceptions, PIF/IPL, real RSP microcode,
  RDP rasterization, interrupts/timers, accurate VI timings, etc.
- Endianness remains normalized to big-endian (.z64 layout) internally.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import struct
import time
import threading
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

# --------------------------- ROM Header & Region Data ---------------------------

N64_REGIONS = {
    '7': 'Beta',
    'A': 'Asia (NTSC)',
    'B': 'Brazil (PAL-M)',
    'C': 'China (iQue)',
    'D': 'Germany (PAL)',
    'E': 'USA (NTSC-U)',
    'F': 'France (PAL)',
    'G': 'Gateway 64 (NTSC)',
    'H': 'Netherlands (PAL)',
    'I': 'Italy (PAL)',
    'J': 'Japan (NTSC-J)',
    'K': 'Korea (NTSC)',
    'L': 'USA (NTSC-U)',
    'N': 'Canada (NTSC)',
    'P': 'Europe (PAL)',
    'S': 'Spain (PAL)',
    'U': 'Australia (PAL)',
    'W': 'Scandinavia (PAL)',
    'X': 'Europe (PAL)',
    'Y': 'Europe (PAL)',
}

class ROMHeader:
    """N64 ROM Header Parser (normalized to big-endian view)."""
    def __init__(self, data_be: bytes):
        self.raw_data = data_be[:0x40]
        self.parse()

    def parse(self):
        self.clock_rate = struct.unpack(">I", self.raw_data[0x04:0x08])[0]
        self.boot_address = struct.unpack(">I", self.raw_data[0x08:0x0C])[0]
        self.release = struct.unpack(">I", self.raw_data[0x0C:0x10])[0]
        self.crc1 = struct.unpack(">I", self.raw_data[0x10:0x14])[0]
        self.crc2 = struct.unpack(">I", self.raw_data[0x14:0x18])[0]
        self.name = self.raw_data[0x20:0x34].decode("ascii", errors="ignore").strip("\x00 ")
        self.game_id = self.raw_data[0x3B:0x3E].decode("ascii", errors="ignore")
        self.cart_id = self.raw_data[0x3F]
        region_char = chr(self.cart_id)
        self.region = f"{N64_REGIONS.get(region_char, 'Unknown')} ({region_char})"

# --------------------------- Byte-Order Normalization ---------------------------

def normalize_rom_to_z64_be(rom: bytes) -> bytes:
    """Normalize ROM to standard big-endian .z64 byte order based on magic."""
    if len(rom) < 4: return rom
    magic = struct.unpack(">I", rom[0:4])[0]
    if magic == 0x80371240:  # .z64 big endian
        return rom
    elif magic == 0x40123780:  # .n64 little endian
        out = bytearray(rom)
        for i in range(0, len(rom), 4):
            out[i:i+4] = rom[i:i+4][::-1]
        return bytes(out)
    elif magic == 0x37804012:  # .v64 byteswapped
        out = bytearray(rom)
        for i in range(0, len(rom), 2):
            out[i], out[i+1] = rom[i+1], rom[i]
        return bytes(out)
    else: # Unknown magic, assume big-endian
        return rom

# --------------------------- RCP (MMIO) ---------------------------

class RCP:
    """
    Reality Co-Processor register stubs & tiny DMA helpers.
    This is deliberately incomplete but sufficient for many boot stubs
    to probe registers without exploding.
    """

    # --- Address windows (physical) ---
    # DMEM: 0x0400_0000 - 0x0400_0FFF (4 KiB)
    # IMEM: 0x0400_1000 - 0x0400_1FFF (4 KiB)
    # SP  : 0x0404_0000 - 0x0404_001F (registers)
    # DP  : 0x0410_0000 - 0x0410_000F (registers)
    # VI  : 0x0440_0000 - 0x0440_0034 (registers)
    # AI  : 0x0450_0000 - 0x0450_0014 (registers)
    # PI  : 0x0460_0000 - 0x0460_0034 (registers subset)
    # SI  : 0x0480_0000 - 0x0480_0018 (registers)
    SP_BASE  = 0x04040000
    DP_BASE  = 0x04100000
    VI_BASE  = 0x04400000
    AI_BASE  = 0x04500000
    PI_BASE  = 0x04600000
    SI_BASE  = 0x04800000

    def __init__(self, mem_read_bytes, mem_write_bytes, logger):
        self.log = logger
        # Accessors to system memory (RDRAM/ROM etc.)
        self._mem_read_bytes = mem_read_bytes
        self._mem_write_bytes = mem_write_bytes

        # --- SP ---
        self.sp_regs = {
            0x00: 0,   # SP_MEM_ADDR
            0x04: 0,   # SP_DRAM_ADDR
            0x08: 0,   # SP_RD_LEN
            0x0C: 0,   # SP_WR_LEN
            0x10: 0,   # SP_STATUS
            0x14: 0,   # SP_DMA_FULL
            0x18: 0,   # SP_DMA_BUSY
            0x1C: 0,   # SP_SEMAPHORE
        }

        # --- DP (RDP) ---
        self.dp_regs = {
            0x00: 0, # DP_START
            0x04: 0, # DP_END
            0x08: 0, # DP_CURRENT
            0x0C: 0, # DP_STATUS
        }

        # --- VI ---
        self.vi_regs = {
            0x00: 0, # VI_STATUS
            0x04: 0, # VI_ORIGIN
            0x08: 0, # VI_WIDTH
            0x0C: 0, # VI_INTR
            0x10: 0, # VI_CURRENT
            0x14: 0, # VI_BURST
            0x18: 0, # VI_V_SYNC
            0x1C: 0, # VI_H_SYNC
            0x20: 0, # VI_LEAP
            0x24: 0, # VI_H_START
            0x28: 0, # VI_V_START
            0x2C: 0, # VI_V_BURST
            0x30: 0, # VI_X_SCALE
            0x34: 0, # VI_Y_SCALE
        }

        # --- AI --- (stub)
        self.ai_regs = {
            0x00: 0, # AI_DRAM_ADDR
            0x04: 0, # AI_LEN
            0x08: 0, # AI_CONTROL
            0x0C: 0, # AI_STATUS
            0x10: 0, # AI_DACRATE
            0x14: 0, # AI_BITRATE
        }

        # --- PI ---
        self.pi_regs = {
            0x00: 0, # PI_DRAM_ADDR
            0x04: 0, # PI_CART_ADDR
            0x08: 0, # PI_RD_LEN
            0x0C: 0, # PI_WR_LEN
            0x10: 0, # PI_STATUS
            0x14: 0, # PI_BSD_DOM1_LAT
            0x18: 0, # PI_BSD_DOM1_PWD
            0x1C: 0, # PI_BSD_DOM1_PGS
            0x20: 0, # PI_BSD_DOM1_RLS
            0x24: 0, # PI_BSD_DOM2_LAT
            0x28: 0, # PI_BSD_DOM2_PWD
            0x2C: 0, # PI_BSD_DOM2_PGS
            0x30: 0, # PI_BSD_DOM2_RLS
        }

        # --- SI --- (stub)
        self.si_regs = {
            0x00: 0, # SI_DRAM_ADDR
            0x04: 0, # SI_PIF_ADDR_RD64B
            0x08: 0, # SI_PIF_ADDR_WR64B
            0x0C: 0, # SI_STATUS
            0x10: 0, # SI_STATUS (alias/clear)
            0x18: 0, # SI_DUMMY
        }

        # Tk image cache (for VI preview)
        self._tk_image = None
        self._last_fb_key = None

    # ------------------- Helpers -------------------

    @staticmethod
    def _rgba5551_to_hex(px):
        r = (px >> 11) & 0x1F
        g = (px >> 6) & 0x1F
        b = (px >> 1) & 0x1F
        # scale 5-bit to 8-bit
        r = (r << 3) | (r >> 2)
        g = (g << 3) | (g >> 2)
        b = (b << 3) | (b >> 2)
        return f"#{r:02x}{g:02x}{b:02x}"

    # ------------------- MMIO read/write -------------------

    def read32(self, phys):
        if self.SP_BASE <= phys < self.SP_BASE + 0x20:
            off = phys - self.SP_BASE
            return self.sp_regs.get(off, 0)
        if self.DP_BASE <= phys < self.DP_BASE + 0x10:
            off = phys - self.DP_BASE
            return self.dp_regs.get(off, 0)
        if self.VI_BASE <= phys < self.VI_BASE + 0x38:
            off = phys - self.VI_BASE
            return self.vi_regs.get(off, 0)
        if self.AI_BASE <= phys < self.AI_BASE + 0x18:
            off = phys - self.AI_BASE
            return self.ai_regs.get(off, 0)
        if self.PI_BASE <= phys < self.PI_BASE + 0x34:
            off = phys - self.PI_BASE
            return self.pi_regs.get(off, 0)
        if self.SI_BASE <= phys < self.SI_BASE + 0x1C:
            off = phys - self.SI_BASE
            return self.si_regs.get(off, 0)
        return None  # not an RCP register

    def write32(self, phys, val):
        # SP
        if self.SP_BASE <= phys < self.SP_BASE + 0x20:
            off = phys - self.SP_BASE
            self.sp_regs[off] = u32(val)
            # DMA triggers
            if off == 0x08:  # SP_RD_LEN (RDRAM -> SP)
                self._sp_dma(read=True)
            elif off == 0x0C: # SP_WR_LEN (SP -> RDRAM)
                self._sp_dma(read=False)
            elif off == 0x10: # SP_STATUS (simple acknowledge/clear bits)
                # Just clear on write to keep software moving
                self.sp_regs[0x10] = 0
            return True

        # DP
        if self.DP_BASE <= phys < self.DP_BASE + 0x10:
            off = phys - self.DP_BASE
            self.dp_regs[off] = u32(val)
            if off == 0x04:  # DP_END updates CURRENT
                self.dp_regs[0x08] = self.dp_regs[0x04]
            return True

        # VI
        if self.VI_BASE <= phys < self.VI_BASE + 0x38:
            off = phys - self.VI_BASE
            self.vi_regs[off] = u32(val)
            return True

        # AI (no real audio; accept writes)
        if self.AI_BASE <= phys < self.AI_BASE + 0x18:
            off = phys - self.AI_BASE
            self.ai_regs[off] = u32(val)
            # Clear FIFO full/busy bits loosely
            if off == 0x0C:
                self.ai_regs[0x0C] = 0
            return True

        # PI
        if self.PI_BASE <= phys < self.PI_BASE + 0x34:
            off = phys - self.PI_BASE
            self.pi_regs[off] = u32(val)
            if off == 0x08:  # PI_RD_LEN (ROM -> RDRAM)
                self._pi_dma(cart_to_dram=True)
            elif off == 0x0C: # PI_WR_LEN (RDRAM -> ROM; ignore safely)
                self._pi_dma(cart_to_dram=False)
            elif off == 0x10:
                self.pi_regs[0x10] = 0  # clear status
            return True

        # SI
        if self.SI_BASE <= phys < self.SI_BASE + 0x1C:
            off = phys - self.SI_BASE
            self.si_regs[off] = u32(val)
            if off in (0x04, 0x08):  # pretend PIF DMA completes
                self.si_regs[0x0C] = 0  # clear status
            return True

        return False

    # ------------------- DMA implementations (very small) -------------------

    def _pi_dma(self, cart_to_dram: bool):
        dram = self.pi_regs[0x00] & 0x00FFFFFF
        cart = self.pi_regs[0x04] & 0x00FFFFFF
        # LEN registers store "length - 1" per HW. Be tolerant.
        length = (self.pi_regs[0x08 if cart_to_dram else 0x0C] & 0x00FFFFFF) + 1
        if cart_to_dram:
            data = self._mem_read_bytes(0x10000000 + cart, length)
            self._mem_write_bytes(dram, data)
            self.log(f"PI DMA ROM→RDRAM: cart=0x{cart:06X} dram=0x{dram:06X} len={length}")
        else:
            # Ignored safely; cartridges are read-only. Log and continue.
            self.log(f"PI DMA (ignored) RDRAM→ROM: dram=0x{dram:06X} cart=0x{cart:06X} len={length}")
        # Clear busy
        self.pi_regs[0x10] = 0

    def _sp_dma(self, read: bool):
        mem_addr  = self.sp_regs[0x00] & 0x1FFF  # DMEM/IMEM windowed
        dram_addr = self.sp_regs[0x04] & 0x00FFFFFF
        length    = (self.sp_regs[0x08 if read else 0x0C] & 0x00FFFF) + 1

        # Select DMEM (0x0000-0x0FFF) or IMEM (0x1000-0x1FFF)
        is_imem = (mem_addr & 0x1000) != 0
        mem_addr &= 0x0FFF
        sp_phys = (0x04001000 if is_imem else 0x04000000) + mem_addr

        if read:
            # RDRAM -> SP
            data = self._mem_read_bytes(dram_addr, length)
            self._mem_write_bytes(sp_phys, data)
            self.log(f"SP DMA RDRAM→SP({'IMEM' if is_imem else 'DMEM'}): dram=0x{dram_addr:06X} sp=0x{mem_addr:04X} len={length}")
        else:
            # SP -> RDRAM
            data = self._mem_read_bytes(sp_phys, length)
            self._mem_write_bytes(dram_addr, data)
            self.log(f"SP DMA SP({'IMEM' if is_imem else 'DMEM'})→RDRAM: sp=0x{mem_addr:04X} dram=0x{dram_addr:06X} len={length}")

        # Clear simple busy flags
        self.sp_regs[0x18] = 0  # DMA_BUSY
        self.sp_regs[0x14] = 0  # DMA_FULL

    # ------------------- VI preview -------------------

    def render_vi_to_canvas(self, canvas, rdram):
        """
        A tiny VI-backed framebuffer preview:
        - Assumes 16bpp RGBA5551 linear framebuffer at VI_ORIGIN.
        - Uses VI_WIDTH for width; height guessed (240) unless overridden by Y_SCALE.
        - Purely for developer feedback; NOT accurate timing-wise.
        """
        origin = self.vi_regs[0x04]
        width  = self.vi_regs[0x08] & 0x7FFF
        if origin == 0 or width == 0:
            return  # nothing to show

        # Height guess: 240 unless Y_SCALE encodes a hint; accept 32..480
        height = 240
        yscale = self.vi_regs[0x34]
        if yscale:
            # crude heuristic: upper 16 bits as numerator; don't rely on it
            # clamp to sensible range
            maybe_h = (yscale >> 16) & 0x3FF
            if 32 <= maybe_h <= 480:
                height = maybe_h

        # Bounds and safety
        total_bytes = width * height * 2
        if origin + total_bytes > len(rdram):
            # Guard against bogus pointers
            height = max(1, (len(rdram) - origin) // (width * 2))

        if height <= 0 or width <= 0:
            return

        key = (origin, width, height)
        if self._tk_image is None or self._last_fb_key != key:
            self._tk_image = tk.PhotoImage(width=width, height=height)
            self._last_fb_key = key

        # Build a scanline-at-a-time string for Tk put()
        # For performance in Python, limit to some cap in preview
        max_h = min(height, 300)  # don't overwhelm Tk
        for y in range(max_h):
            row_off = origin + y * width * 2
            scan = []
            for x in range(width):
                px = struct.unpack_from(">H", rdram, row_off + x*2)[0]
                scan.append(self._rgba5551_to_hex(px))
            self._tk_image.put(" ".join(scan), to=(0, y))

        canvas.delete("fb")
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        # simple letterbox fit
        scale = min(cw / width, ch / max_h) if width and max_h else 1.0
        w = int(width * scale)
        h = int(max_h * scale)
        # place centered
        x0 = (cw - w)//2
        y0 = (ch - h)//2
        canvas.create_image(x0, y0, image=self._tk_image, anchor="nw", tags="fb")

# --------------------------- Memory Subsystem ---------------------------

class Memory:
    """
    Simplified N64 memory map.
      - RDRAM:      0x00000000 - 0x007FFFFF (8MB)
      - SP DMEM:    0x04000000 - 0x04000FFF (4KB)
      - SP IMEM:    0x04001000 - 0x04001FFF (4KB)
      - Cartridge:  0x10000000 - 0x1FBFFFFF (ROM, read-only)
      - RCP regs:   SP/DP/VI/AI/PI/SI
    """
    def __init__(self, logger_func):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)
        self.rom_be = None
        self.rom_size = 0
        self.log = logger_func

        # RCP MMIO
        self.rcp = RCP(self._raw_read_bytes, self._raw_write_bytes, logger_func)

    def load_rom(self, rom_data_be: bytes):
        self.rom_be = rom_data_be
        self.rom_size = len(rom_data_be)

    @staticmethod
    def virt_to_phys(addr: int) -> int:
        return addr & 0x1FFFFFFF

    # Low-level raw byte access without MMIO interpretation (helpers for DMA)
    def _raw_read_bytes(self, phys, length) -> bytes:
        if 0x00000000 <= phys < 0x00800000 and phys + length <= 0x00800000:
            return bytes(self.rdram[phys:phys+length])
        if 0x04000000 <= phys < 0x04001000 and phys + length <= 0x04001000:
            return bytes(self.sp_dmem[phys-0x04000000: phys-0x04000000+length])
        if 0x04001000 <= phys < 0x04002000 and phys + length <= 0x04002000:
            return bytes(self.sp_imem[phys-0x04001000: phys-0x04001000+length])
        if 0x10000000 <= phys < 0x10000000 + self.rom_size and phys + length <= 0x10000000 + self.rom_size:
            start = phys - 0x10000000
            return bytes(self.rom_be[start:start+length])
        # Default empty
        return bytes([0] * length)

    def _raw_write_bytes(self, phys, data: bytes):
        length = len(data)
        if 0x00000000 <= phys < 0x00800000 and phys + length <= 0x00800000:
            self.rdram[phys:phys+length] = data
            return
        if 0x04000000 <= phys < 0x04001000 and phys + length <= 0x04001000:
            self.sp_dmem[phys-0x04000000: phys-0x04000000+length] = data
            return
        if 0x04001000 <= phys < 0x04002000 and phys + length <= 0x04002000:
            self.sp_imem[phys-0x04001000: phys-0x04001000+length] = data
            return
        # Writes to ROM ignored

    def _read_range(self, phys_addr, size):
        if 0x00000000 <= phys_addr < 0x00800000:
            return self.rdram, phys_addr
        elif 0x04000000 <= phys_addr < 0x04001000:
            return self.sp_dmem, phys_addr - 0x04000000
        elif 0x04001000 <= phys_addr < 0x04002000:
            return self.sp_imem, phys_addr - 0x04001000
        elif 0x10000000 <= phys_addr < (0x10000000 + self.rom_size):
            return self.rom_be, phys_addr - 0x10000000
        else:
            return None, 0

    # MMIO-aware loads/stores (32-bit)
    def read_u32(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        # RCP register window?
        mmio = self.rcp.read32(phys)
        if mmio is not None:
            return mmio
        mem, offset = self._read_range(phys, 4)
        if mem and offset + 3 < len(mem):
            return struct.unpack_from(">I", mem, offset)[0]
        return 0

    def write_u32(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFFFFFFFF
        # RCP register window?
        if self.rcp.write32(phys, val):
            return
        if 0x00000000 <= phys < 0x00800000:
            struct.pack_into(">I", self.rdram, phys, val)
        elif 0x04000000 <= phys < 0x04001000:
            struct.pack_into(">I", self.sp_dmem, phys - 0x04000000, val)
        elif 0x04001000 <= phys < 0x04002000:
            struct.pack_into(">I", self.sp_imem, phys - 0x04001000, val)

    def read_u16(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        # MMIO: compose from 32-bit for safety
        mmio = self.rcp.read32(phys & ~3)
        if mmio is not None:
            shift = (2 - (phys & 2)) * 8
            return (mmio >> shift) & 0xFFFF
        mem, offset = self._read_range(phys, 2)
        if mem and offset + 1 < len(mem):
            return struct.unpack_from(">H", mem, offset)[0]
        return 0

    def write_u16(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFFFF
        # MMIO not supported at 16-bit granularity; map via read-modify-write if needed
        if self.rcp.read32(phys & ~3) is not None:
            base = phys & ~3
            cur = self.read_u32(base)
            shift = (2 - (phys & 2)) * 8
            mask = 0xFFFF << shift
            newv = (cur & ~mask) | ((val & 0xFFFF) << shift)
            self.write_u32(base, newv)
            return
        if 0x00000000 <= phys < 0x00800000:
            struct.pack_into(">H", self.rdram, phys, val)
        elif 0x04000000 <= phys < 0x04001000:
            struct.pack_into(">H", self.sp_dmem, phys - 0x04000000, val)
        elif 0x04001000 <= phys < 0x04002000:
            struct.pack_into(">H", self.sp_imem, phys - 0x04001000, val)

    def read_u8(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        mmio = self.rcp.read32(phys & ~3)
        if mmio is not None:
            shift = (3 - (phys & 3)) * 8
            return (mmio >> shift) & 0xFF
        mem, offset = self._read_range(phys, 1)
        if mem and offset < len(mem):
            return mem[offset]
        return 0
    
    def write_u8(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFF
        if self.rcp.read32(phys & ~3) is not None:
            base = phys & ~3
            cur = self.read_u32(base)
            shift = (3 - (phys & 3)) * 8
            mask = 0xFF << shift
            newv = (cur & ~mask) | ((val & 0xFF) << shift)
            self.write_u32(base, newv)
            return
        if 0x00000000 <= phys < 0x00800000:
            self.rdram[phys] = val
        elif 0x04000000 <= phys < 0x04001000:
            self.sp_dmem[phys - 0x04000000] = val
        elif 0x04001000 <= phys < 0x04002000:
            self.sp_imem[phys - 0x04001000] = val

    def load_boot_stub_to_sp_dmem(self):
        if not self.rom_be or self.rom_size < 0x1000: return False
        src = self.rom_be[0x40:0x1000]
        self.sp_dmem = bytearray(0x1000) # Clear
        self.sp_dmem[0x40:0x40+len(src)] = src
        return True

# --------------------------- MIPS R4300i CPU Core ---------------------------

class MIPSCPU:
    """Expanded MIPS R4300i core for N64 boot stubs."""
    def __init__(self, memory: Memory, logger_func):
        self.mem = memory
        self.log = logger_func
        self.reset()

    def reset(self):
        self.gpr = [0] * 32
        self.hi, self.lo = 0, 0
        self.pc = 0xA4000040  # KSEG1 alias for SP DMEM + 0x40
        self.cp0 = [0] * 32
        
        # Set some default 'good' values for COP0 registers
        self.cp0[12] = 0x10000000 # Status Register (SR)
        self.cp0[15] = 0x00000B00 # PRId Register
        self.cp0[16] = 0x0006E463 # Config Register
        
        self.gpr[29] = 0xA4001FF0 # Initial SP
        self.running = False
        self.instructions = 0
        self.branch_taken = False
        self.branch_target = 0
        self.delay_slot_pc = 0

    def _read_gpr(self, i): return 0 if i == 0 else u32(self.gpr[i])
    def _write_gpr(self, i, val):
        if i != 0: self.gpr[i] = u32(val)

    def step(self):
        if not self.running: return

        # Execute instruction from branch target if one was taken
        if self.branch_taken:
            self.pc = u32(self.branch_target)
            self.branch_taken = False
        
        current_pc = self.pc
        try:
            instr = self.mem.read_u32(current_pc)
        except Exception as e:
            self.log(f"FETCH ERROR at PC=0x{current_pc:08X}: {e}")
            self.running = False
            return

        self.pc = u32(current_pc + 4)
        self.delay_slot_pc = self.pc
        
        self.decode_and_execute(instr)
        self.instructions += 1

    def decode_and_execute(self, instr):
        op = bits(instr, 26, 31)
        rs = bits(instr, 21, 25)
        rt = bits(instr, 16, 20)
        rd = bits(instr, 11, 15)
        sa = bits(instr, 6, 10)
        fn = bits(instr, 0, 5)
        imm = bits(instr, 0, 15)
        simm = sext16(imm)
        target = bits(instr, 0, 25)

        rs_val = self._read_gpr(rs)
        rt_val = self._read_gpr(rt)
        addr = u32(rs_val + simm)
        
        # --- Decode ---
        if op == 0x00:  # SPECIAL
            if   fn == 0x00: self._write_gpr(rd, u32(rt_val << sa))  # SLL
            elif fn == 0x02: self._write_gpr(rd, u32(rt_val >> sa))  # SRL
            elif fn == 0x03: self._write_gpr(rd, u32(s32(rt_val) >> sa))  # SRA
            elif fn == 0x04: self._write_gpr(rd, u32(rt_val << (rs_val & 31)))  # SLLV
            elif fn == 0x06: self._write_gpr(rd, u32(rt_val >> (rs_val & 31)))  # SRLV
            elif fn == 0x07: self._write_gpr(rd, u32(s32(rt_val) >> (rs_val & 31)))  # SRAV
            elif fn == 0x08: self.branch_to(rs_val)  # JR
            elif fn == 0x09: self._write_gpr(rd if rd != 0 else 31, self.delay_slot_pc + 4); self.branch_to(rs_val) # JALR
            elif fn == 0x0C: pass  # SYSCALL (ignored)
            elif fn == 0x10: self._write_gpr(rd, self.hi) # MFHI
            elif fn == 0x12: self._write_gpr(rd, self.lo) # MFLO
            elif fn == 0x11: self.hi = rs_val # MTHI
            elif fn == 0x13: self.lo = rs_val # MTLO
            elif fn == 0x18: # MULT (signed)
                res = s32(rs_val) * s32(rt_val)
                self.hi, self.lo = u32((res >> 32) & 0xFFFFFFFF), u32(res & 0xFFFFFFFF)
            elif fn == 0x19: # MULTU
                res = rs_val * rt_val
                self.hi, self.lo = u32(res >> 32), u32(res)
            elif fn == 0x1A: # DIV (signed)
                if rt_val != 0:
                    self.lo = u32(int(s32(rs_val) / s32(rt_val)))
                    self.hi = u32(int(s32(rs_val) % s32(rt_val)))
            elif fn == 0x1B: # DIVU
                if rt_val != 0:
                    self.lo = u32(rs_val // rt_val)
                    self.hi = u32(rs_val % rt_val)
            elif fn == 0x20: # ADD (trap on overflow ignored for skeleton)
                self._write_gpr(rd, u32(s32(rs_val) + s32(rt_val)))
            elif fn == 0x21: self._write_gpr(rd, u32(rs_val + rt_val))  # ADDU
            elif fn == 0x22: # SUB (trap ignored)
                self._write_gpr(rd, u32(s32(rs_val) - s32(rt_val)))
            elif fn == 0x23: self._write_gpr(rd, u32(rs_val - rt_val))  # SUBU
            elif fn == 0x24: self._write_gpr(rd, rs_val & rt_val)  # AND
            elif fn == 0x25: self._write_gpr(rd, rs_val | rt_val)  # OR
            elif fn == 0x26: self._write_gpr(rd, rs_val ^ rt_val)  # XOR
            elif fn == 0x27: self._write_gpr(rd, u32(~(rs_val | rt_val))) # NOR
            elif fn == 0x2A: self._write_gpr(rd, 1 if s32(rs_val) < s32(rt_val) else 0)  # SLT
            elif fn == 0x2B: self._write_gpr(rd, 1 if rs_val < rt_val else 0)  # SLTU

        elif op == 0x01: # REGIMM
            if rt == 0x00:  # BLTZ
                if s32(rs_val) < 0: self.branch_relative(simm)
            elif rt == 0x01: # BGEZ
                if s32(rs_val) >= 0: self.branch_relative(simm)

        elif op == 0x02:  # J
            self.branch_jump(target)
        elif op == 0x03:  # JAL
            self._write_gpr(31, self.delay_slot_pc + 4)
            self.branch_jump(target)
        elif op == 0x04: # BEQ
            if rs_val == rt_val: self.branch_relative(simm)
        elif op == 0x05: # BNE
            if rs_val != rt_val: self.branch_relative(simm)
        elif op == 0x06: # BLEZ
            if s32(rs_val) <= 0: self.branch_relative(simm)
        elif op == 0x07: # BGTZ
            if s32(rs_val) > 0: self.branch_relative(simm)
        elif op == 0x08: # ADDI (trap ignored)
            self._write_gpr(rt, u32(s32(rs_val) + s32(simm)))
        elif op == 0x09: self._write_gpr(rt, u32(rs_val + simm))  # ADDIU
        elif op == 0x0A: self._write_gpr(rt, 1 if s32(rs_val) < s32(simm) else 0)  # SLTI
        elif op == 0x0B: self._write_gpr(rt, 1 if rs_val < u32(simm) else 0)  # SLTIU
        elif op == 0x0C: self._write_gpr(rt, rs_val & imm)  # ANDI
        elif op == 0x0D: self._write_gpr(rt, rs_val | imm)  # ORI
        elif op == 0x0E: self._write_gpr(rt, rs_val ^ imm)  # XORI
        elif op == 0x0F: self._write_gpr(rt, imm << 16)  # LUI
        elif op == 0x10: # COP0
            if rs == 0x00: self._write_gpr(rt, self.cp0[rd])  # MFC0
            elif rs == 0x04: self.cp0[rd] = rt_val # MTC0
        elif op == 0x20: self._write_gpr(rt, sext16(self.mem.read_u8(addr)))   # LB
        elif op == 0x21: self._write_gpr(rt, sext16(self.mem.read_u16(addr)))  # LH
        elif op == 0x22: # LWL - Load Word Left
            shift = (addr & 3) * 8
            mask = 0xFFFFFFFF << shift
            data = self.mem.read_u32(addr & ~3)
            self._write_gpr(rt, (rt_val & ~mask) | (u32(data << shift) & mask))
        elif op == 0x23: self._write_gpr(rt, self.mem.read_u32(addr))          # LW
        elif op == 0x24: self._write_gpr(rt, self.mem.read_u8(addr))           # LBU
        elif op == 0x25: self._write_gpr(rt, self.mem.read_u16(addr))          # LHU
        elif op == 0x26: # LWR - Load Word Right
            shift = (3 - (addr & 3)) * 8
            mask = 0xFFFFFFFF >> shift
            data = self.mem.read_u32(addr & ~3)
            self._write_gpr(rt, (rt_val & ~mask) | (u32(data >> shift) & mask))
        elif op == 0x28: self.mem.write_u8(addr, rt_val)   # SB
        elif op == 0x29: self.mem.write_u16(addr, rt_val)  # SH
        elif op == 0x2A: # SWL - Store Word Left
            shift = (addr & 3) * 8
            mask = 0xFFFFFFFF >> shift
            mem_val = self.mem.read_u32(addr & ~3)
            data = (mem_val & mask) | (u32(rt_val >> shift) & ~mask)
            self.mem.write_u32(addr & ~3, data)
        elif op == 0x2B: self.mem.write_u32(addr, rt_val)  # SW
        elif op == 0x2E: # SWR - Store Word Right
            shift = (3 - (addr & 3)) * 8
            mask = 0xFFFFFFFF << shift
            mem_val = self.mem.read_u32(addr & ~3)
            data = (mem_val & mask) | (u32(rt_val << shift) & ~mask)
            self.mem.write_u32(addr & ~3, data)
        elif op == 0x2F: pass # CACHE (NOP)
        else:
            # Keep running; many optional opcodes unimplemented
            self.log(f"Unimpl opcode 0x{op:X} at PC=0x{self.pc-4:08X}")

    def branch_to(self, target):
        self.branch_taken = True
        self.branch_target = target

    def branch_relative(self, simm):
        self.branch_taken = True
        self.branch_target = u32(self.delay_slot_pc + (simm << 2))
        
    def branch_jump(self, target):
        self.branch_taken = True
        self.branch_target = u32((self.delay_slot_pc & 0xF0000000) | (target << 2))

# --------------------------- UI and Main Application ---------------------------

class MIPSEMU2:
    GPR_NAMES = [
        "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
        "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
        "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"
    ]
    
    def __init__(self, root):
        self.root = root
        self.root.title("MIPSEMU2.1 - N64 Emulator Skeleton (RCP Stubs)")
        self.root.configure(bg="#2E2E2E")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.memory = Memory(self.log)
        self.cpu = MIPSCPU(self.memory, self.log)
        self.rom_header = None
        self.running = False

        self.setup_ui()
        self.log("MIPSEMU2.1 Initialized. Load a ROM to begin.")
        self.update_ui_state()

    def configure_styles(self):
        # Configure styles for a dark theme
        self.style.configure('.', background='#2E2E2E', foreground='white', fieldbackground='#3c3c3c')
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TButton', background='#4A4A4A', foreground='white', borderwidth=1)
        self.style.map('TButton', background=[('active', '#6A6A6A')])
        self.style.configure('TLabel', background='#2E2E2E', foreground='white')
        self.style.configure('Treeview', rowheight=20, fieldbackground='#3c3c3c', background='#3c3c3c', foreground='white')
        self.style.configure('Treeview.Heading', background='#4A4A4A', foreground='white', font=('Consolas', 10, 'bold'))
        self.style.map('Treeview', background=[('selected', '#0078D7')])

    def setup_ui(self):
        # Top controls
        top_frame = ttk.Frame(self.root, padding="5")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(top_frame, text="Open ROM", command=self.cmd_open_rom).pack(side=tk.LEFT, padx=2)
        self.start_btn = ttk.Button(top_frame, text="Start", command=self.start, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(top_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.step_btn = ttk.Button(top_frame, text="Step", command=self.step_once, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, padx=2)

        # Main content area
        main_paned_window = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top pane with screen and registers
        top_pane = ttk.PanedWindow(main_paned_window, orient=tk.HORIZONTAL)
        main_paned_window.add(top_pane, weight=4)

        # "Screen" canvas
        self.canvas = tk.Canvas(top_pane, bg="black", highlightthickness=0)
        top_pane.add(self.canvas, weight=3)

        # Register view
        reg_frame = ttk.Frame(top_pane)
        top_pane.add(reg_frame, weight=1)
        
        self.setup_register_view(reg_frame)

        # Log view
        log_frame = ttk.Frame(main_paned_window, padding="2")
        log_frame.pack_propagate(False)
        main_paned_window.add(log_frame, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, bg="#1E1E1E", fg="#A0D0A0",
                                                  font=("Consolas", 9), relief=tk.SOLID, borderwidth=1)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_register_view(self, parent):
        ttk.Label(parent, text="CPU Registers", font=('Consolas', 12, 'bold')).pack(pady=5)
        
        # PC, HI, LO display
        self.cpu_state_labels = {}
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, padx=5)
        for i, name in enumerate(['PC', 'HI', 'LO', 'Instr #']):
            ttk.Label(info_frame, text=f"{name}:", font=('Consolas', 10, 'bold')).grid(row=i, column=0, sticky='w')
            label = ttk.Label(info_frame, text="0x00000000", font=('Consolas', 10))
            label.grid(row=i, column=1, sticky='w', padx=5)
            self.cpu_state_labels[name] = label

        # GPR Treeview
        self.reg_tree = ttk.Treeview(parent, columns=('name','val'), show='headings', height=32)
        self.reg_tree.heading('name', text='Reg')
        self.reg_tree.heading('val', text='Value')
        self.reg_tree.column('name', width=80, anchor='w')
        self.reg_tree.column('val', width=120, anchor='w')
        self.reg_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i in range(32):
            self.reg_tree.insert('', 'end', iid=i, values=(f"r{i} ({self.GPR_NAMES[i]})", f"0x{0:08X}",))
        
        # Add tags for alternating colors
        self.reg_tree.tag_configure('oddrow', background='#333333')
        self.reg_tree.tag_configure('evenrow', background='#3c3c3c')

    def log(self, msg):
        if not hasattr(self, 'log_text'): return
        t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.insert(tk.END, f"[{t}] {msg}\n")
        self.log_text.see(tk.END)

    def load_rom(self, path):
        try:
            self.stop()
            self.log(f"Loading ROM: {os.path.basename(path)}")
            data = Path(path).read_bytes()
            be = normalize_rom_to_z64_be(data)
            self.memory.load_rom(be)
            self.memory.load_boot_stub_to_sp_dmem()
            
            self.rom_header = ROMHeader(be)
            self.cpu.reset()
            
            self.log(f"ROM: '{self.rom_header.name}'")
            self.log(f"  ID: {self.rom_header.game_id} | Region: {self.rom_header.region}")
            self.log(f"  CRC1: {self.rom_header.crc1:08X} | CRC2: {self.rom_header.crc2:08X}")
            self.log("Boot stub copied to SP DMEM. PC set to 0xA4000040.")
            self.update_ui_state(loaded=True)
            self.render_once()
        except Exception as e:
            self.log(f"ERROR loading ROM: {e}")
            self.update_ui_state(loaded=False)

    def cmd_open_rom(self):
        fn = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[("N64 ROMs", "*.z64 *.n64 *.v64 *.rom"), ("All files", "*.*")]
        )
        if fn:
            self.load_rom(fn)

    def start(self):
        if not self.memory.rom_be:
            self.log("No ROM loaded."); return
        self.running = True
        self.cpu.running = True
        self.update_ui_state(running=True)
        self.log("CPU execution started.")
        threading.Thread(target=self.emu_loop, daemon=True).start()
        self.render_loop()

    def stop(self):
        self.running = False
        self.cpu.running = False
        self.update_ui_state(running=False)
        self.log("CPU execution stopped.")

    def step_once(self):
        if not self.memory.rom_be:
            self.log("No ROM loaded."); return
        if self.running: self.stop()
        self.cpu.running = True
        self.cpu.step()
        self.cpu.running = False
        self.render_once()

    def update_ui_state(self, loaded=None, running=None):
        if loaded is not None:
            state = tk.NORMAL if loaded else tk.DISABLED
            self.start_btn.config(state=state)
            self.step_btn.config(state=state)

        if running is not None:
            self.start_btn.config(state=tk.DISABLED if running else tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL if running else tk.DISABLED)
            self.step_btn.config(state=tk.DISABLED if running else tk.NORMAL)

            # Re-enable start/step if a ROM is loaded but we're stopping
            if not running and self.rom_header:
                self.start_btn.config(state=tk.NORMAL)
                self.step_btn.config(state=tk.NORMAL)

    def emu_loop(self):
        try:
            while self.running:
                # Execute a batch of instructions to improve performance
                for _ in range(2000):
                    if not self.running: break
                    self.cpu.step()
                time.sleep(0.001) # Small sleep to yield thread
        except Exception as e:
            self.log(f"CPU Exception at PC=0x{self.cpu.pc:08X}: {e}")
            self.stop()

    def render_once(self):
        # Update registers
        self.cpu_state_labels['PC'].config(text=f"0x{self.cpu.pc:08X}")
        self.cpu_state_labels['HI'].config(text=f"0x{self.cpu.hi:08X}")
        self.cpu_state_labels['LO'].config(text=f"0x{self.cpu.lo:08X}")
        self.cpu_state_labels['Instr #'].config(text=f"{self.cpu.instructions}")

        for i in range(32):
            self.reg_tree.item(i, values=(f"r{i} ({self.GPR_NAMES[i]})", f"0x{self.cpu.gpr[i]:08X}",), 
                               tags=('evenrow' if i % 2 == 0 else 'oddrow'))

        # Render VI-backed framebuffer preview (if any)
        self.canvas.delete("border")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_rectangle(2, 2, w, h, outline="#00A0A0", width=2, tags="border")

        self.memory.rcp.render_vi_to_canvas(self.canvas, self.memory.rdram)

        if self.rom_header:
            title = self.rom_header.name
            self.canvas.create_text(w/2, h*0.1, text=title, font=("Arial", 18, "bold"), fill="#FFFFFF", tags="title")
            
            info_text = (
                f"Game ID: {self.rom_header.game_id}  |  Region: {self.rom_header.region}  |  "
                f"CPU: {'Running' if self.running else 'Stopped'}"
            )
            self.canvas.create_text(w/2, h*0.18, text=info_text, font=("Consolas", 11), fill="#A0A0A0", justify=tk.CENTER, tags="subtitle")
            
            self.canvas.create_text(w/2, h-18, text="MIPSEMU2.1 - Display Stub (VI Preview)", font=("Arial", 10), fill="#555555", tags="footer")

    def render_loop(self):
        if self.running:
            self.render_once()
            self.root.after(16, self.render_loop) # ~60Hz
        else:
            self.render_once() # Final render on stop

# --------------------------- Main Execution ---------------------------

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    min_w, min_h = 800, 600
    root.minsize(min_w, min_h)
    app = MIPSEMU2(root)
    root.mainloop()

if __name__ == "__main__":
    main()
