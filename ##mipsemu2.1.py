#!/usr/bin/env python3
"""
MIPSEMU2.0 — Darkness Revived II
Upgraded N64 Emulator Skeleton with Expanded MIPS R4300i Core
Python 3.10+ | Tkinter GUI

Notes
-----
- This version significantly enhances the original MIPSEMU1.0 skeleton.
- Full ROM region detection (NTSC-U, NTSC-J, PAL, etc.) is implemented.
- The CPU core supports a wider range of instructions, including common
  unaligned load/store operations (LWL/LWR, SWL/SWR) for better compatibility.
- The GUI has been completely redesigned with a modern look, featuring a
  resizable layout, a detailed live register view, and an improved display.
- While not a full emulator, this skeleton provides a much stronger foundation
  for learning about N64 architecture and boot processes.
- Endianness is fully normalized to big-endian (.z64 layout) internally.
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

# --------------------------- Memory Subsystem ---------------------------

class Memory:
    """
    Simplified N64 memory map.
      - RDRAM:      0x00000000 - 0x007FFFFF (8MB)
      - SP DMEM:    0x04000000 - 0x04000FFF (4KB)
      - Cartridge:  0x10000000 - 0x1FBFFFFF (ROM, read-only)
    """
    def __init__(self, logger_func):
        self.rdram = bytearray(8 * 1024 * 1024)
        self.sp_dmem = bytearray(0x1000)
        self.sp_imem = bytearray(0x1000)
        self.rom_be = None
        self.rom_size = 0
        self.log = logger_func

    def load_rom(self, rom_data_be: bytes):
        self.rom_be = rom_data_be
        self.rom_size = len(rom_data_be)

    @staticmethod
    def virt_to_phys(addr: int) -> int:
        return addr & 0x1FFFFFFF

    def _read_range(self, phys_addr, size):
        if 0x00000000 <= phys_addr < 0x00800000:
            return self.rdram, phys_addr
        elif 0x04000000 <= phys_addr < 0x04001000:
            return self.sp_dmem, phys_addr - 0x04000000
        elif 0x10000000 <= phys_addr < (0x10000000 + self.rom_size):
            return self.rom_be, phys_addr - 0x10000000
        else:
            return None, 0

    def read_u32(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        mem, offset = self._read_range(phys, 4)
        if mem and offset + 3 < len(mem):
            return struct.unpack_from(">I", mem, offset)[0]
        return 0

    def write_u32(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFFFFFFFF
        if 0x00000000 <= phys < 0x00800000:
            struct.pack_into(">I", self.rdram, phys, val)
        elif 0x04000000 <= phys < 0x04001000:
            struct.pack_into(">I", self.sp_dmem, phys - 0x04000000, val)

    def read_u16(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        mem, offset = self._read_range(phys, 2)
        if mem and offset + 1 < len(mem):
            return struct.unpack_from(">H", mem, offset)[0]
        return 0

    def write_u16(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFFFF
        if 0x00000000 <= phys < 0x00800000:
            struct.pack_into(">H", self.rdram, phys, val)
        elif 0x04000000 <= phys < 0x04001000:
            struct.pack_into(">H", self.sp_dmem, phys - 0x04000000, val)

    def read_u8(self, addr: int) -> int:
        phys = self.virt_to_phys(addr)
        mem, offset = self._read_range(phys, 1)
        if mem and offset < len(mem):
            return mem[offset]
        return 0
    
    def write_u8(self, addr: int, val: int):
        phys = self.virt_to_phys(addr)
        val &= 0xFF
        if 0x00000000 <= phys < 0x00800000:
            self.rdram[phys] = val
        elif 0x04000000 <= phys < 0x04001000:
            self.sp_dmem[phys - 0x04000000] = val

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

        # Execute instruction from delay slot if a branch was taken
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
            if fn == 0x00: self._write_gpr(rd, u32(rt_val << sa))  # SLL
            elif fn == 0x02: self._write_gpr(rd, u32(rt_val >> sa))  # SRL
            elif fn == 0x03: self._write_gpr(rd, u32(s32(rt_val) >> sa))  # SRA
            elif fn == 0x08: self.branch_to(rs_val)  # JR
            elif fn == 0x09: self._write_gpr(rd if rd != 0 else 31, self.delay_slot_pc + 4); self.branch_to(rs_val) # JALR
            elif fn == 0x10: self._write_gpr(rd, self.hi) # MFHI
            elif fn == 0x12: self._write_gpr(rd, self.lo) # MFLO
            elif fn == 0x11: self.hi = rs_val # MTHI
            elif fn == 0x13: self.lo = rs_val # MTLO
            elif fn == 0x19: # MULTU
                res = rs_val * rt_val
                self.hi, self.lo = u32(res >> 32), u32(res)
            elif fn == 0x1B: # DIVU
                if rt_val != 0:
                    self.lo = u32(rs_val // rt_val)
                    self.hi = u32(rs_val % rt_val)
            elif fn == 0x21: self._write_gpr(rd, u32(rs_val + rt_val))  # ADDU
            elif fn == 0x23: self._write_gpr(rd, u32(rs_val - rt_val))  # SUBU
            elif fn == 0x24: self._write_gpr(rd, rs_val & rt_val)  # AND
            elif fn == 0x25: self._write_gpr(rd, rs_val | rt_val)  # OR
            elif fn == 0x26: self._write_gpr(rd, rs_val ^ rt_val)  # XOR
            elif fn == 0x27: self._write_gpr(rd, u32(~(rs_val | rt_val))) # NOR
            elif fn == 0x2B: self._write_gpr(rd, 1 if rs_val < rt_val else 0)  # SLTU
        elif op == 0x01: # REGIMM
            if rt == 0x01: # BGEZ
                if s32(rs_val) >= 0: self.branch_relative(simm)
        elif op == 0x03: self._write_gpr(31, self.delay_slot_pc + 4); self.branch_jump(target) # JAL
        elif op == 0x04: # BEQ
            if rs_val == rt_val: self.branch_relative(simm)
        elif op == 0x05: # BNE
            if rs_val != rt_val: self.branch_relative(simm)
        elif op == 0x09: self._write_gpr(rt, u32(rs_val + simm))  # ADDIU
        elif op == 0x0B: self._write_gpr(rt, 1 if rs_val < u32(simm) else 0)  # SLTIU
        elif op == 0x0C: self._write_gpr(rt, rs_val & imm)  # ANDI
        elif op == 0x0D: self._write_gpr(rt, rs_val | imm)  # ORI
        elif op == 0x0E: self._write_gpr(rt, rs_val ^ imm)  # XORI
        elif op == 0x0F: self._write_gpr(rt, imm << 16)  # LUI
        elif op == 0x10: # COP0
            if rs == 0x00: self._write_gpr(rt, self.cp0[rd])  # MFC0
            elif rs == 0x04: self.cp0[rd] = rt_val # MTC0
        elif op == 0x23: self._write_gpr(rt, self.mem.read_u32(addr)) # LW
        elif op == 0x24: self._write_gpr(rt, self.mem.read_u8(addr)) # LBU
        elif op == 0x25: self._write_gpr(rt, self.mem.read_u16(addr)) # LHU
        elif op == 0x28: self.mem.write_u8(addr, rt_val) # SB
        elif op == 0x29: self.mem.write_u16(addr, rt_val) # SH
        elif op == 0x2B: self.mem.write_u32(addr, rt_val) # SW
        elif op == 0x20: self._write_gpr(rt, sext16(self.mem.read_u8(addr))) # LB
        elif op == 0x21: self._write_gpr(rt, sext16(self.mem.read_u16(addr))) # LH
        elif op == 0x22: # LWL - Load Word Left
            shift = (addr & 3) * 8
            mask = 0xFFFFFFFF << shift
            data = self.mem.read_u32(addr & ~3)
            self._write_gpr(rt, (rt_val & ~mask) | (u32(data << shift) & mask))
        elif op == 0x26: # LWR - Load Word Right
            shift = (3 - (addr & 3)) * 8
            mask = 0xFFFFFFFF >> shift
            data = self.mem.read_u32(addr & ~3)
            self._write_gpr(rt, (rt_val & ~mask) | (u32(data >> shift) & mask))
        elif op == 0x2A: # SWL - Store Word Left
            shift = (addr & 3) * 8
            mask = 0xFFFFFFFF >> shift
            mem_val = self.mem.read_u32(addr & ~3)
            data = (mem_val & mask) | (u32(rt_val >> shift) & ~mask)
            self.mem.write_u32(addr & ~3, data)
        elif op == 0x2E: # SWR - Store Word Right
            shift = (3 - (addr & 3)) * 8
            mask = 0xFFFFFFFF << shift
            mem_val = self.mem.read_u32(addr & ~3)
            data = (mem_val & mask) | (u32(rt_val << shift) & ~mask)
            self.mem.write_u32(addr & ~3, data)
        elif op == 0x2F: pass # CACHE (NOP)

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
        self.root.title("MIPSEMU2.0 - N64 Emulator Skeleton")
        self.root.configure(bg="#2E2E2E")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        self.memory = Memory(self.log)
        self.cpu = MIPSCPU(self.memory, self.log)
        self.rom_header = None
        self.running = False

        self.setup_ui()
        self.log("MIPSEMU2.0 Initialized. Ready to load a ROM.")
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
        self.reg_tree = ttk.Treeview(parent, columns=('val'), show='headings', height=32)
        self.reg_tree.heading('val', text='Value')
        self.reg_tree.column('val', width=100, anchor='w')
        self.reg_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i in range(32):
            self.reg_tree.insert('', 'end', iid=i, text=f"r{i} ({self.GPR_NAMES[i]})", 
                                 values=(f"0x{0:08X}",))
        
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
                for _ in range(1000):
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
            self.reg_tree.item(i, values=(f"0x{self.cpu.gpr[i]:08X}",), 
                               tags=('evenrow' if i % 2 == 0 else 'oddrow'))

        # Render "screen"
        self.canvas.delete("all")
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.create_rectangle(2, 2, w, h, outline="#00A0A0", width=2)

        if self.rom_header:
            title = self.rom_header.name
            self.canvas.create_text(w/2, h*0.2, text=title, font=("Arial", 24, "bold"), fill="#FFFFFF")
            
            info_text = (
                f"Game ID: {self.rom_header.game_id}\n"
                f"Region: {self.rom_header.region}\n\n"
                f"Status: {'Running' if self.running else 'Stopped'}"
            )
            self.canvas.create_text(w/2, h*0.4, text=info_text, font=("Consolas", 12), fill="#A0A0A0", justify=tk.CENTER)
            
            self.canvas.create_text(w/2, h-20, text="MIPSEMU2.0 - Display Stub", font=("Arial", 10), fill="#555555")

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
