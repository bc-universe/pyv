import argparse
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

parser = argparse.ArgumentParser(
    prog='SampleRiscVDecoder',
    description='Toy program that decodes *some* rv32i instructions',
    epilog='Much incomplete'
)

parser.add_argument('path', metavar='s', type=str, nargs=1, help='Path to the source asm file')

# I-Type instruction segregated by load and store
load_instructions = ['lw', 'lb', 'lh', 'lbu', 'sb', 'sh', 'sw']
r_type = ['add', 'sll', 'slt',
          'sltu', 'xor', 'srl', 'or', 'and', 'sub', 'sra']
i_type = ['jalr',
          'addi', 'slti', 'sltiu', 'xori', 'ori', 'andi']
# I-Type instructions with shamt
is_type = ['slli', 'srli', 'srai']
s_type = ['sb', 'sh', 'sw']
b_type = ['bne', 'beq', 'blt', 'bge', 'bltu', 'bgeu']
u_type = ['lui', 'auipc']
j_type = ['jal']


@dataclass
class Instruction:
    raw_string: str
    line_number: int
    address: int
    encoding: Optional[int] = None
    error: Optional[str] = None


@dataclass
class TableEntry:
    label: str
    address: int


def imm_to_bin(value: int, truncate_ammount: int) -> str:
    if value >= 0:
        return(bin(value))[2:].zfill(truncate_ammount)
    bin_imm = (value & 0b1111111111111111)
    return str(bin(bin_imm))[-truncate_ammount:]


class Funct3(Enum):
    ADD = SUB = ADDI = 0b000
    SLLI = SLL = 0b001
    SLT = SLTI = 0b010
    SLTU = SLTIU = 0b011

    XOR = XORI = 0b100
    SRL = SRLI = SRA = SRAI = 0b101
    OR = ORI = 0b110
    AND = ANDI = 0b111

    BEQ = 0b000
    BNE = 0b001
    BLT = 0b100
    BGE = 0b101
    BLTU = 0b110
    BGEU = 0b111

    LB = SB = 0b000
    LH = SH = 0b001
    LW = SW = 0b010
    LBU = 0b100
    LHU = 0b101

    JALR = 0b000


class Funct7(Enum):
    SLLI = SRLI = ADD = SLL = SLT = SLTU = XOR = SRL = OR = AND = 0b0000000
    SRAI = SUB = SRA = 0b0100000


def parse_register(register: str) -> int:
    if register is None or not register:
        raise ValueError("Register cannot be None or Empty")

    match register :
        case 'zero':
            return 0
        case 'ra':
            return 1
        case 'sp':
            return 2
        case 'gp':
            return 3
        case 'tp':
            return 4
        case 'fp':
            return 8


    match register[0]:
        case 'x' if register[1].isdecimal:
            return int(register[1:])
        case 'a' if register[1:].isdecimal:
            return int(register[1:]) + 10
        case 's' if register[1:].isdecimal:
            return int(register[1]) + 8 if int(register[1:]) < 2 else int(register[1:]) + 16
        case 't' if register[1:].isdecimal:
            return int(register[1]) + 5 if int(register[1]) < 2 else int(register[1]) + 25
        case 'f':
            raise ValueError("No support for FP Registers")


class RType:
    """R-Type operation"""

    def __init__(self, op: str, rd: str, rs1: str, rs2: str):
        self.op = parse_opcode(op)
        self.rd = parse_register(rd)
        self.funct3 = Funct3[op.upper()]
        self.rs1 = parse_register(rs1)
        self.rs2 = parse_register(rs2)
        self.funct7 = Funct7[op.upper()]

    def get_hex(self) -> int:
        funct7 = self.funct7.value << 25
        rs2 = self.rs2 << 20
        rs1 = self.rs1 << 15
        funct3 = self.funct3.value << 12
        rd = self.rd << 7

        return hex(funct7 | rs2 | rs1 | funct3 | rd | self.op)


class IShiftType:
    """I-Type shift operation"""

    def __init__(self, op: str, rd: str, rs1: str, shamt: str):
        self.op = parse_opcode(op)
        self.rd = parse_register(rd)
        self.funct3 = Funct3[op.upper()]
        self.rs1 = parse_register(rs1)
        self.shamt = int(shamt)
        self.funct7 = Funct7[op.upper()]

    def get_hex(self) -> int:
        funct7 = self.funct7.value << 25
        shamt = self.shamt << 20
        rs1 = self.rs1 << 15
        funct3 = self.funct3.value << 12
        rd = self.rd << 7

        return hex(funct7 | shamt | rs1 | funct3 | rd | self.op)


class IType:
    """I-Type operation"""

    def __init__(self, op: str, rd: str, rs1: str, imm: str):
        self.imm = int(imm)
        self.op = parse_opcode(op)
        self.rd = parse_register(rd)
        self.rs1 = parse_register(rs1)
        self.funct3 = Funct3[op.upper()]

    def get_hex(self) -> int:
        imm = self.imm & 0b111111111111
        imm = imm << 20
        rs1 = self.rs1 << 15
        funct3 = self.funct3.value << 12
        rd = self.rd << 7

        return hex(imm | rs1 | funct3 | rd | self.op)


class SType:
    """S-Type operation"""

    def __init__(self, op: str, rs2: str, rs1: str, imm: str):
        self.imm = int(imm)
        self.op = parse_opcode(op)
        self.rs1 = parse_register(rs1)
        self.rs2 = parse_register(rs2)
        self.funct3 = Funct3[op.upper()]

    def get_hex(self) -> int:
        imm = self.imm & 0b111111100000
        imm >>= 5
        imm <<=  25
        second_imm = self.imm &0b11111
        second_imm <<= 7
        rs2 = self.rs2 << 20
        funct3 = self.funct3.value << 12
        rs1 = self.rs1 << 15

        return hex(imm | rs2 | rs1 | funct3 | second_imm | self.op)


class UType:
    """U-Type Operation"""

    def __init__(self, op: str, rd: str, imm: str):
        self.op = parse_opcode(op)
        self.imm = int(imm)
        self.rd = parse_register(rd)

    def get_hex(self) -> int:
        imm = self.imm & 0b11111111111111111111
        imm <<=  12
        rd = self.rd << 7

        return hex(imm | rd | self.op)


class BType:
    """B-Type Operation"""

    def __init__(self, op: str, rs1: str, rs2: str, imm: str):
       self.op = parse_opcode(op)
       self.imm = int(imm)
       self.rs1 = parse_register(rs1)
       self.rs2 = parse_register(rs2)
       self.funct3 = Funct3[op.upper()]

    def get_hex(self) -> int:
        # imm[12|10:5]
        # imm[4:1|11]
        first_imm = self.imm & 0b100000000000
        first_imm >>= 1
        range_first_imm = self.imm & 0b11111100000
        first_imm |= range_first_imm
        first_imm <<= 25
        rs2 = self.rs2 << 20
        rs1 = self.rs1 << 15
        funct3 = self.funct3.value << 12
        second_imm = (self.imm << 12) & 0b1111
        second_imm = (self.imm << 11) & 0b1111100000
        second_imm <<= 7

        return (hex(first_imm | rs2 | rs1 | funct3 | second_imm | self.op))


class JType:
    """J-Type Operation
        There are diverse ways to call an J-Type instruction it can be like
        jal {label}
    """

    def __init__(self, op: str, rd: str, imm: str):
        self.op = parse_opcode(op)
        self.rd = parse_register(rd)
        self.imm = int(imm)

    def get_hex(self) -> int:
        # imm[20|10:1|11|19:12]
        imm_20 = self.imm & 0b100000000000000000000
        imm_10_1 = self.imm & 0b11111111110
        imm_10_1 >>= 1
        imm_11 = self.imm & 0b100000000000
        imm_19_12 = self.imm & 0b11111111000000000000
        imm = imm_20 | (imm_10_1 << 9 ) | (imm_11 << 8) | (imm_19_12)
        imm <<= 12
        rd = self.rd << 7

        return hex(imm | rd | self.op)


class ECall:
    """Representation of an ECall operation"""

    def get_hex(self) -> int:
        return 0x73


class EBreak:
    """Representaion of an EBreak operation"""

    def get_hex(Self) -> int:
        return 0x100073


def parse_opcode(opcode: str) -> int:
    if opcode in ['beq', 'bne', 'blt', 'bge', 'bgltu', 'bgeu']:
        return 0b1100011
    elif opcode in ['lb', 'lh', 'lw', 'lbu', 'lhu']:
        return 0b0000011
    elif opcode in ['sb', 'sh', 'sw']:
        return 0b0100011
    elif opcode in ['addi', 'slti', 'sltiu', 'xori', 'ori', 'andi', 'slli', 'srli', 'srai']:
        return 0b0010011
    elif opcode in ['add', 'sub', 'sll', 'slt', 'sltu', 'xor', 'srl', 'sra', 'or', 'and']:
        return 0b0110011
    elif opcode == 'lui':
        return 0b0110111
    elif opcode == 'auipc':
        return 0b0010111
    elif opcode == 'jal':
        return 0b1101111
    elif opcode == 'jalr':
        return 0b1100111


def do_first_pass(file_obj) -> Optional[list[Instruction]]:
    """ For the first pass we need only do some checks: remove comments, populate symbol table ...
        After that we return an Instruction list which contains the raw assembly line, program_line and the address
        which is then going to be parsed in sequence.
    """

    program = []
    program_line = 0
    address = 0

    for line in file_obj:
        """ Check for comments and strip them. After we should populate the symbol table"""
        if line is None or line == '\n':
            continue

        current_line = line
        if '#' in line:
            current_line = strip_comments_from_asm_line(line)

        if ':' in line:
            program_line += 1
            entry = symbol_table_entry(current_line, address)

            symbol_table[entry.label] = entry.address
            continue

        instruction = Instruction(current_line.strip(), program_line, hex(address))
        program_line += 1
        address += 4
        program.append(instruction)

    return program


def strip_comments_from_asm_line(line: str) -> Optional[str]:
    """ Here we make sure to remove the comments by splitting for the pound sign. We return a string containing the instruction
    """
    if not isinstance(line, str) :
        return None

    stripped_line = line.split('#')[0]
    return stripped_line.strip()


def symbol_table_entry(line: str, address: int) -> TableEntry:
    label = line.split(':')[0]
    return TableEntry(label, address)


def extract_register_from_parens(line: str) -> str:
    register = [p.split(')')[0] for p in line.split('(') if ')' in p]
    return register

def extract_tokens_from_instruction(line: str) -> list:
    tokens = line.replace(',', '').split()
    if '(' not in tokens[-1]:
        return tokens

    val = tokens[-1].split('(')[0]
    register = extract_register_from_parens(tokens[-1])

    res = tokens[:-1] + register + [val]

    return res


def parse_instruction(instruction: Instruction):
    """ This is the function that parses an instruction and generates one of our Classes that represent the base instruction types.
    """
    tokenized_line = extract_tokens_from_instruction(instruction.raw_string)
    opcode = tokenized_line[0]

    if opcode == 'ecal':
        return ECall()
    if opcode == 'ebreak':
        return EBreak()
    if opcode in r_type:
        if len(tokenized_line) != 4:
            raise ValueError("Expected RType but instruction did not meet criteria")
        return RType(tokenized_line[0], tokenized_line[1], tokenized_line[2], tokenized_line[3])
    if opcode in i_type:
        if len(tokenized_line) != 4:
            raise ValueError("Expected IType but instruction did not meet criteria")
        return IType(tokenized_line[0], tokenized_line[1], tokenized_line[2], tokenized_line[3])
    if opcode in s_type:
        if len(tokenized_line) != 4:
            raise ValueError("Expected 4 elements for SType Instruction")
        return SType(tokenized_line[0], tokenized_line[1], tokenized_line[2], tokenized_line[3])
    if opcode in u_type:
        if len(tokenized_line) != 3:
            raise ValueError("Expected 3 elements for UType Instruction")
        return UType(tokenized_line[0], tokenized_line[1], tokenized_line[2])
    if opcode in is_type:
        if len(tokenized_line) != 4:
            raise ValueError("Expected 4 elements for Shift Instruction")
        return IShiftType(tokenized_line[0], tokenized_line[1], tokenized_line[2], tokenized_line[3])
    if opcode in j_type:
        if tokenized_line[1] not in symbol_table:
            raise "Label is not present on symbol table"
        symbol_table[tokenized_line[1]]
        return JType(tokenized_line[0], symbol_table[tokenized_line[1]])
    if opcode in load_instructions:
        if len(tokenized_line) != 4:
            print(tokenized_line)
            raise ValueError("Expected 4 elements for IType load")
        return IType(tokenized_line[0], tokenized_line[1], tokenized_line[2], tokenized_line[3])

    print(f"Possibly not implemented instruction: {tokenized_line}")
    return tokenized_line

symbol_table = {}

if __name__ == '__main__':
    args = parser.parse_args()
    path = Path(args.path[0])

    if not path.exists():
        raise RuntimeError("Path is incorrect")

    if path.suffixes[0].upper() != '.S':
        raise RuntimeError("Filetype not supported")

    location_counter = 0
    program_instructions = []

    with open(path) as line:
        program_instructions = do_first_pass(line)

    for instruction in program_instructions:
        result = parse_instruction(instruction)
        address = instruction.address
        hex_val = result.get_hex()
        raw_instruction = instruction.raw_string
        print(f"{address}:\t{hex_val}\t{raw_instruction}")
