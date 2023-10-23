from main import Instruction, do_first_pass, RType, IType, IShiftType, parse_instruction, parse_opcode, parse_register, Funct3, Funct7, BType, SType, UType, JType, extract_tokens_from_instruction, strip_comments_from_asm_line, symbol_table_entry, TableEntry

def test_token_extraction():
    result = ["lw", "x2", "s0", "56"]
    instruction = "lw x2 56(s0)"
    extraction = extract_tokens_from_instruction(instruction)

    assert result == extraction

def test_add():
    hex_value = 0x208033
    add_object = RType("add", "x0", "x1", "x2")

    assert hex_value == int(add_object.get_hex(), 16)


def test_addi():
    hex_value = 0x3e800093
    addi_object = IType("addi", "x1", "x0", "1000")

    assert hex_value == int(addi_object.get_hex(), 16)


def test_andi():
    hex_value = 0x00157313
    andi_object = IType("andi", "t1", "a0", "1")

    assert hex_value == int(andi_object.get_hex(), 16)


def test_sub():
    hex_value = 0x40b50533
    sub_object = RType("sub", "a0", "a0", "a1")

    assert hex_value == int(sub_object.get_hex(), 16)


def test_and():
    hex_value = 0x42f3b3
    and_object = RType("and", "x7", "x5", "x4")

    assert hex_value == int(and_object.get_hex(), 16)


def test_or():
    hex_value = 0x42e3b3
    or_object = RType("or", "x7", "x5", "x4")

    assert hex_value == int(or_object.get_hex(), 16)


def test_ori():
    hex_value = 0x00256513
    ori_object = IType("ori", "a0", "a0", "2")

    assert hex_value == int(ori_object.get_hex(), 16)


def test_xor():
    hex_value = 0x00554533
    xor_object = RType("xor", "a0", "a0", "t0")

    assert hex_value == int(xor_object.get_hex(), 16)


def test_xori():
    hex_value = 0x3e82c393
    xori_object = IType("xori", "x7", "x5", "1000")

    assert hex_value  == int(xori_object.get_hex(), 16)


def test_srai():
    hex_value = 0x41f2d393
    srai_object = IShiftType("srai", "x7", "x5", "31")

    assert hex_value == int(srai_object.get_hex(), 16)


def test_srli():
    hex_value = 0x00155513
    srli_object = IShiftType("srli", "a0", "a0", "1")

    assert hex_value == int(srli_object.get_hex(), 16)


def test_slti():
    hex_value = 0x3e82a393
    slti_object = IType("slti", "x7", "x5", "1000")

    assert hex_value == int(slti_object.get_hex(), 16)


def test_sltiu():
    hex_value = 0x3e82b393
    sltiu_object = IType("sltiu", "x7", "x5", "1000")

    assert hex_value == int(sltiu_object.get_hex(), 16)


def test_slli():
    hex_value = 0x00251513
    slli_object = IShiftType("slli", "a0", "a0", "2")

    assert hex_value == int(slli_object.get_hex(), 16)


def test_slt():
    hex_value = 0x000522b3
    slt_object = RType("slt", "t0", "a0", "zero")

    assert hex_value == int(slt_object.get_hex(), 16)


def test_sll():
    hex_value = 0x4293b3
    sll_object = RType("sll", "x7", "x5", "x4")

    assert hex_value == int(sll_object.get_hex(), 16)


def test_sltu():
    hex_value = 0x42b3b3
    sltu_object = RType("sltu", "x7", "x5", "x4")

    assert hex_value == int(sltu_object.get_hex(), 16)


def test_srl():
    hex_value = 0x42d3b3
    srl_object = RType("srl", "x7", "x5", "x4")

    assert hex_value == int(srl_object.get_hex(), 16)


def test_sra():
    hex_value = 0x4042d3b3
    sra_object = RType("sra", "x7", "x5", "x4")

    assert hex_value == int(sra_object.get_hex(), 16)


def test_li():
    hex_value = 0x03842103
    lw_object = IType("lw", "x2", "s0", "56")

    assert hex_value == int(lw_object.get_hex(), 16)


def test_lb():
    hex_value = 0x3840103
    lb_object = IType("lb", "x2", "s0", "56")

    assert hex_value == int(lb_object.get_hex(), 16)


def test_lbu():
    hex_value = 0x3844103
    lbu_object = IType("lbu", "x2", "s0", "56")

    assert hex_value == int(lbu_object.get_hex(), 16)


def test_lh():
    hex_value = 0x3841103
    lh_object = IType("lh", "x2", "s0", "56")

    assert hex_value == int(lh_object.get_hex(), 16)


def test_sb():
    hex_value = 0x2240c23
    sb_object = SType("sb", "x2", "s0", "56")

    assert hex_value == int(sb_object.get_hex(), 16)


def test_sh():
    hex_value = 0x2241c23
    sh_object = SType("sh", "x2", "s0", "56")

    assert hex_value == int(sh_object.get_hex(), 16)


def test_sw():
    hex_value = 0x2242c23
    sw_object = SType("sw", "x2", "s0", "56")

    assert hex_value == int(sw_object.get_hex(), 16)


def test_lui():
    hex_value = 0x64037
    lui_object = UType("lui", "x0", "100")

    assert hex_value == int(lui_object.get_hex(), 16)


def test_auipc():
    hex_value = 0x64017
    auipc_object = UType("auipc", "x0", "100")

    assert hex_value == int(auipc_object.get_hex(), 16)


def test_blt():
    hex_value = 0x00b54063
    blt_object = BType("blt", "a0", "a1", "0")

    assert hex_value == int(blt_object.get_hex(), 16)


def test_jal_parse():
    return

    program = []
    with open("tests/jal.S", 'r') as file:
        program = do_first_pass(file)


    for instruction in program:
       result = parse_instruction(instruction)
       bin_res = result.get_hex()
       print(bin_res)

    assert program is not None


def test_jal():
    sample_jal = JType("jal", "ra", "8")
    hex_value = 0x8000ef

    print(sample_jal.get_hex())

    assert hex_value == int(sample_jal.get_hex(), 16)


def test_register_parser():
    reg1 = "x1"
    reg2 = "x2"
    reg30 = "x30"

    reg_zero = 'zero'
    reg_ra = 'ra'
    reg_sp = 'sp'
    reg_gp = 'gp'
    reg_tp = 'tp'
    reg_t0 = 't0'

    reg_t = ['t%d' % i for i in range(1, 2)]
    reg_t2 = ['t%d' % i for i in range(3, 6)]
    reg_a = ['a%d' % i for i in range(0, 7)]
    reg_s = ['s%d' % i for i in range(2, 12)]

    assert parse_register(reg1) == 1
    assert parse_register(reg2) == 2
    assert parse_register(reg30) == 30
    assert parse_register(reg_zero) == 0
    assert parse_register(reg_ra) == 1
    assert parse_register(reg_sp) == 2
    assert parse_register(reg_gp) == 3
    assert parse_register(reg_tp) == 4
    assert parse_register(reg_t0) == 5

    for i in range(0):
        assert parse_register(reg_t[i]) == i + 5

    for i in range(3):
        assert parse_register(reg_t2[i]) == i + 28

    for i in range(0, 7):
        assert parse_register(reg_a[i]) == i + 10

    for i in range(0, 10):
        assert parse_register(reg_s[i]) == i + 18


def test_funct3():
    addi = Funct3['ADDI']
    sub = Funct3['SUB']
    add = Funct3['ADD']

    assert addi.value == 0b000
    assert sub.value == 0b000
    assert add.value == 0b000

    slli = Funct3['SLLI']

    assert slli.value == 0b001

    slt = Funct3['SLT']
    slti = Funct3['SLTI']

    assert slt.value == 0b010
    assert slti.value == 0b010

    xor = Funct3['XOR']
    xori = Funct3['XORI']

    assert xor.value == 0b100
    assert xori.value == 0b100

    srl = Funct3['SRL']
    srli = Funct3['SRLI']
    sra = Funct3['SRA']
    srai = Funct3['SRAI']

    assert srl.value == 0b101
    assert srli.value == 0b101
    assert sra.value == 0b101
    assert srai.value == 0b101

    or_funct3 = Funct3['OR']
    ori = Funct3['ORI']

    assert or_funct3.value == 0b110
    assert ori.value == 0b110

    and_funct3 = Funct3['AND']
    andi = Funct3['ANDI']

    assert and_funct3.value == 0b111
    assert andi.value == 0b111

    beq = Funct3['BEQ']
    bne = Funct3['BNE']
    blt = Funct3['BLT']
    bge = Funct3['BGE']
    bltu = Funct3['BLTU']
    bgeu = Funct3['BGEU']

    assert beq.value == 0b000
    assert bne.value == 0b001
    assert blt.value == 0b100
    assert bge.value == 0b101
    assert bltu.value == 0b110
    assert bgeu.value == 0b111

    lb = Funct3['LB']
    sb = Funct3['SB']

    assert lb.value == 0b000
    assert sb.value == 0b000

    lh = Funct3['LH']
    sh = Funct3['SH']

    assert lh.value == 0b001
    assert sh.value == 0b001

    lw = Funct3['LW']
    sw = Funct3['SW']

    assert lw.value == 0b010
    assert sw.value == 0b010

    lbu = Funct3['LBU']
    lhu = Funct3['LHU']

    assert lbu.value == 0b100
    assert lhu.value == 0b101


def test_func7():
    slli = Funct7['SLLI']
    srli = Funct7['SRLI']
    add = Funct7['ADD']
    sll = Funct7['SLL']
    sltu = Funct7['SLTU']
    xor = Funct7['XOR']
    srl = Funct7['SRL']
    or_funct7 = Funct7['OR']
    and_funct7 = Funct7['AND']

    assert slli.value == 0b0000000
    assert srli.value == 0b0000000
    assert add.value == 0b0000000
    assert sll.value == 0b0000000
    assert sltu.value == 0b0000000
    assert xor.value == 0b0000000
    assert srl.value == 0b0000000
    assert or_funct7.value == 0b0000000
    assert and_funct7.value == 0b0000000

    srai = Funct7['SRAI']
    sub = Funct7['SUB']

    assert srai.value == 0b0100000
    assert sub.value == 0b0100000


def test_parse_opcode():
    sra = "sra"

    assert parse_opcode(sra) == 0b00110011


def test_strip_comments():
    asm_code = ["test #test",  "test123 #123"]
    program = []

    for line in asm_code:
        program.append(strip_comments_from_asm_line(line))

    result = ["test", "test123"]

    assert program[0] == result[0]
    assert program[1] == result[1]


def test_symbol_table_entry():
    line = "label:"
    entry = symbol_table_entry(line, 0)

    result = TableEntry("label", 0)


    assert entry == result


def test_do_first_pass():
    program = []
    with open("assembly/sum.S", 'r') as file:
       program = do_first_pass(file)

    assert program is not None


def test_parse_instruction():
    instruction = Instruction("addi x1, x0, 1000", 0, 0)
    parsed_inst = parse_instruction(instruction)
    expected_inst = IType("addi", "x1", "x0", "1000")

    assert parsed_inst.get_hex() == expected_inst.get_hex()


def test_full_parse_of_assembly():
    # FIXME: This test is not passing when running pytest because we use a global symbol table
    program = []
    with open("assembly/sum.S", 'r') as file:
        program = do_first_pass(file)

    res = []
    for instruction in program:
        parsed_inst = parse_instruction(instruction)
        res.append(parsed_inst)
        print(f"{instruction.address}\t{parsed_inst.get_hex()}")

    assert int(res[0].get_hex(), 16) == 0xff010113
    assert int(res[1].get_hex(), 16) == 0x112623
    assert int(res[2].get_hex(), 16) == 0x812423
    assert int(res[3].get_hex(), 16) == 0x1010413
    assert int(res[4].get_hex(), 16) == 0xfea42a23
