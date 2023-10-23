        addi    sp, sp, -16
        sw      ra, 12(sp)
        sw      s0, 8(sp)
        addi    s0, sp, 16
        sw      a0, -12(s0)
        sw      a1, -16(s0)
        lw      a0, -12(s0)
        lw      a1, -16(s0)
        add     a0, a0, a1
        lw      ra, 12(sp)
        lw      s0, 8(sp)
        addi    sp, sp, 16
        jalr x0, 0(x1)
