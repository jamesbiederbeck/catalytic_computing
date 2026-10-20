; cmtz-core runtime library: n x n matrices over F_p in data memory,
; row-major.
;
; Calling convention (all routines):
;   arguments  X4 = A, X5 = B, X6 = C (destination), X7 = n
;   clobbers   X8..X13, R29..R31
;   preserves  everything else, including X4..X7
; The compiler never allocates X4..X15 or R24..R31 for its own values.

; C = A * B. C must not overlap A or B.
__matmul:
        LIX   X8, 0                 ; i
__mm_i:
        LIX   X9, 0                 ; j
__mm_j:
        MOV   R31, R0               ; acc = 0
        LIX   X10, 0                ; k
        MULX  X11, X8, X7
        ADDX  X11, X11, X4          ; pa = A + i*n
        ADDX  X12, X5, X9           ; pb = B + j
__mm_k:
        LD    R30, [X11]
        LD    R29, [X12]
        MAC   R31, R30, R29
        ADDIX X11, X11, 1           ; pa += 1
        ADDX  X12, X12, X7          ; pb += n
        ADDIX X10, X10, 1
        BLTX  X10, X7, __mm_k
        MULX  X13, X8, X7
        ADDX  X13, X13, X9
        ADDX  X13, X13, X6
        ST    R31, [X13]            ; C[i][j] = acc
        ADDIX X9, X9, 1
        BLTX  X9, X7, __mm_j
        ADDIX X8, X8, 1
        BLTX  X8, X7, __mm_i
        RET

; C = A (n*n words).
__matcopy:
        MULX  X8, X7, X7
        LIX   X9, 0
__mc_loop:
        BEQX  X9, X8, __mc_done
        ADDX  X10, X4, X9
        LD    R31, [X10]
        ADDX  X10, X6, X9
        ST    R31, [X10]
        ADDIX X9, X9, 1
        JMP   __mc_loop
__mc_done:
        RET

; C = identity.
__matident:
        MULX  X8, X7, X7
        LIX   X9, 0
__mi_zero:
        BEQX  X9, X8, __mi_diag
        ADDX  X10, X6, X9
        ST    R0, [X10]
        ADDIX X9, X9, 1
        JMP   __mi_zero
__mi_diag:
        LDI   R31, 1
        ADDIX X11, X7, 1            ; stride n + 1 walks the diagonal
        LIX   X9, 0
__mi_loop:
        BEQX  X9, X8, __mi_done
        ADDX  X10, X6, X9
        ST    R31, [X10]
        ADDX  X9, X9, X11
        BLTX  X9, X8, __mi_loop
__mi_done:
        RET
