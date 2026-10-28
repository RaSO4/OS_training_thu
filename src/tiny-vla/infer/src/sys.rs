//! RISC-V time/affinity helpers.  On the host these are no-ops.

#[cfg(target_arch = "riscv64")]
#[inline]
pub fn rdtime() -> u64 {
    let t: u64;
    unsafe {
        core::arch::asm!("rdtime {}", out(reg) t, options(nomem, nostack));
    }
    t
}

#[cfg(not(target_arch = "riscv64"))]
#[inline]
pub fn rdtime() -> u64 {
    0
}

#[cfg(target_arch = "riscv64")]
pub fn set_affinity(cpu: usize) -> i64 {
    if cpu >= 128 {
        return -1;
    }
    let mut mask = [0u64; 2];
    mask[cpu / 64] |= 1u64 << (cpu % 64);
    let ret: i64;
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a7") 122i64,           // sched_setaffinity
            in("a0") 0i64,             // current thread
            in("a1") 16u64,            // mask length in bytes
            in("a2") mask.as_ptr(),
            lateout("a0") ret,
            options(nostack)
        );
    }
    ret
}

#[cfg(not(target_arch = "riscv64"))]
pub fn set_affinity(_cpu: usize) -> i64 {
    0
}

#[cfg(target_arch = "riscv64")]
pub fn get_cpu() -> i64 {
    let ret: i64;
    unsafe {
        core::arch::asm!(
            "ecall",
            in("a7") 168i64,           // sched_getcpu
            lateout("a0") ret,
            options(nostack)
        );
    }
    ret
}

#[cfg(not(target_arch = "riscv64"))]
pub fn get_cpu() -> i64 {
    -1
}
