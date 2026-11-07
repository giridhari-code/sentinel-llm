//! CUDA Driver API launch path for Tesla-2 (dynamic load, no cuBLAS).
//!
//! Loads `libcuda.so` at runtime. If CUDA is missing, returns a clear error
//! and the caller falls back to HostMirror.

use sentinel_core::{Result, SentinelError};
use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;
use std::sync::OnceLock;

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;

const CUDA_SUCCESS: CUresult = 0;
const CU_CTX_SCHED_AUTO: c_uint = 0;

#[derive(Debug)]
struct DriverFns {
    cu_init: extern "C" fn(c_uint) -> CUresult,
    cu_device_get: extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    cu_ctx_create: extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    cu_ctx_set_current: extern "C" fn(CUcontext) -> CUresult,
    cu_module_load_data: extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_get_function: extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cu_mem_alloc: extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cu_memcpy_dtoh: extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_launch_kernel: extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    cu_ctx_synchronize: extern "C" fn() -> CUresult,
    cu_get_error_string: extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

static DRIVER: OnceLock<Option<DriverFns>> = OnceLock::new();

fn load_driver() -> Option<&'static DriverFns> {
    DRIVER
        .get_or_init(|| unsafe { try_load_driver() })
        .as_ref()
}

unsafe fn try_load_driver() -> Option<DriverFns> {
    // Prefer libcuda.so.1 (driver), fall back to libcuda.so
    let lib = load_lib(&["libcuda.so.1", "libcuda.so"])?;
    macro_rules! sym {
        ($name:expr) => {{
            let p = lib_sym(&lib, $name)?;
            std::mem::transmute(p)
        }};
    }
    macro_rules! sym_alt {
        ($a:expr, $b:expr) => {{
            if let Some(p) = lib_sym(&lib, $a) {
                std::mem::transmute(p)
            } else {
                let p = lib_sym(&lib, $b)?;
                std::mem::transmute(p)
            }
        }};
    }
    Some(DriverFns {
        cu_init: sym!("cuInit"),
        cu_device_get: sym!("cuDeviceGet"),
        cu_ctx_create: sym_alt!("cuCtxCreate_v2", "cuCtxCreate"),
        cu_ctx_set_current: sym!("cuCtxSetCurrent"),
        cu_module_load_data: sym!("cuModuleLoadData"),
        cu_module_get_function: sym!("cuModuleGetFunction"),
        cu_mem_alloc: sym_alt!("cuMemAlloc_v2", "cuMemAlloc"),
        cu_mem_free: sym_alt!("cuMemFree_v2", "cuMemFree"),
        cu_memcpy_htod: sym_alt!("cuMemcpyHtoD_v2", "cuMemcpyHtoD"),
        cu_memcpy_dtoh: sym_alt!("cuMemcpyDtoH_v2", "cuMemcpyDtoH"),
        cu_launch_kernel: sym!("cuLaunchKernel"),
        cu_ctx_synchronize: sym!("cuCtxSynchronize"),
        cu_get_error_string: sym!("cuGetErrorString"),
    })
}

struct DynLib {
    handle: *mut c_void,
}

unsafe impl Send for DynLib {}
unsafe impl Sync for DynLib {}

unsafe fn load_lib(names: &[&str]) -> Option<DynLib> {
    for name in names {
        let c = CString::new(*name).ok()?;
        let h = libc_dlopen(c.as_ptr());
        if !h.is_null() {
            return Some(DynLib { handle: h });
        }
    }
    None
}

unsafe fn lib_sym(lib: &DynLib, name: &str) -> Option<*mut c_void> {
    let c = CString::new(name).ok()?;
    let p = libc_dlsym(lib.handle, c.as_ptr());
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

// Minimal libc dlopen bindings (avoid extra crates)
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

unsafe fn libc_dlopen(name: *const c_char) -> *mut c_void {
    dlopen(name, RTLD_NOW)
}
unsafe fn libc_dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    dlsym(handle, name)
}

fn cu_err(fns: &DriverFns, code: CUresult, what: &str) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    let mut msg: *const c_char = ptr::null();
    unsafe {
        (fns.cu_get_error_string)(code, &mut msg);
    }
    let detail = if msg.is_null() {
        format!("code {}", code)
    } else {
        unsafe { std::ffi::CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    Err(SentinelError::Kernel(format!("CUDA {}: {}", what, detail)))
}

/// Live CUDA context + loaded Tesla gemm kernel.
pub struct CudaLaunch {
    #[allow(dead_code)]
    context: CUcontext,
    #[allow(dead_code)]
    module: CUmodule,
    gemm: CUfunction,
    device_id: u32,
}

// SAFETY: CUDA context used from one thread in this scaffold; Send for Gpu storage.
unsafe impl Send for CudaLaunch {}
unsafe impl Sync for CudaLaunch {}

impl CudaLaunch {
    pub fn available() -> bool {
        load_driver().is_some()
    }

    pub fn new(device_id: u32) -> Result<Self> {
        let fns = load_driver().ok_or_else(|| {
            SentinelError::Kernel(
                "CUDA driver not found (libcuda.so). Install NVIDIA driver / CUDA.".into(),
            )
        })?;

        unsafe {
            cu_err(fns, (fns.cu_init)(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            cu_err(
                fns,
                (fns.cu_device_get)(&mut dev, device_id as c_int),
                "cuDeviceGet",
            )?;
            let mut ctx: CUcontext = ptr::null_mut();
            cu_err(
                fns,
                (fns.cu_ctx_create)(&mut ctx, CU_CTX_SCHED_AUTO, dev),
                "cuCtxCreate",
            )?;
            cu_err(fns, (fns.cu_ctx_set_current)(ctx), "cuCtxSetCurrent")?;

            let ptx = include_str!("ptx/tesla_gemm_f32.ptx");
            // cuModuleLoadData expects a null-terminated image
            let mut ptx_bytes = ptx.as_bytes().to_vec();
            ptx_bytes.push(0);
            let mut module: CUmodule = ptr::null_mut();
            cu_err(
                fns,
                (fns.cu_module_load_data)(&mut module, ptx_bytes.as_ptr() as *const c_void),
                "cuModuleLoadData(tesla_gemm_f32.ptx)",
            )?;

            let fname = CString::new("tesla_gemm_f32").unwrap();
            let mut func: CUfunction = ptr::null_mut();
            cu_err(
                fns,
                (fns.cu_module_get_function)(&mut func, module, fname.as_ptr()),
                "cuModuleGetFunction(tesla_gemm_f32)",
            )?;

            log::info!(
                "Tesla-2 CUDA launch ready on device {} (tesla_gemm_f32 PTX loaded)",
                device_id
            );
            Ok(Self {
                context: ctx,
                module,
                gemm: func,
                device_id,
            })
        }
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// C = A @ B, row-major f32. Shapes A[m,k], B[k,n].
    pub fn gemm_f32(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(SentinelError::Kernel("gemm_f32: shape/data mismatch".into()));
        }
        let fns = load_driver().ok_or_else(|| {
            SentinelError::Kernel("CUDA driver disappeared".into())
        })?;

        unsafe {
            cu_err(fns, (fns.cu_ctx_set_current)(self.context), "cuCtxSetCurrent")?;

            let bytes_a = m * k * 4;
            let bytes_b = k * n * 4;
            let bytes_c = m * n * 4;

            let mut d_a: CUdeviceptr = 0;
            let mut d_b: CUdeviceptr = 0;
            let mut d_c: CUdeviceptr = 0;
            cu_err(fns, (fns.cu_mem_alloc)(&mut d_a, bytes_a), "cuMemAlloc A")?;
            cu_err(fns, (fns.cu_mem_alloc)(&mut d_b, bytes_b), "cuMemAlloc B")?;
            cu_err(fns, (fns.cu_mem_alloc)(&mut d_c, bytes_c), "cuMemAlloc C")?;

            let result = (|| {
                cu_err(
                    fns,
                    (fns.cu_memcpy_htod)(d_a, a.as_ptr() as *const c_void, bytes_a),
                    "H2D A",
                )?;
                cu_err(
                    fns,
                    (fns.cu_memcpy_htod)(d_b, b.as_ptr() as *const c_void, bytes_b),
                    "H2D B",
                )?;

                let mut m_i = m as i32;
                let mut n_i = n as i32;
                let mut k_i = k as i32;
                let mut ptr_a = d_a;
                let mut ptr_b = d_b;
                let mut ptr_c = d_c;

                let mut args = [
                    &mut ptr_a as *mut _ as *mut c_void,
                    &mut ptr_b as *mut _ as *mut c_void,
                    &mut ptr_c as *mut _ as *mut c_void,
                    &mut m_i as *mut _ as *mut c_void,
                    &mut n_i as *mut _ as *mut c_void,
                    &mut k_i as *mut _ as *mut c_void,
                ];

                // 16x16 threads
                let block_x = 16u32;
                let block_y = 16u32;
                let grid_x = ((n as u32) + block_x - 1) / block_x;
                let grid_y = ((m as u32) + block_y - 1) / block_y;

                cu_err(
                    fns,
                    (fns.cu_launch_kernel)(
                        self.gemm,
                        grid_x,
                        grid_y,
                        1,
                        block_x,
                        block_y,
                        1,
                        0,
                        ptr::null_mut(),
                        args.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel(tesla_gemm_f32)",
                )?;
                cu_err(fns, (fns.cu_ctx_synchronize)(), "cuCtxSynchronize")?;

                let mut out = vec![0.0f32; m * n];
                cu_err(
                    fns,
                    (fns.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, d_c, bytes_c),
                    "D2H C",
                )?;
                Ok(out)
            })();

            let _ = (fns.cu_mem_free)(d_a);
            let _ = (fns.cu_mem_free)(d_b);
            let _ = (fns.cu_mem_free)(d_c);
            result
        }
    }
}

/// Try create real CUDA launch path; None if driver/PTX unavailable.
pub fn try_cuda_launch(device_id: u32) -> Option<CudaLaunch> {
    match CudaLaunch::new(device_id) {
        Ok(c) => Some(c),
        Err(e) => {
            log::warn!("Tesla-2 CUDA launch unavailable: {}", e);
            None
        }
    }
}
