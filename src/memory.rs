//! Cooperative ownership of resident accelerator models and host memory admission.
use anyhow::{Context, Result, bail};
use candle_core::Device;
use serde::Serialize;
use std::{
    collections::VecDeque,
    ffi::CStr,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::{
        fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
        io::AsRawFd,
    },
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const GIB: u64 = 1 << 30;
const MIN_SYSTEM_RESERVE: u64 = 16 * GIB;
const SYSTEM_RESERVE_PERCENT: u64 = 10;
const POLL: Duration = Duration::from_millis(50);

unsafe extern "C" {
    fn mach_port_deallocate(
        task: libc::mach_port_t,
        name: libc::mach_port_t,
    ) -> libc::kern_return_t;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pressure {
    Unknown,
    Normal,
    Warning,
    Critical,
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub timestamp_ms: u64,
    pub physical_bytes: Option<u64>,
    pub system_used_bytes: Option<u64>,
    pub process_footprint_bytes: Option<u64>,
    pub pressure: Pressure,
    pub metal_allocated_bytes: Option<u64>,
    pub metal_recommended_bytes: Option<u64>,
}

pub struct Coordinator {
    directory: PathBuf,
    owned: AtomicBool,
    managed: AtomicBool,
    reservation: AtomicU64,
    waiters: AtomicUsize,
    queue: Mutex<(u64, VecDeque<u64>)>,
    reacquire_after: Mutex<Instant>,
    latest: Mutex<Snapshot>,
    telemetry: Mutex<Option<File>>,
}

static GLOBAL: OnceLock<Arc<Coordinator>> = OnceLock::new();

fn coordinator() -> &'static Arc<Coordinator> {
    GLOBAL.get_or_init(|| {
        let uid = unsafe { libc::geteuid() };
        let value = Arc::new(Coordinator {
            directory: PathBuf::from(format!("/private/tmp/xwen-memory-{uid}")),
            owned: AtomicBool::new(false),
            managed: AtomicBool::new(false),
            reservation: AtomicU64::new(0),
            waiters: AtomicUsize::new(0),
            queue: Mutex::new((0, VecDeque::new())),
            reacquire_after: Mutex::new(Instant::now()),
            latest: Mutex::new(host_sample()),
            telemetry: Mutex::new(open_telemetry()),
        });
        if let Some(file) = value
            .telemetry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            write_telemetry(file, &startup_entry());
        }
        value.record(
            "initial",
            &value.latest.lock().unwrap_or_else(|e| e.into_inner()),
        );
        let monitor = Arc::downgrade(&value);
        thread::spawn(move || {
            let mut previous_pressure = Pressure::Unknown;
            let mut previous_stop_reason = None;
            let mut ticks = 0u32;
            while let Some(value) = monitor.upgrade() {
                let snapshot = host_sample();
                *value.latest.lock().unwrap_or_else(|e| e.into_inner()) = snapshot.clone();
                let stop_reason = runtime_stop_reason(&snapshot);
                if stop_reason != previous_stop_reason {
                    value.record(
                        stop_reason.unwrap_or("runtime: memory headroom recovered"),
                        &snapshot,
                    );
                }
                previous_stop_reason = stop_reason;
                if snapshot.pressure != previous_pressure
                    || (value.managed.load(Ordering::Acquire) && ticks.is_multiple_of(10))
                {
                    value.record("monitor", &snapshot);
                }
                previous_pressure = snapshot.pressure;
                ticks = ticks.wrapping_add(1);
                drop(value);
                thread::sleep(Duration::from_millis(500));
            }
        });
        value
    })
}

fn secure_directory(path: &Path) -> Result<()> {
    match fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(e) => return Err(e.into()),
    }
    let m = fs::symlink_metadata(path)?;
    if !m.is_dir() || m.uid() != unsafe { libc::geteuid() } || m.mode() & 0o077 != 0 {
        bail!(
            "memory coordinator directory is not private: {}",
            path.display()
        );
    }
    Ok(())
}

fn secure_file(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let m = file.metadata()?;
    if !m.is_file()
        || m.uid() != unsafe { libc::geteuid() }
        || m.mode() & 0o077 != 0
        || m.nlink() != 1
    {
        bail!("memory coordinator file is not private: {}", path.display());
    }
    Ok(file)
}

fn try_lock(file: &File, mode: i32) -> Result<bool> {
    if unsafe { libc::flock(file.as_raw_fd(), mode | libc::LOCK_NB) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock
        || error.kind() == std::io::ErrorKind::Interrupted
    {
        Ok(false)
    } else {
        Err(error.into())
    }
}

struct Waiting<'a> {
    coordinator: &'a Coordinator,
    ticket: u64,
}
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.coordinator
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .retain(|t| *t != self.ticket);
        self.coordinator.waiters.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Owns the cross-process resident-model slot. Drop model allocations before this lease.
pub struct Lease {
    coordinator: Arc<Coordinator>,
    owner: File,
    intent: File,
}

/// Declare before loader-owned tensors so unwinding drains submitted GPU work
/// after their drop and before the enclosing ownership lease can be released.
pub struct DeviceDrain(pub Device);

impl Drop for DeviceDrain {
    fn drop(&mut self) {
        if let Err(error) = self.0.synchronize() {
            eprintln!(
                "cannot drain accelerator work safely: {error}; terminating without releasing resident ownership early"
            );
            std::process::abort();
        }
    }
}

impl Coordinator {
    pub fn acquire(
        self: &Arc<Self>,
        label: &str,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<Lease> {
        let start = Instant::now();
        let mut reported_wait = false;
        secure_directory(&self.directory)?;
        let owner = secure_file(&self.directory.join("owner.lock"))?;
        let intent = secure_file(&self.directory.join("waiters.lock"))?;
        let ticket = {
            let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            let ticket = queue.0;
            queue.0 = queue.0.wrapping_add(1);
            queue.1.push_back(ticket);
            ticket
        };
        self.waiters.fetch_add(1, Ordering::AcqRel);
        let waiting = Waiting {
            coordinator: self,
            ticket,
        };
        while !try_lock(&intent, libc::LOCK_SH)? {
            cancel()?;
            report_wait(&mut reported_wait, label, Pressure::Unknown);
            thread::sleep(POLL);
        }
        loop {
            cancel()?;
            let pressure = self
                .latest
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pressure;
            if !matches!(pressure, Pressure::Warning | Pressure::Critical)
                && self
                    .queue
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .1
                    .front()
                    == Some(&ticket)
                && Instant::now()
                    >= *self
                        .reacquire_after
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                && self
                    .owned
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                match try_lock(&owner, libc::LOCK_EX) {
                    Ok(true) => {
                        unsafe {
                            libc::flock(intent.as_raw_fd(), libc::LOCK_UN);
                        }
                        drop(waiting);
                        self.managed.store(true, Ordering::Release);
                        if reported_wait {
                            crate::host_log::host_line(format!(
                                "xwen: {label}: resident memory ownership acquired after {:.2}s",
                                start.elapsed().as_secs_f64()
                            ));
                        }
                        let lease = Lease {
                            coordinator: Arc::clone(self),
                            owner,
                            intent,
                        };
                        self.record(
                            label,
                            &self.latest.lock().unwrap_or_else(|e| e.into_inner()),
                        );
                        return Ok(lease);
                    }
                    Ok(false) => self.owned.store(false, Ordering::Release),
                    Err(e) => {
                        self.owned.store(false, Ordering::Release);
                        return Err(e);
                    }
                }
            }
            report_wait(&mut reported_wait, label, pressure);
            thread::sleep(POLL);
        }
    }

    fn record(&self, event: &str, snapshot: &Snapshot) {
        // Labels are internal event names; callers must never pass prompt contents.
        if let Some(file) = self
            .telemetry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            let entry = serde_json::json!({"event": event, "pid": std::process::id(),
                "reservation_bytes": self.reservation.load(Ordering::Acquire), "memory": snapshot});
            write_telemetry(file, &entry);
        }
    }

    fn record_admission(&self, label: &str, snapshot: &Snapshot, projected: u64, accepted: bool) {
        if let Some(file) = self
            .telemetry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            write_telemetry(
                file,
                &serde_json::json!({"event":"admission", "label":label,
                "pid":std::process::id(), "projected_bytes":projected, "accepted":accepted,
                "reservation_bytes":self.reservation.load(Ordering::Acquire), "memory":snapshot}),
            );
        }
    }
}

fn report_wait(reported: &mut bool, label: &str, pressure: Pressure) {
    if *reported {
        return;
    }
    *reported = true;
    let reason = if matches!(pressure, Pressure::Warning | Pressure::Critical) {
        format!("system memory pressure is {pressure:?}")
    } else {
        "another resident model owns the accelerator".to_owned()
    };
    crate::host_log::host_line(format!(
        "xwen: {label}: waiting for resident memory ownership ({reason})"
    ));
}

fn startup_entry() -> serde_json::Value {
    let executable = std::env::current_exe().ok();
    let metadata = executable.as_ref().and_then(|p| fs::metadata(p).ok());
    let modified = metadata
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|t| t.as_secs());
    fn knob(name: &str, allowed: &[&str]) -> Option<String> {
        std::env::var(name).ok().map(|value| {
            let value = value.trim().to_ascii_lowercase();
            if allowed.contains(&value.as_str()) {
                value
            } else {
                "unrecognized".to_owned()
            }
        })
    }
    serde_json::json!({"event":"startup", "pid":std::process::id(),
    "timestamp_ms":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
    "executable":executable, "executable_modified_epoch":modified,
    "executable_bytes":metadata.map(|m|m.len()), "crate_version":env!("CARGO_PKG_VERSION"),
    "knobs":{
        "XWEN_ZIMAGE_ATTN":knob("XWEN_ZIMAGE_ATTN", &["","tensor","xwen","flash","steel","fused","basic"]),
        "XWEN_ZIMAGE_LINEAR":knob("XWEN_ZIMAGE_LINEAR", &["","xwen","tensor","candle"]),
        "XWEN_ZIMAGE_VAE":knob("XWEN_ZIMAGE_VAE", &["","xwen","direct","candle"])
    }})
}

fn write_telemetry(file: &mut File, entry: &serde_json::Value) {
    // Keeping the inode stable lets all processes coordinate writes and bounded truncation.
    if !matches!(try_lock(file, libc::LOCK_EX), Ok(true)) {
        return;
    }
    if file.metadata().is_ok_and(|m| m.len() >= 16 * 1024 * 1024) && file.set_len(0).is_err() {
        unsafe {
            libc::flock(file.as_raw_fd(), libc::LOCK_UN);
        }
        return;
    }
    let _ = writeln!(file, "{entry}");
    unsafe {
        libc::flock(file.as_raw_fd(), libc::LOCK_UN);
    }
}

impl Lease {
    pub fn should_yield(&self) -> bool {
        if self.coordinator.waiters.load(Ordering::Acquire) > 0 {
            return true;
        }
        let snapshot = self
            .coordinator
            .latest
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if snapshot.pressure == Pressure::Warning || runtime_stop_reason(&snapshot).is_some() {
            return true;
        }
        drop(snapshot);
        match try_lock(&self.intent, libc::LOCK_EX) {
            Ok(true) => {
                unsafe {
                    libc::flock(self.intent.as_raw_fd(), libc::LOCK_UN);
                }
                false
            }
            Ok(false) | Err(_) => true,
        }
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.should_yield() {
            // A yielding process waits through several polling intervals so existing
            // cross-process waiters can acquire before its next local request.
            *self
                .coordinator
                .reacquire_after
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = Instant::now() + POLL * 3;
        }
        self.coordinator.managed.store(false, Ordering::Release);
        self.coordinator.reservation.store(0, Ordering::Release);
        self.coordinator.record(
            "lease_released",
            &self
                .coordinator
                .latest
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        unsafe {
            libc::flock(self.owner.as_raw_fd(), libc::LOCK_UN);
        }
        self.coordinator.owned.store(false, Ordering::Release);
    }
}

pub fn acquire(label: &str, cancel: &dyn Fn() -> Result<()>) -> Result<Lease> {
    coordinator().acquire(label, cancel)
}

/// Whether this process currently owns a managed resident-model slot.
pub fn is_managed() -> bool {
    GLOBAL
        .get()
        .is_some_and(|c| c.managed.load(Ordering::Acquire))
}

pub fn admit_additional(label: &str, additional_bytes: u64) -> Result<()> {
    admit(label, additional_bytes)
}

/// Admit a new allocation peak atop measured host use, up to physical RAM at normal
/// pressure. Unknown pressure retains a reserve; warning and critical refuse admission.
/// Existing allocations receive no footprint credit: Mach footprint and host resident
/// accounting differ. Warm callers should pass only their additional allocation bound.
pub fn admit(label: &str, projected_peak_bytes: u64) -> Result<()> {
    let snapshot = host_sample();
    *coordinator()
        .latest
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = snapshot.clone();
    let result = admit_snapshot(&snapshot, projected_peak_bytes);
    if result.is_ok() {
        let _ = coordinator()
            .reservation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_add(projected_peak_bytes))
            });
    }
    coordinator().record_admission(label, &snapshot, projected_peak_bytes, result.is_ok());
    result
}

fn admit_snapshot(s: &Snapshot, projected: u64) -> Result<()> {
    if matches!(s.pressure, Pressure::Warning | Pressure::Critical) {
        bail!(
            "system memory pressure is {:?}; wait for memory to recover",
            s.pressure
        );
    }
    let physical = s
        .physical_bytes
        .context("cannot read physical RAM for memory admission")?;
    let used = s
        .system_used_bytes
        .context("cannot read system memory use for memory admission")?;
    s.process_footprint_bytes
        .context("cannot read process footprint for memory admission")?;
    let (budget, reserve) = system_budget(physical, s.pressure);
    // Do not subtract process footprint from a differently accounted system counter.
    let future = used
        .checked_add(projected)
        .context("projected memory use overflows")?;
    if future > budget {
        bail!(
            "memory admission refused: projected system use {:.1} GiB exceeds {:.1} GiB budget ({:.1} GiB reserved)",
            future as f64 / GIB as f64,
            budget as f64 / GIB as f64,
            reserve as f64 / GIB as f64
        );
    }
    Ok(())
}

/// Cached host check; safe to call between decode steps without querying the kernel.
pub fn check_runtime() -> Result<()> {
    if let Some(c) = GLOBAL.get() {
        let snapshot = c.latest.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(reason) = runtime_stop_reason(&snapshot) {
            c.record(reason, &snapshot);
            bail!("{reason}; stopping inference to preserve system memory headroom");
        }
    }
    Ok(())
}

fn runtime_stop_reason(snapshot: &Snapshot) -> Option<&'static str> {
    if snapshot.pressure == Pressure::Critical {
        return Some("runtime: system memory pressure is critical");
    }
    if let (Some(physical), Some(used)) = (snapshot.physical_bytes, snapshot.system_used_bytes) {
        let (budget, _) = system_budget(physical, snapshot.pressure);
        if used >= budget {
            return Some("runtime: system memory headroom exhausted");
        }
    }
    None
}

fn system_budget(physical: u64, pressure: Pressure) -> (u64, u64) {
    let reserve = if pressure == Pressure::Normal {
        0
    } else {
        MIN_SYSTEM_RESERVE.max(physical.saturating_mul(SYSTEM_RESERVE_PERCENT) / 100)
    };
    (physical.saturating_sub(reserve), reserve)
}

pub fn sample(_label: &str, device: Option<&Device>) -> Snapshot {
    let mut snapshot = host_sample();
    if let Some(metal) = device.and_then(|d| d.as_metal_device().ok()) {
        snapshot.metal_allocated_bytes = Some(metal.device().current_allocated_size() as u64);
        snapshot.metal_recommended_bytes =
            Some(metal.device().recommended_max_working_set_size() as u64);
    }
    snapshot
}

pub fn log_event(label: &str, device: Option<&Device>) {
    coordinator().record(label, &sample(label, device));
}

/// Conservative initial admission envelope; larger resolutions need measured peak evidence.
pub fn image_peak(width: u32, height: u32, control: bool, loras: usize) -> Result<u64> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .context("image pixel count overflow")?;
    if width == 0 || height == 0 || pixels > 1024 * 1024 {
        bail!("memory admission supports images up to 1,048,576 pixels with nonzero dimensions");
    }
    let adapters = u64::try_from(loras)?
        .checked_mul(8 * GIB)
        .context("LoRA memory estimate overflow")?;
    (40 * GIB)
        .checked_add(if control { 24 * GIB } else { 0 })
        .and_then(|n| n.checked_add(adapters))
        .context("image memory estimate overflow")
}

fn sysctl_value<T: Copy>(name: &CStr) -> Option<T> {
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let mut size = std::mem::size_of::<T>();
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result == 0 && size == std::mem::size_of::<T>() {
        Some(unsafe { value.assume_init() })
    } else {
        None
    }
}

#[allow(deprecated)] // These stable system Mach ABI aliases avoid another dependency.
fn host_sample() -> Snapshot {
    let physical = sysctl_value::<u64>(c"hw.memsize");
    let pressure = match sysctl_value::<u32>(c"kern.memorystatus_vm_pressure_level") {
        Some(1) => Pressure::Normal,
        Some(2) => Pressure::Warning,
        Some(4) => Pressure::Critical,
        _ => Pressure::Unknown,
    };
    let mut vm = unsafe { std::mem::zeroed::<libc::vm_statistics64>() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let host = unsafe { libc::mach_host_self() };
    let result = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut vm as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    unsafe {
        mach_port_deallocate(libc::mach_task_self(), host);
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let used = if result == 0 && page_size > 0 {
        Some(host_used_bytes(
            vm.internal_page_count,
            vm.wire_count,
            vm.compressor_page_count,
            page_size as u64,
        ))
    } else {
        None
    };
    // TASK_VM_INFO revision 1 places phys_footprint at byte 144. A larger aligned
    // output buffer accommodates later revisions; the returned count proves presence.
    let mut task = [0u64; 64];
    let mut task_count = (std::mem::size_of_val(&task) / 4) as u32;
    let task_result = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            22,
            task.as_mut_ptr().cast(),
            &mut task_count,
        )
    };
    let footprint = if task_result == 0 && task_count >= 38 {
        Some(task[18])
    } else {
        None
    };
    Snapshot {
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
        physical_bytes: physical,
        system_used_bytes: used,
        process_footprint_bytes: footprint,
        pressure,
        metal_allocated_bytes: None,
        metal_recommended_bytes: None,
    }
}

fn host_used_bytes(internal: u32, wired: u32, compressed: u32, page_size: u64) -> u64 {
    // vm_statistics64 reports anonymous resident pages separately from file-backed
    // pages. Include wired and physical compressor storage without crediting caches.
    (u64::from(internal) + u64::from(wired) + u64::from(compressed)).saturating_mul(page_size)
}

fn open_telemetry() -> Option<File> {
    let result = (|| -> Result<File> {
        let mut record = unsafe { std::mem::zeroed::<libc::passwd>() };
        let mut storage = vec![0u8; 16384];
        let mut found = std::ptr::null_mut();
        let status = unsafe {
            libc::getpwuid_r(
                libc::geteuid(),
                &mut record,
                storage.as_mut_ptr().cast(),
                storage.len(),
                &mut found,
            )
        };
        if status != 0 || found.is_null() {
            bail!("cannot resolve account home");
        }
        let home = unsafe { CStr::from_ptr(record.pw_dir) }.to_str()?;
        let directory = PathBuf::from(home).join(".local/state/xwen");
        fs::create_dir_all(&directory)?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(directory.join("memory.jsonl"))
            .map_err(Into::into)
    })();
    match result {
        Ok(file) => Some(file),
        Err(e) => {
            eprintln!("memory telemetry unavailable: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_metrics_smoke() {
        let s = host_sample();
        eprintln!("{s:?}");
        assert!(s.physical_bytes.is_some_and(|n| n > 0));
        assert!(s.process_footprint_bytes.is_some_and(|n| n > 0));
        assert!(s.system_used_bytes.is_some_and(|n| n > 0));
    }
    fn snapshot() -> Snapshot {
        Snapshot {
            timestamp_ms: 0,
            physical_bytes: Some(128 * GIB),
            system_used_bytes: Some(30 * GIB),
            process_footprint_bytes: Some(20 * GIB),
            pressure: Pressure::Normal,
            metal_allocated_bytes: None,
            metal_recommended_bytes: None,
        }
    }
    fn isolated() -> Arc<Coordinator> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let directory = PathBuf::from(format!(
            "/private/tmp/xwen-memory-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        Arc::new(Coordinator {
            directory,
            owned: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            queue: Mutex::new((0, VecDeque::new())),
            reacquire_after: Mutex::new(Instant::now()),
            latest: Mutex::new(snapshot()),
            managed: AtomicBool::new(false),
            reservation: AtomicU64::new(0),
            telemetry: Mutex::new(None),
        })
    }
    #[test]
    fn normal_pressure_admits_flash_next_load_and_request() {
        let mut s = snapshot();
        s.system_used_bytes = Some(24_545_001_472);
        assert!(admit_snapshot(&s, 93_446_057_472).is_ok());
        s.system_used_bytes = Some(121_846_759_424);
        s.process_footprint_bytes = Some(20_160_107_960);
        assert!(admit_snapshot(&s, 1_551_679_488).is_ok());
        assert!(runtime_stop_reason(&s).is_none());
    }
    #[test]
    fn normal_pressure_uses_physical_limit_without_footprint_credit() {
        let mut s = snapshot();
        assert!(admit_snapshot(&s, 98 * GIB).is_ok());
        let error = admit_snapshot(&s, 98 * GIB + 1).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("128.0 GiB budget (0.0 GiB reserved)")
        );
        s.process_footprint_bytes = Some(127 * GIB);
        assert!(admit_snapshot(&s, 99 * GIB).is_err());
        s.system_used_bytes = Some(128 * GIB - 1);
        assert!(runtime_stop_reason(&s).is_none());
        s.system_used_bytes = Some(128 * GIB);
        assert!(runtime_stop_reason(&s).is_some());
        s.system_used_bytes = Some(128 * GIB + 1);
        assert!(runtime_stop_reason(&s).is_some());
    }
    #[test]
    fn unknown_pressure_never_credits_process_footprint_and_keeps_reserve() {
        let mut s = snapshot();
        s.pressure = Pressure::Unknown;
        assert!(admit_snapshot(&s, 70 * GIB).is_ok());
        assert!(admit_snapshot(&s, 82 * GIB).is_ok());
        assert!(admit_snapshot(&s, 82 * GIB + 1).is_err());
        assert!(admit_snapshot(&s, 90 * GIB).is_err());
        s.process_footprint_bytes = Some(127 * GIB);
        assert!(admit_snapshot(&s, 90 * GIB).is_err());
        s.pressure = Pressure::Warning;
        assert!(admit_snapshot(&s, GIB).is_err());
        s.pressure = Pressure::Critical;
        assert!(admit_snapshot(&s, GIB).is_err());
        s.pressure = Pressure::Unknown;
        assert!(admit_snapshot(&s, GIB).is_ok());
        s.physical_bytes = None;
        assert!(admit_snapshot(&s, GIB).is_err());
    }
    #[test]
    fn admission_requires_counters_and_rejects_overflow() {
        for pressure in [Pressure::Normal, Pressure::Unknown] {
            for missing in 0..3 {
                let mut s = snapshot();
                s.pressure = pressure;
                match missing {
                    0 => s.physical_bytes = None,
                    1 => s.system_used_bytes = None,
                    _ => s.process_footprint_bytes = None,
                }
                assert!(admit_snapshot(&s, 0).is_err());
            }
            let mut s = snapshot();
            s.pressure = pressure;
            assert!(
                admit_snapshot(&s, u64::MAX)
                    .unwrap_err()
                    .to_string()
                    .contains("overflows")
            );
        }
    }
    #[test]
    fn image_bounds_and_overflow() {
        assert_eq!(image_peak(1024, 1024, false, 0).unwrap(), 40 * GIB);
        assert!(image_peak(1025, 1024, false, 0).is_err());
        assert!(image_peak(0, 512, false, 0).is_err());
        assert!(image_peak(512, 512, true, usize::MAX).is_err());
    }
    #[test]
    fn host_accounting_adds_anonymous_wired_and_physical_compressor_pages() {
        assert_eq!(host_used_bytes(23, 5, 2, 16384), 30 * 16384);
        assert_eq!(
            host_used_bytes(u32::MAX, u32::MAX, u32::MAX, u64::MAX),
            u64::MAX
        );
    }
    #[test]
    fn runtime_budget_applies_without_pressure_notifications() {
        let mut s = snapshot();
        assert!(runtime_stop_reason(&s).is_none());
        let physical = s.physical_bytes.unwrap();
        let (budget, _) = system_budget(physical, Pressure::Unknown);
        s.pressure = Pressure::Unknown;
        s.system_used_bytes = Some(budget);
        assert_eq!(
            runtime_stop_reason(&s),
            Some("runtime: system memory headroom exhausted")
        );
        s.pressure = Pressure::Normal;
        assert!(runtime_stop_reason(&s).is_none());
        s.pressure = Pressure::Warning;
        assert!(runtime_stop_reason(&s).is_some());
        s.system_used_bytes = Some(budget - 1);
        assert!(runtime_stop_reason(&s).is_none());
        s.pressure = Pressure::Critical;
        assert_eq!(
            runtime_stop_reason(&s),
            Some("runtime: system memory pressure is critical")
        );
    }
    #[test]
    fn queued_local_waiter_precedes_former_owners_next_request() {
        let a = isolated();
        let first = a.acquire("first", &|| Ok(())).unwrap();
        let order = Arc::new(AtomicUsize::new(0));
        let peer = Arc::clone(&a);
        let peer_order = Arc::clone(&order);
        let waiter = thread::spawn(move || {
            let _lease = peer.acquire("waiting", &|| Ok(())).unwrap();
            assert_eq!(peer_order.fetch_add(1, Ordering::SeqCst), 0);
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while a.waiters.load(Ordering::Acquire) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(a.waiters.load(Ordering::Acquire), 1);
        assert!(first.should_yield());
        drop(first);
        let next = a.acquire("next", &|| Ok(())).unwrap();
        assert_eq!(order.fetch_add(1, Ordering::SeqCst), 1);
        waiter.join().unwrap();
        drop(next);
        fs::remove_dir_all(&a.directory).unwrap();
    }
    #[test]
    fn telemetry_is_bounded_without_replacing_the_inode() {
        let c = isolated();
        secure_directory(&c.directory).unwrap();
        let path = c.directory.join("events");
        let mut first = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&path)
            .unwrap();
        let second = OpenOptions::new().append(true).open(&path).unwrap();
        first.set_len(16 * 1024 * 1024).unwrap();
        let inode = first.metadata().unwrap().ino();
        write_telemetry(&mut first, &serde_json::json!({"event":"test"}));
        assert_eq!(second.metadata().unwrap().ino(), inode);
        assert!(second.metadata().unwrap().len() < 1024);
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"event\":\"test\"}\n");
        drop(first);
        drop(second);
        fs::remove_dir_all(&c.directory).unwrap();
    }
    #[test]
    fn cancellation_raii_and_cross_coordinator_intent() {
        let a = isolated();
        let holder = a.acquire("test", &|| Ok(())).unwrap();
        assert!(!holder.should_yield());
        let b = Arc::new(Coordinator {
            directory: a.directory.clone(),
            owned: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            queue: Mutex::new((0, VecDeque::new())),
            reacquire_after: Mutex::new(Instant::now()),
            latest: Mutex::new(snapshot()),
            managed: AtomicBool::new(false),
            reservation: AtomicU64::new(0),
            telemetry: Mutex::new(None),
        });
        let cancelled = Arc::new(AtomicBool::new(false));
        let c = Arc::clone(&cancelled);
        let peer = Arc::clone(&b);
        let thread = thread::spawn(move || {
            peer.acquire("test", &|| {
                if c.load(Ordering::Acquire) {
                    bail!("cancelled")
                } else {
                    Ok(())
                }
            })
            .is_err()
        });
        for _ in 0..100 {
            if holder.should_yield() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(holder.should_yield());
        cancelled.store(true, Ordering::Release);
        assert!(thread.join().unwrap());
        assert_eq!(b.waiters.load(Ordering::Acquire), 0);
        assert!(!holder.should_yield());
        drop(holder);
        let lease = b.acquire("test", &|| Ok(())).unwrap();
        drop(lease);
        assert!(!b.owned.load(Ordering::Acquire));
        fs::remove_dir_all(&a.directory).unwrap();
    }
    #[test]
    fn pressure_yields_and_cancelled_admission_does_not_own() {
        let a = isolated();
        let lease = a.acquire("test", &|| Ok(())).unwrap();
        a.latest.lock().unwrap().pressure = Pressure::Warning;
        assert!(lease.should_yield());
        drop(lease);
        assert!(a.acquire("test", &|| bail!("cancelled")).is_err());
        assert!(!a.owned.load(Ordering::Acquire));
        assert_eq!(a.waiters.load(Ordering::Acquire), 0);
        fs::remove_dir_all(&a.directory).unwrap();
    }
    #[test]
    fn concurrent_threads_never_share_ownership() {
        let a = isolated();
        let active = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..6)
            .map(|_| {
                let a = Arc::clone(&a);
                let active = Arc::clone(&active);
                thread::spawn(move || {
                    let _lease = a.acquire("test", &|| Ok(())).unwrap();
                    assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
                    thread::sleep(Duration::from_millis(5));
                    assert_eq!(active.fetch_sub(1, Ordering::SeqCst), 1);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        fs::remove_dir_all(&a.directory).unwrap();
    }
    #[test]
    fn crossprocess_waiter() {
        // Only this test binary accepts an injected path; production locks have a fixed location.
        if let Some(path) = std::env::var_os("XWEN_MEMORY_TEST_CHILD") {
            let mut child = isolated();
            Arc::get_mut(&mut child).unwrap().directory = PathBuf::from(path);
            let _lease = child.acquire("child", &|| Ok(())).unwrap();
            fs::write(child.directory.join("child-acquired"), b"yes").unwrap();
            return;
        }
        let a = isolated();
        let lease = a.acquire("parent", &|| Ok(())).unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "memory::tests::crossprocess_waiter",
                "--nocapture",
            ])
            .env("XWEN_MEMORY_TEST_CHILD", &a.directory)
            .spawn()
            .unwrap();
        let mut yielded = false;
        for _ in 0..200 {
            if lease.should_yield() {
                yielded = true;
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let before = a.directory.join("child-acquired").exists();
        drop(lease);
        let status = child.wait().unwrap();
        assert!(yielded);
        assert!(!before);
        assert!(status.success());
        assert!(a.directory.join("child-acquired").exists());
        fs::remove_dir_all(&a.directory).unwrap();
    }
}
