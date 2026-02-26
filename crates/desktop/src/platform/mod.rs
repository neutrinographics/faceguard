#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

/// Detect whether the operating system is in dark mode.
pub fn is_dark_mode() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::is_dark_mode()
    }
    #[cfg(target_os = "windows")]
    {
        windows::is_dark_mode()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        true
    }
}
