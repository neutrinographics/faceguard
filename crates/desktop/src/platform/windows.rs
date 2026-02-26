pub fn is_dark_mode() -> bool {
    // HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize
    // AppsUseLightTheme: DWORD 0 = dark, 1 = light
    std::process::Command::new("reg")
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            "/v",
            "AppsUseLightTheme",
        ])
        .output()
        .map(|o| {
            let stdout = String::from_utf8_lossy(&o.stdout);
            // Output contains "0x0" for dark mode, "0x1" for light mode
            stdout.contains("0x0")
        })
        .unwrap_or(true)
}
