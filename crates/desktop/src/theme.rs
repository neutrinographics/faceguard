use iced::color;
use iced::theme::Palette;
use iced::{Color, Theme};

use crate::settings::Appearance;

/// Resolve the iced Theme from appearance + high_contrast settings.
pub fn resolve_theme(appearance: Appearance, high_contrast: bool) -> Theme {
    let is_dark = match appearance {
        Appearance::Dark => true,
        Appearance::Light => false,
        Appearance::System => detect_system_dark_mode(),
    };

    let palette = match (is_dark, high_contrast) {
        (true, false) => dark_palette(),
        (false, false) => light_palette(),
        (true, true) => high_contrast_dark_palette(),
        (false, true) => high_contrast_light_palette(),
    };

    Theme::custom("Video Blur", palette)
}

/// Return a muted text color appropriate for the current theme.
pub fn muted_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.6, ..p.text }
}

/// Return a section heading color (dimmer than muted).
pub fn section_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.4, ..p.text }
}

fn dark_palette() -> Palette {
    Palette {
        background: color!(0x11, 0x11, 0x11),
        text: color!(0xe5, 0xe5, 0xe5),
        primary: color!(0x3b, 0x82, 0xf6),
        success: color!(0x30, 0xd1, 0x58),
        warning: color!(0xff, 0xcc, 0x00),
        danger: color!(0xff, 0x45, 0x3a),
    }
}

fn light_palette() -> Palette {
    Palette {
        background: color!(0xfa, 0xfa, 0xfa),
        text: color!(0x1a, 0x1a, 0x1a),
        primary: color!(0x25, 0x63, 0xeb),
        success: color!(0x34, 0xc7, 0x59),
        warning: color!(0xff, 0x9f, 0x0a),
        danger: color!(0xff, 0x3b, 0x30),
    }
}

fn high_contrast_dark_palette() -> Palette {
    Palette {
        background: color!(0x00, 0x00, 0x00),
        text: color!(0xff, 0xff, 0xff),
        primary: color!(0x60, 0xa5, 0xfa),
        success: color!(0x4a, 0xde, 0x80),
        warning: color!(0xfb, 0xbf, 0x24),
        danger: color!(0xf8, 0x71, 0x71),
    }
}

fn high_contrast_light_palette() -> Palette {
    Palette {
        background: color!(0xff, 0xff, 0xff),
        text: color!(0x00, 0x00, 0x00),
        primary: color!(0x1d, 0x4e, 0xd8),
        success: color!(0x15, 0x80, 0x3d),
        warning: color!(0xa1, 0x64, 0x07),
        danger: color!(0xb9, 0x1c, 0x1c),
    }
}

fn detect_system_dark_mode() -> bool {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("defaults")
            .args(["read", "-g", "AppleInterfaceStyle"])
            .output()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .eq_ignore_ascii_case("dark")
            })
            .unwrap_or(true)
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}
