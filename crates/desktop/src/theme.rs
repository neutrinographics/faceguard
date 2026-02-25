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

/// Return a muted text color (secondary) appropriate for the current theme.
pub fn muted_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.55, ..p.text }
}

/// Return a tertiary text color (dimmer than muted).
pub fn tertiary_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.45, ..p.text }
}

/// Return an elevated surface color (white in light theme, slightly lighter in dark).
pub fn surface_color(theme: &Theme) -> Color {
    let p = theme.palette();
    let luma = p.background.r * 0.299 + p.background.g * 0.587 + p.background.b * 0.114;
    if luma > 0.5 {
        Color::WHITE
    } else {
        Color {
            r: (p.background.r + 0.06).min(1.0),
            g: (p.background.g + 0.06).min(1.0),
            b: (p.background.b + 0.06).min(1.0),
            a: 1.0,
        }
    }
}

/// Return a section heading color (same as tertiary for section labels).
pub fn section_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.40, ..p.text }
}

fn dark_palette() -> Palette {
    Palette {
        background: color!(0x1c, 0x1a, 0x17),
        text: color!(0xed, 0xe8, 0xe2),
        primary: color!(0x5b, 0x86, 0xf0),
        success: color!(0x4a, 0xde, 0x80),
        warning: color!(0xfb, 0xbf, 0x24),
        danger: color!(0xf8, 0x71, 0x71),
    }
}

fn light_palette() -> Palette {
    Palette {
        background: color!(0xf7, 0xf5, 0xf2),
        text: color!(0x2d, 0x2a, 0x26),
        primary: color!(0x3b, 0x6c, 0xe7),
        success: color!(0x2e, 0x8b, 0x57),
        warning: color!(0xd4, 0x93, 0x0a),
        danger: color!(0xc9, 0x40, 0x3a),
    }
}

fn high_contrast_dark_palette() -> Palette {
    Palette {
        background: color!(0x00, 0x00, 0x00),
        text: color!(0xff, 0xff, 0xff),
        primary: color!(0x7b, 0xa0, 0xf7),
        success: color!(0x4a, 0xde, 0x80),
        warning: color!(0xfb, 0xbf, 0x24),
        danger: color!(0xf8, 0x71, 0x71),
    }
}

fn high_contrast_light_palette() -> Palette {
    Palette {
        background: color!(0xff, 0xff, 0xff),
        text: color!(0x00, 0x00, 0x00),
        primary: color!(0x2b, 0x52, 0xb8),
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
