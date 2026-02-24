use iced::color;
use iced::theme::Palette;
use iced::Theme;

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

fn dark_palette() -> Palette {
    Palette {
        background: color!(0x1c, 0x1c, 0x1e),
        text: color!(0xcc, 0xcc, 0xcc),
        primary: color!(0x5e, 0x9f, 0xf5),
        success: color!(0x30, 0xd1, 0x58),
        warning: color!(0xff, 0xcc, 0x00),
        danger: color!(0xff, 0x45, 0x3a),
    }
}

fn light_palette() -> Palette {
    Palette {
        background: color!(0xf5, 0xf5, 0xf7),
        text: color!(0x1d, 0x1d, 0x1f),
        primary: color!(0x34, 0x78, 0xf6),
        success: color!(0x34, 0xc7, 0x59),
        warning: color!(0xff, 0x9f, 0x0a),
        danger: color!(0xff, 0x3b, 0x30),
    }
}

fn high_contrast_dark_palette() -> Palette {
    Palette {
        background: color!(0x00, 0x00, 0x00),
        text: color!(0xff, 0xff, 0xff),
        primary: color!(0x6c, 0xb4, 0xff),
        success: color!(0x30, 0xd1, 0x58),
        warning: color!(0xff, 0xd6, 0x0a),
        danger: color!(0xff, 0x45, 0x3a),
    }
}

fn high_contrast_light_palette() -> Palette {
    Palette {
        background: color!(0xff, 0xff, 0xff),
        text: color!(0x00, 0x00, 0x00),
        primary: color!(0x00, 0x50, 0xd0),
        success: color!(0x24, 0x8a, 0x3d),
        warning: color!(0xb2, 0x5c, 0x00),
        danger: color!(0xd7, 0x00, 0x15),
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
