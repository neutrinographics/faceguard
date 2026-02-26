use iced::widget::{button, checkbox, column, container, row, slider, text, Space};
use iced::{Color, Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::settings::{Appearance, BlurShape, Settings};
use crate::theme::{muted_color, section_color, surface_color, tertiary_color};
use crate::widgets::secondary_button;

pub fn view<'a>(
    settings: &Settings,
    gpu_available: bool,
    restore_defaults_hovered: bool,
) -> Element<'a, Message> {
    let fs = settings.font_scale;
    let theme = crate::theme::resolve_theme(settings.appearance, settings.high_contrast);
    let muted = muted_color(&theme);
    let section = section_color(&theme);
    let tertiary = tertiary_color(&theme);
    let surface = surface_color(&theme);
    let border = border_light_color(&theme);
    let accent = theme.palette().primary;

    let restore_btn = secondary_button::secondary_button_small(
        move || text("Restore Defaults").size(scaled(14.0, fs)).into(),
        Message::RestoreDefaults,
        restore_defaults_hovered,
        Message::RestoreDefaultsHover,
        [8, 18],
    );

    column![
        blur_section(
            settings,
            fs,
            muted,
            section,
            tertiary,
            surface,
            border,
            accent,
            gpu_available
        ),
        Space::new().height(28),
        detection_section(settings, fs, muted, section, tertiary, surface, border, accent),
        Space::new().height(28),
        appearance_section(settings, fs, section, tertiary, surface, border, accent),
        Space::new().height(24),
        restore_btn,
    ]
    .spacing(0)
    .into()
}

fn setting_card<'a>(
    content: impl Into<Element<'a, Message>>,
    surface: Color,
    border_color: Color,
) -> Element<'a, Message> {
    container(content)
        .padding(18)
        .width(Length::Fill)
        .style(move |_theme: &Theme| container::Style {
            background: Some(surface.into()),
            border: iced::border::Border {
                color: border_color,
                width: 1.0,
                radius: 12.0.into(),
            },
            ..container::Style::default()
        })
        .into()
}

fn blur_section<'a>(
    settings: &Settings,
    fs: f32,
    _muted: iced::Color,
    section: iced::Color,
    tertiary: iced::Color,
    surface: iced::Color,
    border: iced::Color,
    accent: iced::Color,
    gpu_available: bool,
) -> Element<'a, Message> {
    let intensity_label = blur_intensity_label(settings.blur_strength);
    let backend_label = if gpu_available {
        "Blur backend: GPU accelerated"
    } else {
        "Blur backend: CPU"
    };

    let shape_pills: Element<'a, Message> = row(BlurShape::ALL.iter().map(|&variant| {
        let is_active = variant == settings.blur_shape;
        pill_button(
            variant.to_string(),
            is_active,
            Message::BlurShapeChanged(variant),
            fs,
            accent,
            border,
            tertiary,
        )
    }))
    .spacing(8)
    .into();

    let shape_card = setting_card(
        column![
            setting_name("Shape", fs),
            Space::new().height(8),
            shape_pills,
            Space::new().height(4),
            text(match settings.blur_shape {
                BlurShape::Ellipse => "Ellipse follows the natural shape of a face.",
                BlurShape::Rect => "Rectangle covers a wider area.",
            })
            .size(scaled(14.0, fs))
            .color(tertiary),
        ]
        .spacing(0),
        surface,
        border,
    );

    let intensity_card = setting_card(
        column![
            row![
                setting_name("Intensity", fs),
                Space::new().width(Length::Fill),
                value_badge(intensity_label, fs, accent),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("How heavily faces are blurred.")
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(
                51..=401,
                settings.blur_strength,
                Message::BlurStrengthChanged
            )
            .step(2u32)
            .style(slider_style),
        ]
        .spacing(0),
        surface,
        border,
    );

    let quality_card = setting_card(
        column![
            row![
                setting_name("Output quality", fs),
                Space::new().width(Length::Fill),
                value_badge(quality_label(settings.quality), fs, accent),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("Higher quality produces larger files.")
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(0..=100, settings.quality, Message::QualityChanged).style(slider_style),
            Space::new().height(8),
            text(backend_label).size(scaled(12.0, fs)).color(tertiary),
        ]
        .spacing(0),
        surface,
        border,
    );

    column![
        section_label("BLUR", fs, section),
        Space::new().height(14),
        shape_card,
        Space::new().height(10),
        intensity_card,
        Space::new().height(10),
        quality_card,
    ]
    .spacing(0)
    .into()
}

fn detection_section<'a>(
    settings: &Settings,
    fs: f32,
    _muted: iced::Color,
    section: iced::Color,
    tertiary: iced::Color,
    surface: iced::Color,
    border: iced::Color,
    accent: iced::Color,
) -> Element<'a, Message> {
    let sens_label = sensitivity_label(settings.confidence);

    let sensitivity_card = setting_card(
        column![
            row![
                setting_name("Sensitivity", fs),
                Space::new().width(Length::Fill),
                value_badge(sens_label, fs, accent),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("How certain the detector must be that something is a face. Lower values catch more faces but may have false positives.")
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(10..=100, settings.confidence, Message::ConfidenceChanged).style(slider_style),
        ]
        .spacing(0),
        surface,
        border,
    );

    let lookahead_card = setting_card(
        column![
            row![
                setting_name("Lookahead", fs),
                Space::new().width(Length::Fill),
                value_badge(format!("{} frames", settings.lookahead), fs, accent),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("Scan ahead to blur faces before they fully enter the frame. A larger value helps with fast moving faces.")
                .size(scaled(14.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(0..=30, settings.lookahead, Message::LookaheadChanged)
                .style(slider_style),
        ]
        .spacing(0),
        surface,
        border,
    );

    column![
        section_label("DETECTION", fs, section),
        Space::new().height(14),
        sensitivity_card,
        Space::new().height(10),
        lookahead_card,
    ]
    .spacing(0)
    .into()
}

fn appearance_section<'a>(
    settings: &Settings,
    fs: f32,
    section: iced::Color,
    tertiary: iced::Color,
    surface: iced::Color,
    border: iced::Color,
    accent: iced::Color,
) -> Element<'a, Message> {
    let current = settings.appearance;
    let theme_pills: Element<'a, Message> = row(Appearance::ALL.iter().map(|&variant| {
        let is_active = variant == current;
        pill_button(
            variant.to_string(),
            is_active,
            Message::AppearanceChanged(variant),
            fs,
            accent,
            border,
            tertiary,
        )
    }))
    .spacing(8)
    .into();

    let theme_card = setting_card(
        column![
            setting_name("Theme", fs),
            Space::new().height(10),
            theme_pills,
            Space::new().height(12),
            checkbox(settings.high_contrast)
                .label("High contrast")
                .on_toggle(Message::HighContrastChanged)
                .text_size(scaled(14.0, fs)),
        ]
        .spacing(0),
        surface,
        border,
    );

    let scale_card = setting_card(
        column![
            row![
                setting_name("Interface scale", fs),
                Space::new().width(Length::Fill),
                value_badge(format!("{:.0}%", settings.font_scale * 100.0), fs, accent),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(12),
            slider(0.8..=1.5, settings.font_scale, Message::FontScaleChanged)
                .step(0.05)
                .style(slider_style),
        ]
        .spacing(0),
        surface,
        border,
    );

    column![
        section_label("APPEARANCE", fs, section),
        Space::new().height(14),
        theme_card,
        Space::new().height(10),
        scale_card,
    ]
    .spacing(0)
    .into()
}

fn blur_intensity_label(strength: u32) -> String {
    let qual = match strength {
        51..=150 => "Light",
        151..=275 => "Medium",
        _ => "Heavy",
    };
    qual.to_string()
}

fn quality_label(quality: u32) -> String {
    let qual = match quality {
        0..=25 => "Low",
        26..=55 => "Medium",
        56..=80 => "High",
        _ => "Very high",
    };
    format!("{qual} ({quality}%)")
}

fn sensitivity_label(confidence: u32) -> String {
    let qual = match confidence {
        10..=35 => "Low",
        36..=65 => "Medium",
        _ => "High",
    };
    format!("{qual} ({confidence}%)")
}

fn section_label<'a>(label: &str, fs: f32, color: Color) -> iced::widget::Text<'a> {
    text(label.to_string())
        .size(scaled(12.0, fs))
        .color(color)
        .font(iced::Font {
            weight: iced::font::Weight::Bold,
            ..iced::Font::DEFAULT
        })
}

fn border_light_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.15, ..p.text }
}

fn setting_name<'a>(label: &str, fs: f32) -> iced::widget::Text<'a> {
    text(label.to_string())
        .size(scaled(15.0, fs))
        .font(iced::Font {
            weight: iced::font::Weight::Semibold,
            ..iced::Font::DEFAULT
        })
}

fn pill_button<'a>(
    label: String,
    is_active: bool,
    on_press: Message,
    fs: f32,
    accent: Color,
    border_color: Color,
    tertiary: Color,
) -> Element<'a, Message> {
    let (text_color, bg, pill_border_color, pill_border_width) = if is_active {
        let accent_bg = Color { a: 0.12, ..accent };
        (accent, accent_bg, accent, 1.5)
    } else {
        (tertiary, Color::TRANSPARENT, border_color, 1.0)
    };

    button(
        text(label)
            .size(scaled(14.0, fs))
            .color(text_color)
            .font(iced::Font {
                weight: iced::font::Weight::Semibold,
                ..iced::Font::DEFAULT
            }),
    )
    .on_press(on_press)
    .padding([6, 16])
    .style(move |_theme: &Theme, status| {
        let hovered = matches!(status, button::Status::Hovered | button::Status::Pressed);
        let bg = if !is_active && hovered {
            Color { a: 0.08, ..border_color }
        } else {
            bg
        };
        let pill_border_color = if !is_active && hovered {
            Color { a: 0.25, ..border_color }
        } else {
            pill_border_color
        };
        button::Style {
            background: Some(bg.into()),
            text_color,
            border: iced::border::Border {
                color: pill_border_color,
                width: pill_border_width,
                radius: 20.0.into(),
            },
            ..button::Style::default()
        }
    })
    .into()
}

fn value_badge<'a>(label: String, fs: f32, accent: Color) -> Element<'a, Message> {
    let accent_bg = Color { a: 0.12, ..accent };
    container(
        text(label)
            .size(scaled(14.0, fs))
            .color(accent)
            .font(iced::Font {
                weight: iced::font::Weight::Semibold,
                ..iced::Font::DEFAULT
            }),
    )
    .padding([2, 10])
    .style(move |_theme: &Theme| container::Style {
        background: Some(accent_bg.into()),
        border: iced::border::Border {
            radius: 20.0.into(),
            ..iced::border::Border::default()
        },
        ..container::Style::default()
    })
    .into()
}

fn slider_style(theme: &Theme, status: slider::Status) -> slider::Style {
    let p = theme.palette();
    let accent = p.primary;
    let mut s = slider::default(theme, status);
    s.rail.backgrounds.1 = p.background.into();
    s.rail.width = 6.0;
    s.rail.border.radius = 3.0.into();
    s.handle.shape = slider::HandleShape::Circle { radius: 8.0 };
    s.handle.background = Color::WHITE.into();
    s.handle.border_color = accent;
    s.handle.border_width = 2.0;
    s
}
