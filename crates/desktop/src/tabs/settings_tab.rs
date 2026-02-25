use iced::widget::{button, checkbox, column, container, pick_list, row, slider, text, Space};
use iced::Element;

use crate::app::{scaled, Message};
use crate::settings::{Appearance, BlurShape, Settings};
use crate::theme::{muted_color, section_color, tertiary_color};

pub fn view<'a>(settings: &Settings, gpu_available: bool) -> Element<'a, Message> {
    let fs = settings.font_scale;
    let theme = crate::theme::resolve_theme(settings.appearance, settings.high_contrast);
    let muted = muted_color(&theme);
    let section = section_color(&theme);
    let tertiary = tertiary_color(&theme);

    column![
        blur_section(settings, fs, muted, section, tertiary, gpu_available),
        Space::new().height(28),
        detection_section(settings, fs, muted, section, tertiary),
        Space::new().height(28),
        appearance_section(settings, fs, section, tertiary),
        Space::new().height(24),
        button(text("Restore Defaults").size(scaled(13.0, fs)))
            .on_press(Message::RestoreDefaults)
            .padding([8, 18])
            .style(button::secondary),
    ]
    .spacing(0)
    .into()
}

fn setting_card<'a>(content: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    container(content)
        .padding(18)
        .style(container::rounded_box)
        .width(iced::Length::Fill)
        .into()
}

fn blur_section<'a>(
    settings: &Settings,
    fs: f32,
    _muted: iced::Color,
    section: iced::Color,
    tertiary: iced::Color,
    gpu_available: bool,
) -> Element<'a, Message> {
    let intensity_label = blur_intensity_label(settings.blur_strength);
    let backend_label = if gpu_available {
        "Blur backend: GPU accelerated"
    } else {
        "Blur backend: CPU"
    };

    let shape_card = setting_card(
        column![
            row![
                text("Shape").size(scaled(14.0, fs)),
                Space::new().width(12),
                pick_list(
                    BlurShape::ALL,
                    Some(settings.blur_shape),
                    Message::BlurShapeChanged
                )
                .text_size(scaled(13.0, fs)),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text(match settings.blur_shape {
                BlurShape::Ellipse => "Ellipse follows the natural shape of a face.",
                BlurShape::Rect => "Rectangle covers a wider area.",
            })
            .size(scaled(13.0, fs))
            .color(tertiary),
        ]
        .spacing(0),
    );

    let intensity_card = setting_card(
        column![
            row![
                text("Intensity").size(scaled(14.0, fs)),
                Space::new().width(iced::Length::Fill),
                text(intensity_label).size(scaled(13.0, fs)).color(tertiary),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("How heavily faces are blurred.")
                .size(scaled(13.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(
                51..=401,
                settings.blur_strength,
                Message::BlurStrengthChanged
            )
            .step(2u32),
        ]
        .spacing(0),
    );

    let quality_card = setting_card(
        column![
            row![
                text("Output quality").size(scaled(14.0, fs)),
                Space::new().width(iced::Length::Fill),
                text(quality_label(settings.quality))
                    .size(scaled(13.0, fs))
                    .color(tertiary),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("Higher quality produces larger files.")
                .size(scaled(13.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(0..=100, settings.quality, Message::QualityChanged),
            Space::new().height(8),
            text(backend_label).size(scaled(11.0, fs)).color(tertiary),
        ]
        .spacing(0),
    );

    column![
        text("BLUR").size(scaled(11.0, fs)).color(section),
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
) -> Element<'a, Message> {
    let sensitivity_label = sensitivity_label(settings.confidence);

    let sensitivity_card = setting_card(
        column![
            row![
                text("Sensitivity").size(scaled(14.0, fs)),
                Space::new().width(iced::Length::Fill),
                text(sensitivity_label)
                    .size(scaled(13.0, fs))
                    .color(tertiary),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("How certain the detector must be that something is a face.")
                .size(scaled(13.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(10..=100, settings.confidence, Message::ConfidenceChanged),
        ]
        .spacing(0),
    );

    let lookahead_card = setting_card(
        column![
            row![
                text("Lookahead").size(scaled(14.0, fs)),
                Space::new().width(iced::Length::Fill),
                text(format!("{} frames", settings.lookahead))
                    .size(scaled(13.0, fs))
                    .color(tertiary),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(4),
            text("Scan ahead to blur faces before they fully enter the frame.")
                .size(scaled(13.0, fs))
                .color(tertiary),
            Space::new().height(12),
            slider(0..=30, settings.lookahead, Message::LookaheadChanged),
        ]
        .spacing(0),
    );

    column![
        text("DETECTION").size(scaled(11.0, fs)).color(section),
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
) -> Element<'a, Message> {
    let theme_card = setting_card(
        column![
            text("Theme").size(scaled(14.0, fs)),
            Space::new().height(8),
            pick_list(Appearance::ALL, Some(settings.appearance), |a| {
                Message::AppearanceChanged(a)
            })
            .text_size(scaled(13.0, fs)),
            Space::new().height(12),
            checkbox(settings.high_contrast)
                .label("High contrast")
                .on_toggle(Message::HighContrastChanged)
                .text_size(scaled(13.0, fs)),
        ]
        .spacing(0),
    );

    let scale_card = setting_card(
        column![
            row![
                text("Interface scale").size(scaled(14.0, fs)),
                Space::new().width(iced::Length::Fill),
                text(format!("{:.0}%", settings.font_scale * 100.0))
                    .size(scaled(13.0, fs))
                    .color(tertiary),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(12),
            slider(0.8..=1.5, settings.font_scale, Message::FontScaleChanged).step(0.05),
        ]
        .spacing(0),
    );

    column![
        text("APPEARANCE").size(scaled(11.0, fs)).color(section),
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
    format!("{qual} ({:.2})", confidence as f64 / 100.0)
}
