use std::path::Path;

use iced::widget::{column, container, row, text, Space};
use iced::{Color, Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::{surface_color, tertiary_color};
use crate::widgets::secondary_button;

const CORNER_RADIUS: f32 = 12.0;

pub fn file_row<'a>(
    fs: f32,
    label: &str,
    path: Option<&Path>,
    on_browse: Message,
    hovered: bool,
    on_hover: impl Fn(bool) -> Message + 'a,
    theme: &Theme,
) -> Element<'a, Message> {
    let tertiary = tertiary_color(theme);
    let surface = surface_color(theme);
    let border_base = border_light_color(theme);
    let border_hover = border_color(theme);

    let display_text: Element<'a, Message> = if let Some(name) = path.and_then(|p| p.file_name()) {
        text(name.to_string_lossy().to_string())
            .size(scaled(16.0, fs))
            .font(iced::Font {
                weight: iced::font::Weight::Medium,
                ..iced::Font::DEFAULT
            })
            .into()
    } else {
        text("No file selected")
            .size(scaled(16.0, fs))
            .color(tertiary)
            .into()
    };

    let btn = secondary_button::secondary_button_small(
        move || text("Change").size(scaled(14.0, fs)).into(),
        on_browse,
        hovered,
        on_hover,
        [6, 14],
    );

    let label_text = text(label.to_uppercase())
        .size(scaled(12.0, fs))
        .font(iced::Font {
            weight: iced::font::Weight::Semibold,
            ..iced::Font::DEFAULT
        })
        .color(tertiary);

    let info = column![label_text, Space::new().height(2), display_text].width(Length::Fill);

    let content = row![info, btn].spacing(8).align_y(iced::Alignment::Center);

    container(content)
        .padding([14, 16])
        .width(Length::Fill)
        .style(move |_theme: &Theme| container::Style {
            background: Some(iced::Background::Color(surface)),
            border: iced::border::Border {
                color: if hovered { border_hover } else { border_base },
                width: 1.0,
                radius: CORNER_RADIUS.into(),
            },
            ..container::Style::default()
        })
        .into()
}

/// A very light border color (--border-light in the mockup).
fn border_light_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.10, ..p.text }
}

/// A slightly stronger border color (--border in the mockup).
fn border_color(theme: &Theme) -> Color {
    let p = theme.palette();
    Color { a: 0.18, ..p.text }
}
