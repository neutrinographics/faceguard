use iced::widget::{column, container, mouse_area, row, text, Space};
use iced::{Color, Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::surface_color;
use crate::widgets::dashed_container::{dashed_container, DashedBorderStyle};
use crate::widgets::primary_button;

pub fn view(
    fs: f32,
    tertiary: Color,
    theme: &Theme,
    browse_hovered: bool,
    drop_zone_hovered: bool,
) -> Element<'static, Message> {
    let palette = theme.extended_palette();
    let accent = palette.primary.base.color;

    let base_style = DashedBorderStyle {
        border_color: Color { a: 0.20, ..palette.background.base.text },
        border_width: 2.0,
        dash_length: 3.0,
        gap_length: 3.0,
        corner_radius: 16.0,
        background: surface_color(theme),
    };

    let hover_style = DashedBorderStyle {
        border_color: Color { a: 0.50, ..accent },
        background: Color { a: 0.06, ..accent },
        ..base_style
    };

    let inner_content = build_inner(fs, tertiary, accent, browse_hovered);

    let drop_zone = dashed_container(
        base_style,
        [scaled(56.0, fs) as u16, 40],
        inner_content,
    )
    .hover_style(hover_style, drop_zone_hovered);

    let wrapped = container(drop_zone)
        .width(Length::Fill)
        .height(Length::Fill)
        .center_y(Length::Fill);

    mouse_area(wrapped)
        .on_enter(Message::DropZoneHover(true))
        .on_exit(Message::DropZoneHover(false))
        .into()
}

fn build_inner(
    fs: f32,
    tertiary: Color,
    accent: Color,
    browse_hovered: bool,
) -> Element<'static, Message> {
    let icon_circle = container(
        text("\u{2B06}")
            .size(scaled(22.0, fs))
            .align_x(iced::Alignment::Center)
            .align_y(iced::Alignment::Center),
    )
    .width(scaled(56.0, fs))
    .height(scaled(56.0, fs))
    .center_x(Length::Shrink)
    .center_y(Length::Shrink)
    .style(move |_theme: &Theme| {
        container::Style {
            background: Some(iced::Background::Color(Color { a: 0.12, ..accent })),
            border: iced::border::Border {
                radius: 100.0.into(),
                ..iced::border::Border::default()
            },
            text_color: Some(accent),
            ..container::Style::default()
        }
    });

    let browse_btn = primary_button::primary_button(
        move || {
            row![
                text("\u{1F4C2}").size(scaled(14.0, fs)),
                text("Browse Files")
                    .size(scaled(14.0, fs))
                    .color(Color::WHITE)
                    .font(iced::Font {
                        weight: iced::font::Weight::Bold,
                        ..iced::Font::DEFAULT
                    }),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center)
            .into()
        },
        Message::SelectInput,
        browse_hovered,
        Message::BrowseHover,
        [10, 24],
    );

    column![
        icon_circle,
        Space::new().height(16),
        text("Drop a file here to get started")
            .size(scaled(17.0, fs))
            .font(iced::Font {
                weight: iced::font::Weight::Bold,
                ..iced::Font::DEFAULT
            }),
        Space::new().height(6),
        text("or click to browse your computer")
            .size(scaled(14.0, fs))
            .color(tertiary),
        Space::new().height(20),
        browse_btn,
        Space::new().height(16),
        text("MP4, MOV, AVI, JPG, PNG")
            .size(scaled(12.0, fs))
            .color(tertiary),
    ]
    .align_x(iced::Alignment::Center)
    .into()
}
