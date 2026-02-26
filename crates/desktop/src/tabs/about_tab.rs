use iced::widget::{column, container, row, svg, text, Space};
use iced::{Color, Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::{muted_color, surface_color, tertiary_color};
use crate::widgets::primary_button;

pub fn view(fs: f32, theme: &Theme, website_hovered: bool) -> Element<'static, Message> {
    let muted = muted_color(theme);
    let tertiary = tertiary_color(theme);
    let surface = surface_color(theme);
    let palette = theme.palette();
    let accent = palette.primary;
    let success = palette.success;
    let warning = palette.warning;
    let border_color = Color { a: 0.15, ..palette.text };
    let version = env!("CARGO_PKG_VERSION");

    let semibold = iced::Font {
        weight: iced::font::Weight::Semibold,
        ..iced::Font::DEFAULT
    };

    // --- Logo icon ---
    let accent_bg = Color { a: 0.12, ..accent };
    let logo_svg = svg(svg::Handle::from_memory(include_bytes!(
        "../../assets/logo.svg"
    )))
    .width(28)
    .height(28)
    .style(move |_, _| svg::Style {
        color: Some(accent),
    });
    let logo_box = container(logo_svg)
        .width(56)
        .height(56)
        .center_x(56)
        .center_y(56)
        .style(move |_theme: &Theme| container::Style {
            background: Some(accent_bg.into()),
            border: iced::border::Border {
                radius: 12.0.into(),
                ..iced::border::Border::default()
            },
            ..container::Style::default()
        });

    // --- Header ---
    let header = column![
        logo_box,
        Space::new().height(16),
        text("Video Blur")
            .size(scaled(26.0, fs))
            .font(semibold),
        Space::new().height(4),
        text(format!("Version {version}"))
            .size(scaled(13.0, fs))
            .color(tertiary),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC")
            .size(scaled(14.0, fs))
            .font(semibold)
            .color(muted),
        Space::new().height(20),
        text(
            "We help ministries and mission-driven organizations advance the \
             gospel by building innovative software platforms that improve how \
             they operate, disciple, and measure impact.",
        )
        .size(scaled(15.0, fs))
        .color(muted),
        Space::new().height(24),
        website_button(fs, website_hovered),
    ]
    .align_x(iced::Alignment::Center);

    // --- Privacy card ---
    let lock_icon = svg(svg::Handle::from_memory(include_bytes!(
        "../../assets/lock.svg"
    )))
    .width(16)
    .height(16)
    .style(move |_, _| svg::Style {
        color: Some(success),
    });

    let privacy_card = styled_card(
        column![
            row![
                lock_icon,
                Space::new().width(8),
                text("Your data stays on your device")
                    .size(scaled(14.0, fs))
                    .font(semibold),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(8),
            text(
                "Video Blur processes everything locally on your computer. \
                 Your videos and images are never uploaded to any server.",
            )
            .size(scaled(13.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "The only network activity is a one-time download of face \
                 detection data when you first launch the app. After that, \
                 the app works entirely offline.",
            )
            .size(scaled(13.0, fs))
            .color(muted),
            Space::new().height(8),
            text("No analytics. No tracking. No cloud processing.")
                .size(scaled(13.0, fs))
                .font(semibold)
                .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    // --- Warning card ---
    let warning_bg = Color {
        r: 1.0,
        g: 0.973,
        b: 0.906,
        a: 1.0,
    };
    let warning_border = Color {
        r: 0.941,
        g: 0.875,
        b: 0.627,
        a: 1.0,
    };
    let warning_text_color = Color {
        r: 0.478,
        g: 0.396,
        b: 0.125,
        a: 1.0,
    };

    let warning_card = styled_card(
        column![
            text("\u{26A0} Blurring is permanent")
                .size(scaled(13.0, fs))
                .font(semibold)
                .color(warning),
            Space::new().height(4),
            text(
                "Once a face is blurred, it cannot be undone \u{2014} the original \
                 pixels are replaced. Always keep a copy of your original file.",
            )
            .size(scaled(13.0, fs))
            .color(warning_text_color),
        ]
        .spacing(0),
        warning_bg,
        warning_border,
    );

    // --- CTA card ---
    let accent_soft = Color { a: 0.06, ..accent };
    let accent_border = Color { a: 0.15, ..accent };

    let cta_card = container(
        row![
            text("Need custom software?")
                .size(scaled(14.0, fs))
                .font(semibold),
            text(" We\u{2019}d love to hear about your project.")
                .size(scaled(14.0, fs))
                .color(muted),
        ],
    )
    .padding(18)
    .width(Length::Fill)
    .center_x(Length::Fill)
    .style(move |_theme: &Theme| container::Style {
        background: Some(accent_soft.into()),
        border: iced::border::Border {
            color: accent_border,
            width: 1.0,
            radius: 12.0.into(),
        },
        ..container::Style::default()
    });

    column![
        header,
        Space::new().height(24),
        privacy_card,
        Space::new().height(12),
        warning_card,
        Space::new().height(24),
        cta_card,
    ]
    .align_x(iced::Alignment::Center)
    .spacing(0)
    .into()
}

fn website_button(fs: f32, hovered: bool) -> Element<'static, Message> {
    primary_button::primary_button(
        move || {
            let link_icon = svg(svg::Handle::from_memory(include_bytes!(
                "../../assets/external-link.svg"
            )))
            .width(14)
            .height(14)
            .style(|_, _| svg::Style {
                color: Some(Color::WHITE),
            });

            row![
                text("Visit neutrinographics.com")
                    .size(scaled(14.0, fs))
                    .font(iced::Font {
                        weight: iced::font::Weight::Semibold,
                        ..iced::Font::DEFAULT
                    }),
                Space::new().width(6),
                link_icon,
            ]
            .align_y(iced::Alignment::Center)
            .into()
        },
        Message::OpenWebsite,
        hovered,
        Message::WebsiteHover,
        [10, 20],
    )
}

fn styled_card<'a>(
    content: impl Into<Element<'a, Message>>,
    bg: Color,
    border_color: Color,
) -> Element<'a, Message> {
    container(content)
        .padding(20)
        .width(Length::Fill)
        .style(move |_theme: &Theme| container::Style {
            background: Some(bg.into()),
            border: iced::border::Border {
                color: border_color,
                width: 1.0,
                radius: 12.0.into(),
            },
            ..container::Style::default()
        })
        .into()
}
