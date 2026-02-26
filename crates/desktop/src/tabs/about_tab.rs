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
    let border_color = Color {
        a: 0.15,
        ..palette.text
    };
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
        text("FaceGuard").size(scaled(28.0, fs)).font(semibold),
        Space::new().height(4),
        text(format!("Version {version}"))
            .size(scaled(14.0, fs))
            .color(tertiary),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC")
            .size(scaled(15.0, fs))
            .font(semibold)
            .color(muted),
    ]
    .width(Length::Fill)
    .align_x(iced::Alignment::Center);

    // --- Section 1: Why we built this ---
    let why_section = column![styled_card(
        column![
            text("Why we built this")
                .size(scaled(15.0, fs))
                .font(semibold),
            Space::new().height(8),
            text(
                "In many parts of the world, sharing photos and videos from meaningful \
                     work can put people at risk.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "Even when content is shared privately with supporters, it\u{2019}s easy for \
                     faces, backgrounds, and small details to expose someone\u{2019}s identity in \
                     ways you never intended.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "We built this app to make protection simple. Actually, we built this for \
                     ourselves, and realized that it could help a lot of other people, too.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "So upload a video or photo, choose who should be anonymized, and export a \
                     version you can share with confidence \u{2014} without spending hours editing \
                     or taking unnecessary risks.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    ),];

    // --- Section 2: Privacy and how it works ---
    let lock_icon = svg(svg::Handle::from_memory(include_bytes!(
        "../../assets/lock.svg"
    )))
    .width(16)
    .height(16)
    .style(move |_, _| svg::Style {
        color: Some(palette.success),
    });

    let privacy_card = styled_card(
        column![
            row![
                lock_icon,
                Space::new().width(8),
                text("Privacy comes first")
                    .size(scaled(14.0, fs))
                    .font(semibold),
            ]
            .align_y(iced::Alignment::Center),
            Space::new().height(8),
            text(
                "This app is designed for sensitive environments. Your content should \
                 stay in your control.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text("All processing happens locally on your device.")
                .size(scaled(15.0, fs))
                .font(semibold)
                .color(muted),
            text("Your photos and videos are never uploaded to a server.",)
                .size(scaled(14.0, fs))
                .color(muted),
            Space::new().height(8),
            text(
                "After a one-time download of the face detection library, the app works \
                 entirely offline.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text("No analytics. No tracking. No cloud processing.")
                .size(scaled(15.0, fs))
                .font(semibold)
                .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    let data_card = styled_card(
        column![
            text("How your data is handled")
                .size(scaled(15.0, fs))
                .font(semibold),
            Space::new().height(8),
            bullet_point(
                "Your original files remain unchanged unless you choose to delete them",
                fs,
                muted,
            ),
            Space::new().height(4),
            bullet_point(
                "The app creates a new file with the edits applied",
                fs,
                muted,
            ),
            Space::new().height(4),
            bullet_point(
                "You decide which faces to blur and how strong the effect is",
                fs,
                muted,
            ),
            Space::new().height(4),
            bullet_point(
                "Exported files can be shared safely without revealing identities",
                fs,
                muted,
            ),
            Space::new().height(4),
            bullet_point(
                "Metadata such as GPS location, camera details, and timestamps is removed from exported files",
                fs,
                muted,
            ),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    // --- Important note card ---
    let important_card = styled_card(
        column![
            text("Blurring is permanent")
                .size(scaled(15.0, fs))
                .font(semibold),
            Space::new().height(8),
            text(
                "Once a face has been blurred, the original pixels are replaced and cannot \
                 be recovered. There is no way to reverse the edit or restore hidden faces \
                 from the exported file.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "If you may need the original media later, keep a separate copy before exporting.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    let safeguard_card = styled_card(
        column![
            text("Final safeguard")
                .size(scaled(15.0, fs))
                .font(semibold),
            Space::new().height(8),
            text("This tool is designed to help reduce risk, not eliminate it entirely.",)
                .size(scaled(14.0, fs))
                .color(muted),
            Space::new().height(8),
            text(
                "Always review your final file before sharing, especially in sensitive situations.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    let privacy_section = column![
        section_label("PRIVACY & HOW IT WORKS", fs, tertiary),
        Space::new().height(14),
        privacy_card,
        Space::new().height(10),
        data_card,
        Space::new().height(14),
        section_label("IMPORTANT NOTE", fs, tertiary),
        Space::new().height(14),
        important_card,
        Space::new().height(10),
        safeguard_card,
    ];

    // --- Section 3: About Neutrino Graphics ---
    let accent_soft = Color { a: 0.06, ..accent };
    let accent_border_color = Color { a: 0.15, ..accent };

    let about_ng_card = styled_card(
        column![
            text(
                "This app was built by a small software team that works closely with \
                 ministries and mission-driven organizations.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "We understand the challenges of operating in sensitive environments \u{2014} \
                 balancing impact, communication, and security while trying to serve people well.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "Most of our work focuses on building practical, custom tools that help \
                 teams operate more effectively in the field.",
            )
            .size(scaled(14.0, fs))
            .color(muted),
        ]
        .spacing(0),
        surface,
        border_color,
    );

    let cta_card = container(
        column![
            text("If your organization is facing a challenge that doesn\u{2019}t have a simple off-the-shelf solution, we\u{2019}d be glad to hear about it.")
                .size(scaled(15.0, fs))
                .color(muted)
                .align_x(iced::Alignment::Center),
            Space::new().height(16),
            website_button(fs, website_hovered),
        ]
        .align_x(iced::Alignment::Center),
    )
    .padding(18)
    .width(Length::Fill)
    .center_x(Length::Fill)
    .style(move |_theme: &Theme| container::Style {
        background: Some(accent_soft.into()),
        border: iced::border::Border {
            color: accent_border_color,
            width: 1.0,
            radius: 12.0.into(),
        },
        ..container::Style::default()
    });

    let ng_section = column![
        section_label("ABOUT NEUTRINO GRAPHICS", fs, tertiary),
        Space::new().height(14),
        about_ng_card,
        Space::new().height(10),
        cta_card,
    ];

    column![
        header,
        Space::new().height(28),
        why_section,
        Space::new().height(28),
        privacy_section,
        Space::new().height(28),
        ng_section,
    ]
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
                text("Visit our website")
                    .size(scaled(15.0, fs))
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

fn section_label<'a>(label: &str, fs: f32, color: Color) -> iced::widget::Text<'a> {
    text(label.to_string())
        .size(scaled(12.0, fs))
        .color(color)
        .font(iced::Font {
            weight: iced::font::Weight::Bold,
            ..iced::Font::DEFAULT
        })
}

fn bullet_point<'a>(content: &str, fs: f32, color: Color) -> Element<'a, Message> {
    row![
        text("\u{2022} ").size(scaled(14.0, fs)).color(color),
        text(content.to_string())
            .size(scaled(14.0, fs))
            .color(color),
    ]
    .into()
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
