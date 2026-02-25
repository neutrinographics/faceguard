use iced::widget::{button, column, container, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::{muted_color, tertiary_color};

pub fn view(fs: f32, theme: &Theme) -> Element<'static, Message> {
    let muted = muted_color(theme);
    let tertiary = tertiary_color(theme);
    let version = env!("CARGO_PKG_VERSION");

    let header = column![
        text("Video Blur").size(scaled(26.0, fs)),
        Space::new().height(4),
        text(format!("Version {version}"))
            .size(scaled(13.0, fs))
            .color(tertiary),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC")
            .size(scaled(14.0, fs))
            .color(muted),
        Space::new().height(20),
        text(
            "We help ministries and mission-driven organizations advance the \
             gospel by building innovative software platforms that improve how \
             they operate, disciple, and measure impact."
        )
        .size(scaled(15.0, fs))
        .color(muted),
        Space::new().height(24),
        button(text("Visit neutrinographics.com").size(scaled(14.0, fs)))
            .on_press(Message::OpenWebsite)
            .padding([10, 20]),
    ]
    .align_x(iced::Alignment::Center);

    let privacy_card = container(
        column![
            text("Your data stays on your device").size(scaled(14.0, fs)),
            Space::new().height(8),
            text(
                "Video Blur processes everything locally on your computer. \
                 Your videos and images are never uploaded to any server."
            )
            .size(scaled(13.0, fs))
            .color(muted),
            Space::new().height(8),
            text(
                "The only network activity is a one-time download of face \
                 detection data when you first launch the app. After that, \
                 the app works entirely offline."
            )
            .size(scaled(13.0, fs))
            .color(muted),
            Space::new().height(8),
            text("No analytics. No tracking. No cloud processing.")
                .size(scaled(13.0, fs))
                .color(muted),
        ]
        .spacing(0),
    )
    .padding(20)
    .style(container::rounded_box)
    .width(Length::Fill);

    let warning_text = column![
        text("\u{26A0} Blurring is permanent").size(scaled(13.0, fs)),
        Space::new().height(4),
        text(
            "Once a face is blurred, it cannot be undone \u{2014} the original \
             pixels are replaced. Always keep a copy of your original file."
        )
        .size(scaled(13.0, fs))
        .color(muted),
    ]
    .spacing(0);

    let cta_card = container(
        text("Need custom software? We\u{2019}d love to hear about your project.")
            .size(scaled(14.0, fs))
            .color(muted)
            .align_x(iced::Alignment::Center),
    )
    .padding(18)
    .style(container::rounded_box)
    .width(Length::Fill)
    .center_x(Length::Fill);

    column![
        header,
        Space::new().height(24),
        privacy_card,
        Space::new().height(12),
        warning_text,
        Space::new().height(24),
        cta_card,
    ]
    .align_x(iced::Alignment::Center)
    .spacing(0)
    .into()
}
