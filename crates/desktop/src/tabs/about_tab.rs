use iced::widget::{button, column, rule, text, Space};
use iced::{Element, Theme};

use crate::app::{scaled, Message};
use crate::theme::{muted_color, section_color};

pub fn view(fs: f32, theme: &Theme) -> Element<'static, Message> {
    let muted = muted_color(theme);
    let section = section_color(theme);
    let version = env!("CARGO_PKG_VERSION");

    column![
        text("Video Blur").size(scaled(20.0, fs)),
        Space::new().height(4),
        text(format!("Version {version}")).size(scaled(12.0, fs)),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC").size(scaled(14.0, fs)),
        Space::new().height(16),
        text(
            "We help ministries and mission-driven organizations advance the \
             gospel by building innovative software platforms that improve how \
             they operate, disciple, and measure impact."
        )
        .size(scaled(14.0, fs)),
        Space::new().height(16),
        button(text("Visit neutrinographics.com").size(scaled(13.0, fs)))
            .on_press(Message::OpenWebsite)
            .padding([8, 16])
            .style(button::secondary),
        Space::new().height(20),
        rule::horizontal(1),
        Space::new().height(20),
        text("PRIVACY").size(scaled(12.0, fs)).color(section),
        Space::new().height(12),
        text(
            "Video Blur processes everything locally on your computer. \
             Your videos and images are never uploaded to any server.\n\n\
             The only network activity is a one-time download of face \
             detection data when you first launch the app. After that, \
             the app works entirely offline.\n\n\
             No analytics, no tracking, no cloud processing."
        )
        .size(scaled(14.0, fs)),
        Space::new().height(16),
        text(
            "\u{26A0} Blurring is permanent \u{2014} once a face is blurred, \
             it cannot be unblurred. The original pixels are permanently \
             replaced, so there is no way for anyone to recover the hidden \
             faces from the output file. Always keep a copy of your original \
             video or image if you might need it later."
        )
        .size(scaled(14.0, fs))
        .color(muted),
    ]
    .spacing(0)
    .into()
}
