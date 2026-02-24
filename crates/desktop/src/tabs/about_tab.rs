use iced::widget::{button, column, text, Space};
use iced::Element;

use crate::app::{scaled, Message};

pub fn view(fs: f32) -> Element<'static, Message> {
    let version = env!("CARGO_PKG_VERSION");

    column![
        text("Video Blur").size(scaled(22.0, fs)),
        Space::new().height(4),
        text(format!("Version {version}")).size(scaled(13.0, fs)),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC").size(scaled(13.0, fs)),
        Space::new().height(12),
        text(
            "We partner with mission-driven organizations to build software \
             that works the way you do\u{2014}so you can focus on the work that \
             matters most."
        )
        .size(scaled(13.0, fs)),
        Space::new().height(16),
        button(text("Visit neutrinographics.com").size(scaled(13.0, fs)))
            .on_press(Message::OpenWebsite)
            .padding([8, 16]),
        Space::new().height(24),
        text("Need custom software?\nWe'd love to hear about your project.").size(scaled(13.0, fs)),
    ]
    .spacing(0)
    .into()
}
