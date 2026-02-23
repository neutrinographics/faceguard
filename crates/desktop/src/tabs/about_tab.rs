use iced::widget::{button, column, text, Space};
use iced::Element;

use crate::app::Message;

pub fn view<'a>() -> Element<'a, Message> {
    let version = env!("CARGO_PKG_VERSION");

    column![
        text("Video Blur").size(22),
        Space::new().height(4),
        text(format!("Version {version}")).size(13),
        Space::new().height(4),
        text("Made by Neutrino Graphics LLC").size(13),
        Space::new().height(12),
        text(
            "We partner with mission-driven organizations to build software \
             that works the way you do\u{2014}so you can focus on the work that \
             matters most."
        )
        .size(13),
        Space::new().height(16),
        button(text("Visit neutrinographics.com").size(13))
            .on_press(Message::OpenWebsite)
            .padding([8, 16]),
        Space::new().height(24),
        text("Need custom software?\nWe'd love to hear about your project.").size(13),
    ]
    .spacing(0)
    .into()
}
