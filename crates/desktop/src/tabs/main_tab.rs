use iced::widget::{column, text};
use iced::Element;

use crate::app::{scaled, Message};

pub fn view(fs: f32) -> Element<'static, Message> {
    column![
        text("Blur faces in videos and photos. Select a file to get started.")
            .size(scaled(13.0, fs)),
    ]
    .spacing(12)
    .into()
}
