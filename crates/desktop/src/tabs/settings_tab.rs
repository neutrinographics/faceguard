use iced::widget::{column, text};
use iced::Element;

use crate::app::{scaled, Message};

pub fn view(fs: f32) -> Element<'static, Message> {
    column![text("Settings").size(scaled(16.0, fs)),]
        .spacing(12)
        .into()
}
