use iced::widget::{column, text};
use iced::Element;

use crate::app::Message;

pub fn view<'a>() -> Element<'a, Message> {
    column![text("Settings").size(16),].spacing(12).into()
}
