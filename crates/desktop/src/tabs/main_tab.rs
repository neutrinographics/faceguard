use iced::widget::{column, text};
use iced::Element;

use crate::app::Message;

pub fn view<'a>() -> Element<'a, Message> {
    column![text("Blur faces in videos and photos. Select a file to get started.").size(13),]
        .spacing(12)
        .into()
}
