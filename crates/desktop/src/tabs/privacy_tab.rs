use iced::widget::{column, text, Space};
use iced::Element;

use crate::app::Message;

pub fn view<'a>() -> Element<'a, Message> {
    column![
        text("Your data stays on your device").size(16),
        Space::new().height(8),
        text(
            "Video Blur processes everything locally on your computer. \
             Your videos and images are never uploaded to any server.\n\n\
             The only network activity is a one-time download of face \
             detection data when you first launch the app. After that, \
             the app works entirely offline.\n\n\
             No analytics, no tracking, no cloud processing."
        )
        .size(13),
        Space::new().height(20),
        text("Blurring is permanent").size(16),
        Space::new().height(8),
        text(
            "Once a face is blurred, it cannot be unblurred. The original \
             pixels are permanently replaced, so there is no way for anyone \
             to recover the hidden faces from the output file. For this \
             reason, always keep a copy of your original video or image if \
             you might need it later."
        )
        .size(13),
    ]
    .spacing(0)
    .into()
}
