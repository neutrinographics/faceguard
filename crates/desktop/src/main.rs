mod app;
mod tabs;

use app::App;

fn main() -> iced::Result {
    env_logger::init();

    iced::application(App::new, App::update, App::view)
        .title("Video Blur \u{2014} Neutrino Graphics")
        .theme(App::theme)
        .window_size(iced::Size::new(520.0, 360.0))
        .run()
}
