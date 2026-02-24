mod app;
mod settings;
mod tabs;
mod theme;
mod widgets;
mod workers;

use app::App;

fn main() -> iced::Result {
    env_logger::init();

    iced::application(App::new, App::update, App::view)
        .title("Video Blur \u{2014} Neutrino Graphics")
        .theme(App::theme)
        .subscription(App::subscription)
        .window_size(iced::Size::new(520.0, 360.0))
        .run()
}
