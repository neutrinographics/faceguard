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
        .window(iced::window::Settings {
            size: iced::Size::new(620.0, 540.0),
            icon: load_icon(),
            ..Default::default()
        })
        .run()
}

fn load_icon() -> Option<iced::window::Icon> {
    let icon_bytes = include_bytes!("../assets/icon.png");
    let img = image::load_from_memory(icon_bytes).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    iced::window::icon::from_rgba(img.into_raw(), w, h).ok()
}
