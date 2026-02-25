use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::Receiver;
use iced::widget::{button, column, container, mouse_area, operation, row, scrollable, text, Space};
use iced::{Color, Element, Length, Subscription, Task, Theme};
use iced_anim::AnimationBuilder;
use iced_anim::transition::Easing;

use crate::settings::{Appearance, BlurShape, Settings};
use crate::tabs;
use crate::theme;
use crate::widgets::faces_well::FacesWellState;
use crate::workers::blur_worker::{self, BlurParams, WorkerMessage};
use crate::workers::model_cache::ModelCache;
use crate::workers::preview_worker::{self, PreviewMessage, PreviewParams};
use video_blur_core::blurring::infrastructure::blurrer_factory;
use video_blur_core::blurring::infrastructure::gpu_context::GpuContext;
use video_blur_core::shared::region::Region;

const WEBSITE_URL: &str = "https://www.neutrinographics.com/";
const SCROLL_ID: &str = "tab-scroll";

pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png", "bmp", "tiff", "webp",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Blur,
    Settings,
    About,
}

impl Tab {
    const ALL: &[Tab] = &[Tab::Blur, Tab::Settings, Tab::About];

    fn label(self) -> &'static str {
        match self {
            Tab::Blur => "Blur",
            Tab::Settings => "Settings",
            Tab::About => "About",
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProcessingState {
    Idle,
    Preparing,
    Downloading(u64, u64),
    Scanning(usize, usize),
    Previewed,
    Blurring(usize, usize),
    Complete,
    Error(String),
}

#[derive(Debug, Clone)]
pub enum Message {
    TabSelected(Tab),
    OpenWebsite,
    SelectInput,
    InputSelected(Option<PathBuf>),
    SelectOutput,
    OutputSelected(Option<PathBuf>),
    RunPreview,
    RunBlur,
    CancelWork,
    WorkerTick,
    ShowInFolder,
    StartOver,
    ToggleFace(u32),
    ToggleGroup(usize),
    GroupFacesToggled(bool),
    BlurShapeChanged(BlurShape),
    ConfidenceChanged(u32),
    BlurStrengthChanged(u32),
    LookaheadChanged(u32),
    RestoreDefaults,
    AppearanceChanged(Appearance),
    HighContrastChanged(bool),
    QualityChanged(u32),
    FontScaleChanged(f32),
    PollSystemTheme,
    FileDropped(PathBuf),
    TabHover(usize, bool),
    BrowseHover(bool),
    DropZoneHover(bool),
    BlurButtonHover(bool),
    ChangeInputHover(bool),
    ChangeOutputHover(bool),
    ChooseFacesHover(bool),
    CancelHover(bool),
    RescanHover(bool),
}

pub struct App {
    active_tab: Tab,
    pub settings: Settings,
    pub input_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub processing: ProcessingState,
    pub faces_well: FacesWellState,
    detection_cache: Option<Arc<HashMap<usize, Vec<Region>>>>,
    gpu_context: Option<Arc<GpuContext>>,
    model_cache: Arc<ModelCache>,
    preview_rx: Option<Receiver<PreviewMessage>>,
    worker_rx: Option<Receiver<WorkerMessage>>,
    worker_cancel: Option<Arc<AtomicBool>>,
    tab_hovered: [bool; 3],
    pub browse_hovered: bool,
    pub drop_zone_hovered: bool,
    pub blur_button_hovered: bool,
    pub change_input_hovered: bool,
    pub change_output_hovered: bool,
    pub choose_faces_hovered: bool,
    pub cancel_hovered: bool,
    pub rescan_hovered: bool,
}

impl App {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                active_tab: Tab::Blur,
                settings: Settings::load(),
                input_path: None,
                output_path: None,
                processing: ProcessingState::Idle,
                faces_well: FacesWellState::new(),
                detection_cache: None,
                gpu_context: blurrer_factory::create_gpu_context(),
                model_cache: ModelCache::new(),
                preview_rx: None,
                worker_rx: None,
                worker_cancel: None,
                tab_hovered: [false; 3],
                browse_hovered: false,
                drop_zone_hovered: false,
                blur_button_hovered: false,
                change_input_hovered: false,
                change_output_hovered: false,
                choose_faces_hovered: false,
                cancel_hovered: false,
                rescan_hovered: false,
            },
            Task::none(),
        )
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::TabSelected(tab) => {
                self.active_tab = tab;
                return operation::snap_to(SCROLL_ID, operation::RelativeOffset::START);
            }
            Message::OpenWebsite => {
                let _ = open::that(WEBSITE_URL);
            }
            Message::SelectInput => return self.pick_input_file(),
            Message::InputSelected(Some(path)) => self.set_input(path),
            Message::InputSelected(None) => {}
            Message::SelectOutput => return self.pick_output_file(),
            Message::OutputSelected(Some(path)) => {
                self.output_path = Some(path);
            }
            Message::OutputSelected(None) => {}
            Message::RunPreview => self.start_preview(),
            Message::RunBlur => self.start_blur(),
            Message::CancelWork => {
                if let Some(ref cancel) = self.worker_cancel {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            Message::WorkerTick => {
                self.drain_preview_messages();
                self.drain_blur_messages();
            }
            Message::ToggleFace(track_id) => {
                self.faces_well.toggle_face(track_id);
            }
            Message::ToggleGroup(group_idx) => {
                self.faces_well.toggle_group(group_idx);
            }
            Message::GroupFacesToggled(enabled) => {
                self.faces_well.group_faces = enabled;
            }
            Message::ShowInFolder => {
                if let Some(ref output) = self.output_path {
                    if let Some(parent) = output.parent() {
                        let _ = open::that(parent);
                    }
                }
            }
            Message::StartOver => self.reset(),
            Message::BlurShapeChanged(shape) => {
                self.settings.blur_shape = shape;
                self.settings.save();
            }
            Message::ConfidenceChanged(val) => {
                self.settings.confidence = val;
                self.settings.save();
                self.invalidate_detection();
            }
            Message::BlurStrengthChanged(val) => {
                self.settings.blur_strength = if val % 2 == 0 { val + 1 } else { val };
                self.settings.save();
            }
            Message::LookaheadChanged(val) => {
                self.settings.lookahead = val;
                self.settings.save();
            }
            Message::QualityChanged(val) => {
                self.settings.quality = val;
                self.settings.save();
            }
            Message::RestoreDefaults => self.restore_defaults(),
            Message::AppearanceChanged(appearance) => {
                self.settings.appearance = appearance;
                self.settings.save();
            }
            Message::HighContrastChanged(enabled) => {
                self.settings.high_contrast = enabled;
                self.settings.save();
            }
            Message::FontScaleChanged(scale) => {
                self.settings.font_scale = scale;
                self.settings.save();
            }
            Message::PollSystemTheme => {}
            Message::FileDropped(path) => {
                if has_supported_extension(&path) {
                    return self.update(Message::InputSelected(Some(path)));
                }
            }
            Message::TabHover(idx, hovered) => {
                if idx < self.tab_hovered.len() {
                    self.tab_hovered[idx] = hovered;
                }
            }
            Message::BrowseHover(hovered) => {
                self.browse_hovered = hovered;
            }
            Message::DropZoneHover(hovered) => {
                self.drop_zone_hovered = hovered;
            }
            Message::BlurButtonHover(hovered) => {
                self.blur_button_hovered = hovered;
            }
            Message::ChangeInputHover(hovered) => {
                self.change_input_hovered = hovered;
            }
            Message::ChangeOutputHover(hovered) => {
                self.change_output_hovered = hovered;
            }
            Message::ChooseFacesHover(hovered) => {
                self.choose_faces_hovered = hovered;
            }
            Message::CancelHover(hovered) => {
                self.cancel_hovered = hovered;
            }
            Message::RescanHover(hovered) => {
                self.rescan_hovered = hovered;
            }
        }
        Task::none()
    }

    pub fn view(&self) -> Element<'_, Message> {
        let fs = self.settings.font_scale;
        let current_theme = self.theme();
        let palette = current_theme.palette();

        let surface = theme::surface_color(&current_theme);
        let border_light = iced::Color {
            a: 0.12,
            ..palette.text
        };

        let tab_row = container(
            row(Tab::ALL
                .iter()
                .enumerate()
                .map(|(idx, &tab)| {
                    tab_button(tab, tab == self.active_tab, self.tab_hovered[idx], idx, palette, fs)
                })
                .collect::<Vec<_>>())
            .spacing(4),
        )
        .width(Length::Fill)
        .padding([0, 20])
        .style(move |_theme: &Theme| container::Style {
            background: Some(surface.into()),
            ..container::Style::default()
        });

        let tab_divider = container(Space::new().height(0))
            .width(Length::Fill)
            .height(1)
            .style(move |_theme: &Theme| container::Style {
                background: Some(border_light.into()),
                ..container::Style::default()
            });

        let tab_bar = column![tab_row, tab_divider].spacing(0);

        let content: Element<'_, Message> = match self.active_tab {
            Tab::Blur => tabs::main_tab::view(
                fs,
                self.input_path.as_deref(),
                self.output_path.as_deref(),
                &self.processing,
                &self.faces_well,
                &current_theme,
                self.browse_hovered,
                self.drop_zone_hovered,
                self.blur_button_hovered,
                self.change_input_hovered,
                self.change_output_hovered,
                self.choose_faces_hovered,
                self.cancel_hovered,
                self.rescan_hovered,
            ),
            Tab::Settings => tabs::settings_tab::view(&self.settings, self.gpu_context.is_some()),
            Tab::About => tabs::about_tab::view(fs, &current_theme),
        };

        let tab_content = container(
            scrollable(content)
                .id(iced::widget::Id::new(SCROLL_ID))
                .spacing(4)
                .height(Length::Fill),
        )
        .padding(24)
        .height(Length::Fill);

        column![tab_bar, tab_content]
            .spacing(0)
            .height(Length::Fill)
            .into()
    }

    pub fn theme(&self) -> Theme {
        theme::resolve_theme(self.settings.appearance, self.settings.high_contrast)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let mut subs = vec![];

        if self.settings.appearance == Appearance::System {
            subs.push(iced::time::every(Duration::from_secs(2)).map(|_| Message::PollSystemTheme));
        }

        if self.worker_rx.is_some() || self.preview_rx.is_some() {
            subs.push(iced::time::every(Duration::from_millis(50)).map(|_| Message::WorkerTick));
        }

        subs.push(iced::event::listen_with(|event, _status, _id| {
            if let iced::Event::Window(iced::window::Event::FileDropped(path)) = event {
                Some(Message::FileDropped(path))
            } else {
                None
            }
        }));

        Subscription::batch(subs)
    }
}

// --- Private helpers (lower-level details) ---

impl App {
    fn pick_input_file(&self) -> Task<Message> {
        Task::perform(
            async {
                rfd::AsyncFileDialog::new()
                    .set_title("Select input file")
                    .add_filter("Media Files", SUPPORTED_EXTENSIONS)
                    .pick_file()
                    .await
                    .map(|h| h.path().to_path_buf())
            },
            Message::InputSelected,
        )
    }

    fn set_input(&mut self, path: PathBuf) {
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let ext = path
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default();
        self.output_path = Some(path.with_file_name(format!("{stem}_blurred{ext}")));
        self.input_path = Some(path);
        self.processing = ProcessingState::Idle;
        self.faces_well.clear();
        self.detection_cache = None;
    }

    fn pick_output_file(&self) -> Task<Message> {
        let start_dir = self
            .output_path
            .as_ref()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));
        let start_name = self
            .output_path
            .as_ref()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()));

        Task::perform(
            async move {
                let mut dialog = rfd::AsyncFileDialog::new()
                    .set_title("Save output as")
                    .add_filter("Media Files", SUPPORTED_EXTENSIONS);
                if let Some(dir) = start_dir {
                    dialog = dialog.set_directory(dir);
                }
                if let Some(name) = start_name {
                    dialog = dialog.set_file_name(name);
                }
                dialog.save_file().await.map(|h| h.path().to_path_buf())
            },
            Message::OutputSelected,
        )
    }

    fn start_preview(&mut self) {
        if let Some(input) = self.input_path.clone() {
            let params = PreviewParams {
                input_path: input,
                confidence: self.settings.confidence,
                model_cache: self.model_cache.clone(),
            };
            let (rx, cancel) = preview_worker::spawn(params);
            self.preview_rx = Some(rx);
            self.worker_cancel = Some(cancel);
            self.processing = ProcessingState::Preparing;
        }
    }

    fn start_blur(&mut self) {
        if let (Some(input), Some(output)) = (self.input_path.clone(), self.output_path.clone()) {
            let params = BlurParams {
                input_path: input,
                output_path: output,
                blur_shape: self.settings.blur_shape,
                confidence: self.settings.confidence,
                blur_strength: self.settings.blur_strength,
                lookahead: self.settings.lookahead,
                quality: self.settings.quality,
                detection_cache: self.detection_cache.clone(),
                blur_ids: self.faces_well.get_selected_ids(),
                model_cache: self.model_cache.clone(),
                gpu_context: self.gpu_context.clone(),
            };
            let (rx, cancel) = blur_worker::spawn(params);
            self.worker_rx = Some(rx);
            self.worker_cancel = Some(cancel);
            self.processing = ProcessingState::Preparing;
        }
    }

    fn drain_preview_messages(&mut self) {
        let messages: Vec<_> = self
            .preview_rx
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();

        for msg in messages {
            match msg {
                PreviewMessage::DownloadProgress(dl, total) => {
                    self.processing = ProcessingState::Downloading(dl, total);
                }
                PreviewMessage::ScanProgress(current, total) => {
                    self.processing = ProcessingState::Scanning(current, total);
                }
                PreviewMessage::Complete(result) => {
                    self.faces_well
                        .populate(result.crops, result.groups, result.temp_dir);
                    self.detection_cache = Some(Arc::new(result.detection_cache));
                    self.processing = ProcessingState::Previewed;
                    self.clear_worker_state(true);
                }
                PreviewMessage::Error(e) => {
                    self.processing = ProcessingState::Error(e);
                    self.clear_worker_state(true);
                }
                PreviewMessage::Cancelled => {
                    self.processing = ProcessingState::Idle;
                    self.clear_worker_state(true);
                }
            }
        }
    }

    fn drain_blur_messages(&mut self) {
        let messages: Vec<_> = self
            .worker_rx
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();

        for msg in messages {
            match msg {
                WorkerMessage::DownloadProgress(dl, total) => {
                    self.processing = ProcessingState::Downloading(dl, total);
                }
                WorkerMessage::BlurProgress(current, total) => {
                    self.processing = ProcessingState::Blurring(current, total);
                }
                WorkerMessage::Complete => {
                    self.processing = ProcessingState::Complete;
                    self.clear_worker_state(false);
                }
                WorkerMessage::Error(e) => {
                    self.processing = ProcessingState::Error(e);
                    self.clear_worker_state(false);
                }
                WorkerMessage::Cancelled => {
                    self.processing = ProcessingState::Idle;
                    self.clear_worker_state(false);
                }
            }
        }
    }

    fn clear_worker_state(&mut self, is_preview: bool) {
        if is_preview {
            self.preview_rx = None;
        } else {
            self.worker_rx = None;
        }
        self.worker_cancel = None;
    }

    fn invalidate_detection(&mut self) {
        if self.detection_cache.is_some() {
            self.detection_cache = None;
            self.faces_well.clear();
            if matches!(self.processing, ProcessingState::Previewed) {
                self.processing = ProcessingState::Idle;
            }
        }
    }

    fn reset(&mut self) {
        self.processing = ProcessingState::Idle;
        self.input_path = None;
        self.output_path = None;
        self.faces_well.clear();
        self.detection_cache = None;
    }

    fn restore_defaults(&mut self) {
        let defaults = Settings::default();
        let detection_changed = self.settings.confidence != defaults.confidence;
        self.settings = Settings {
            // Preserve nothing — full restore
            ..defaults
        };
        self.settings.save();
        if detection_changed {
            self.invalidate_detection();
        }
    }
}

fn tab_button<'a>(
    tab: Tab,
    is_active: bool,
    is_hovered: bool,
    idx: usize,
    palette: iced::theme::Palette,
    fs: f32,
) -> Element<'a, Message> {
    let inactive_color = Color {
        a: 0.45,
        ..palette.text
    };
    let hover_color = Color {
        a: 0.65,
        ..palette.text
    };
    let active_color = palette.primary;

    let target_color = if is_active {
        active_color
    } else if is_hovered {
        hover_color
    } else {
        inactive_color
    };

    let primary = palette.primary;

    let tab_content: Element<'_, Message> = AnimationBuilder::new(target_color, move |color| {
        let label = text(tab.label())
            .size(scaled(14.0, fs))
            .color(color)
            .font(iced::Font {
                weight: iced::font::Weight::Semibold,
                ..iced::Font::DEFAULT
            });
        let btn = button(label)
            .on_press(Message::TabSelected(tab))
            .padding([12, 20])
            .width(Length::Shrink)
            .style(move |_theme: &Theme, _status: button::Status| button::Style {
                text_color: color,
                ..button::Style::default()
            });

        let bar: Element<'_, Message> = if is_active {
            container(
                container(Space::new().height(0))
                    .width(Length::Fill)
                    .height(2.5)
                    .style(move |_theme: &Theme| container::Style {
                        background: Some(primary.into()),
                        border: iced::border::Border {
                            radius: iced::border::Radius {
                                top_left: 2.0,
                                top_right: 2.0,
                                bottom_right: 0.0,
                                bottom_left: 0.0,
                            },
                            ..iced::border::Border::default()
                        },
                        ..Default::default()
                    }),
            )
            .padding([0, 12])
            .into()
        } else {
            Space::new().height(2.5).into()
        };

        column![btn, bar]
            .width(Length::Shrink)
            .align_x(iced::Alignment::Center)
            .into()
    })
    .animation(Easing::EASE_OUT.with_duration(Duration::from_millis(200)))
    .into();

    mouse_area(tab_content)
        .on_enter(Message::TabHover(idx, true))
        .on_exit(Message::TabHover(idx, false))
        .into()
}

fn has_supported_extension(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SUPPORTED_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn scaled(base: f32, font_scale: f32) -> f32 {
    (base * font_scale).round()
}
