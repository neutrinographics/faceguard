use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use iced::widget::{button, checkbox, column, container, image, row, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::muted_color;

/// Card size in logical pixels.
const CARD_SIZE: f32 = 96.0;
/// Spacing between cards.
const CARD_SPACING: f32 = 10.0;

/// State for the faces well.
pub struct FacesWellState {
    /// track_id → path to thumbnail JPEG
    pub crops: HashMap<u32, PathBuf>,
    /// Groups of track_ids (same person)
    pub groups: Vec<Vec<u32>>,
    /// Whether to group similar faces
    pub group_faces: bool,
    /// Currently selected face IDs
    pub selected: HashSet<u32>,
    /// Temp directory holding crop images (cleaned up via RAII on drop).
    temp_dir: Option<tempfile::TempDir>,
}

impl FacesWellState {
    pub fn new() -> Self {
        Self {
            crops: HashMap::new(),
            groups: vec![],
            group_faces: true,
            selected: HashSet::new(),
            temp_dir: None,
        }
    }

    /// Populate with preview results. Selects all faces by default.
    pub fn populate(
        &mut self,
        crops: HashMap<u32, PathBuf>,
        groups: Vec<Vec<u32>>,
        temp_dir: tempfile::TempDir,
    ) {
        self.selected = crops.keys().copied().collect();
        self.crops = crops;
        self.groups = groups;
        self.temp_dir = Some(temp_dir);
    }

    /// Clear all state and clean up temp directory.
    pub fn clear(&mut self) {
        self.crops.clear();
        self.groups.clear();
        self.selected.clear();
        self.temp_dir = None;
    }

    /// Toggle selection of a single face.
    pub fn toggle_face(&mut self, track_id: u32) {
        if self.selected.contains(&track_id) {
            self.selected.remove(&track_id);
        } else {
            self.selected.insert(track_id);
        }
    }

    /// Toggle selection of all faces in a group.
    pub fn toggle_group(&mut self, group_idx: usize) {
        if let Some(group) = self.groups.get(group_idx) {
            let all_selected = group.iter().all(|id| self.selected.contains(id));
            for &id in group {
                if all_selected {
                    self.selected.remove(&id);
                } else {
                    self.selected.insert(id);
                }
            }
        }
    }

    /// Returns None if all faces are selected (meaning blur all),
    /// or Some(set) if only a subset is selected.
    pub fn get_selected_ids(&self) -> Option<HashSet<u32>> {
        if self.selected.len() == self.crops.len() {
            None
        } else {
            Some(self.selected.clone())
        }
    }

    /// Total number of faces.
    pub fn total_count(&self) -> usize {
        self.crops.len()
    }

    /// Number of selected faces.
    pub fn selected_count(&self) -> usize {
        self.selected.len()
    }

    /// Whether any faces are loaded.
    pub fn has_faces(&self) -> bool {
        !self.crops.is_empty()
    }
}

/// Render the faces well UI.
pub fn view<'a>(state: &FacesWellState, fs: f32, theme: &Theme) -> Element<'a, Message> {
    if !state.has_faces() {
        return column![].into();
    }

    let muted = muted_color(theme);
    let selected = state.selected_count();
    let total = state.total_count();

    let label = if state.group_faces && !state.groups.is_empty() {
        let group_count = state.groups.len();
        format!("{selected} of {total} faces selected ({group_count} groups)")
    } else {
        format!("{selected} of {total} faces selected")
    };

    let mut col = column![].spacing(0).width(Length::Fill);

    // Card grid
    if state.group_faces && !state.groups.is_empty() {
        col = col.push(build_grouped_grid(state, fs));
    } else {
        col = col.push(build_individual_grid(state, fs));
    }

    // Footer: checkbox + count label
    col = col.push(Space::new().height(10));
    col = col.push(
        row![
            checkbox(state.group_faces)
                .label("Group similar faces")
                .on_toggle(Message::GroupFacesToggled)
                .text_size(scaled(12.0, fs)),
            Space::new().width(Length::Fill),
            text(label).size(scaled(11.0, fs)).color(muted),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center),
    );

    container(col)
        .padding(10)
        .style(container::rounded_box)
        .width(Length::Fill)
        .into()
}

/// Build a grid of individual face cards.
fn build_individual_grid<'a>(state: &FacesWellState, fs: f32) -> Element<'a, Message> {
    let mut sorted_ids: Vec<u32> = state.crops.keys().copied().collect();
    sorted_ids.sort();

    let cards: Vec<Element<'a, Message>> = sorted_ids
        .into_iter()
        .filter_map(|track_id| {
            let path = state.crops.get(&track_id)?;
            Some(face_card(
                track_id,
                path,
                state.selected.contains(&track_id),
                fs,
            ))
        })
        .collect();

    wrap_cards(cards)
}

/// Build a grid of group cards.
fn build_grouped_grid<'a>(state: &FacesWellState, fs: f32) -> Element<'a, Message> {
    let cards: Vec<Element<'a, Message>> = state
        .groups
        .iter()
        .enumerate()
        .filter_map(|(group_idx, group)| {
            let representative_id = group.first()?;
            let path = state.crops.get(representative_id)?;
            let all_selected = group.iter().all(|id| state.selected.contains(id));
            let count = group.len();
            Some(group_card(group_idx, path, all_selected, count, fs))
        })
        .collect();

    wrap_cards(cards)
}

/// Arrange cards in wrapping rows.
fn wrap_cards(cards: Vec<Element<'_, Message>>) -> Element<'_, Message> {
    let cards_per_row = ((480.0) / (CARD_SIZE + CARD_SPACING)).floor() as usize;
    let cards_per_row = cards_per_row.max(1);

    let mut rows_col = column![].spacing(CARD_SPACING);
    let mut current_row = row![].spacing(CARD_SPACING);
    let mut count_in_row = 0;

    for card in cards {
        current_row = current_row.push(card);
        count_in_row += 1;
        if count_in_row >= cards_per_row {
            rows_col = rows_col.push(current_row);
            current_row = row![].spacing(CARD_SPACING);
            count_in_row = 0;
        }
    }

    if count_in_row > 0 {
        rows_col = rows_col.push(current_row);
    }

    rows_col.into()
}

/// A single face card: 96x96 thumbnail with selection state.
fn face_card<'a>(track_id: u32, path: &PathBuf, selected: bool, fs: f32) -> Element<'a, Message> {
    let img = image(image::Handle::from_path(path))
        .width(CARD_SIZE)
        .height(CARD_SIZE);

    let label = if selected { "\u{2713}" } else { "" };

    let btn =
        button(column![img, text(label).size(scaled(10.0, fs)),].align_x(iced::Alignment::Center))
            .on_press(Message::ToggleFace(track_id))
            .padding(2);

    let btn = if selected {
        btn.style(button::primary)
    } else {
        btn.style(button::secondary)
    };

    container(btn).width(CARD_SIZE + 8.0).into()
}

/// A group card: shows representative thumbnail with count badge.
fn group_card<'a>(
    group_idx: usize,
    path: &PathBuf,
    selected: bool,
    count: usize,
    fs: f32,
) -> Element<'a, Message> {
    let img = image(image::Handle::from_path(path))
        .width(CARD_SIZE)
        .height(CARD_SIZE);

    let badge = if count > 1 {
        format!("x{count}")
    } else {
        String::new()
    };

    let label = if selected { "\u{2713}" } else { "" };

    let btn = button(
        column![
            img,
            row![
                text(label).size(scaled(10.0, fs)),
                Space::new().width(Length::Fill),
                text(badge).size(scaled(10.0, fs)),
            ],
        ]
        .align_x(iced::Alignment::Center),
    )
    .on_press(Message::ToggleGroup(group_idx))
    .padding(2);

    let btn = if selected {
        btn.style(button::primary)
    } else {
        btn.style(button::secondary)
    };

    container(btn).width(CARD_SIZE + 8.0).into()
}
