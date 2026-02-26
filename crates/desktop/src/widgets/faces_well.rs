use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use iced::widget::{checkbox, column, row, text, Space};
use iced::{Element, Length, Theme};

use crate::app::{scaled, Message};
use crate::theme::tertiary_color;
use crate::widgets::face_card;

const CARD_SPACING: f32 = 10.0;

pub struct FacesWellState {
    pub crops: HashMap<u32, PathBuf>,
    pub groups: Vec<Vec<u32>>,
    pub group_faces: bool,
    pub selected: HashSet<u32>,
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

    pub fn clear(&mut self) {
        self.crops.clear();
        self.groups.clear();
        self.selected.clear();
        self.temp_dir = None;
    }

    pub fn toggle_face(&mut self, track_id: u32) {
        if !self.selected.remove(&track_id) {
            self.selected.insert(track_id);
        }
    }

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

    /// Returns None when all faces are selected (blur all),
    /// or Some(subset) for selective blurring.
    pub fn get_selected_ids(&self) -> Option<HashSet<u32>> {
        if self.selected.len() == self.crops.len() {
            None
        } else {
            Some(self.selected.clone())
        }
    }

    pub fn total_count(&self) -> usize {
        self.crops.len()
    }

    pub fn selected_count(&self) -> usize {
        self.selected.len()
    }

    pub fn has_faces(&self) -> bool {
        !self.crops.is_empty()
    }
}

pub fn view<'a>(state: &FacesWellState, fs: f32, theme: &Theme, hovered: &std::collections::HashSet<u32>) -> Element<'a, Message> {
    if !state.has_faces() {
        return column![].into();
    }

    let tertiary = tertiary_color(theme);
    let selected = state.selected_count();
    let total = state.total_count();

    let count_label = if state.group_faces && !state.groups.is_empty() {
        format!(
            "{selected} of {total} selected ({} groups)",
            state.groups.len()
        )
    } else {
        format!("{selected} of {total} selected")
    };

    // Header row: title + count on left, group toggle on right
    let header = row![
        row![
            text("Detected Faces").size(scaled(16.0, fs)),
            Space::new().width(10),
            text(count_label).size(scaled(14.0, fs)).color(tertiary),
        ]
        .align_y(iced::Alignment::Center),
        Space::new().width(Length::Fill),
        checkbox(state.group_faces)
            .label("Group similar")
            .on_toggle(Message::GroupFacesToggled)
            .text_size(scaled(14.0, fs)),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let grid = if state.group_faces && !state.groups.is_empty() {
        build_grouped_grid(state, fs, theme, hovered)
    } else {
        build_individual_grid(state, fs, theme, hovered)
    };

    column![header, Space::new().height(14), grid,]
        .spacing(0)
        .width(Length::Fill)
        .into()
}

fn build_individual_grid<'a>(state: &FacesWellState, fs: f32, theme: &Theme, hovered: &std::collections::HashSet<u32>) -> Element<'a, Message> {
    let mut sorted_ids: Vec<u32> = state.crops.keys().copied().collect();
    sorted_ids.sort();

    let cards: Vec<Element<'a, Message>> = sorted_ids
        .into_iter()
        .filter_map(|track_id| {
            let path = state.crops.get(&track_id)?;
            let is_selected = state.selected.contains(&track_id);
            Some(face_card::face_card(
                path,
                is_selected,
                Message::ToggleFace(track_id),
                None,
                hovered.contains(&track_id),
                track_id,
                fs,
                theme,
            ))
        })
        .collect();

    wrap_cards(cards)
}

fn build_grouped_grid<'a>(state: &FacesWellState, fs: f32, theme: &Theme, hovered: &std::collections::HashSet<u32>) -> Element<'a, Message> {
    let cards: Vec<Element<'a, Message>> = state
        .groups
        .iter()
        .enumerate()
        .filter_map(|(group_idx, group)| {
            let representative_id = group.first()?;
            let path = state.crops.get(representative_id)?;
            let all_selected = group.iter().all(|id| state.selected.contains(id));
            let badge = if group.len() > 1 {
                Some(format!("\u{00d7}{}", group.len()))
            } else {
                None
            };
            Some(face_card::face_card(
                path,
                all_selected,
                Message::ToggleGroup(group_idx),
                badge,
                hovered.contains(representative_id),
                *representative_id,
                fs,
                theme,
            ))
        })
        .collect();

    wrap_cards(cards)
}

fn wrap_cards(cards: Vec<Element<'_, Message>>) -> Element<'_, Message> {
    let cards_per_row = ((548.0) / (face_card::FULL_CARD_SIZE + CARD_SPACING)).floor() as usize;
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

