/// Domain interface for grouping face crops by identity.
///
/// Takes a list of (track_id, crop_data, width, height) tuples and
/// returns groups of track_ids that represent the same person.
pub trait FaceGrouper: Send {
    fn group(
        &self,
        crops: &[(u32, &[u8], u32, u32)],
    ) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>>;
}
