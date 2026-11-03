use crate::api::{TranscriptSearchResult, TranscriptSegment};
use chrono::Utc;
use sqlx::{Connection, Error as SqlxError, SqlitePool};
use tracing::{error, info};
use uuid::Uuid;

pub struct TranscriptsRepository;

impl TranscriptsRepository {
    /// Saves a new meeting and its associated transcript segments.
    /// This function uses a transaction to ensure that either both the meeting
    /// and all its transcripts are saved, or none of them are.
    pub async fn save_transcript(
        pool: &SqlitePool,
        meeting_title: &str,
        transcripts: &[TranscriptSegment],
        folder_path: Option<String>,
    ) -> Result<String, SqlxError> {
        let meeting_id = format!("meeting-{}", Uuid::new_v4());

        let mut conn = pool.acquire().await?;
        let mut transaction = conn.begin().await?;

        let now = Utc::now();

        // 1. Create the new meeting
        let result = sqlx::query(
            "INSERT INTO meetings (id, title, created_at, updated_at, folder_path) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&meeting_id)
        .bind(meeting_title)
        .bind(now)
        .bind(now)
        .bind(&folder_path)
        .execute(&mut *transaction)
        .await;

        if let Err(e) = result {
            error!("Failed to create meeting '{}': {}", meeting_title, e);
            transaction.rollback().await?;
            return Err(e);
        }

        info!("Successfully created meeting with id: {}", meeting_id);

        // 2. Save each transcript segment with audio timing fields
        for segment in transcripts {
            let transcript_id = format!("transcript-{}", Uuid::new_v4());
            let result = sqlx::query(
                "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, speaker)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&transcript_id)
            .bind(&meeting_id)
            .bind(&segment.text)
            .bind(&segment.timestamp)
            .bind(segment.audio_start_time)
            .bind(segment.audio_end_time)
            .bind(segment.duration)
            .bind(None::<String>)
            .execute(&mut *transaction)
            .await;

            if let Err(e) = result {
                error!(
                    "Failed to save transcript segment for meeting {}: {}",
                    meeting_id, e
                );
                transaction.rollback().await?;
                return Err(e);
            }
        }

        info!(
            "Successfully saved {} transcript segments for meeting {}",
            transcripts.len(),
            meeting_id
        );

        // Commit the transaction
        transaction.commit().await?;

        Ok(meeting_id)
    }

    /// Relabel all of one speaker's rows in a meeting (speaker naming).
    /// Returns the number of rows changed.
    pub async fn rename_speaker_rows(
        pool: &SqlitePool,
        meeting_id: &str,
        old_speaker: &str,
        new_speaker: &str,
    ) -> Result<u64, SqlxError> {
        let result =
            sqlx::query("UPDATE transcripts SET speaker = ? WHERE meeting_id = ? AND speaker = ?")
                .bind(new_speaker)
                .bind(meeting_id)
                .bind(old_speaker)
                .execute(pool)
                .await?;
        Ok(result.rows_affected())
    }

    /// Searches for a query string within the transcripts.
    /// It returns a list of matching transcripts with context.
    pub async fn search_transcripts(
        pool: &SqlitePool,
        query: &str,
    ) -> Result<Vec<TranscriptSearchResult>, SqlxError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let query_lower: Vec<char> = query.chars().flat_map(|c| c.to_lowercase()).collect();

        // SQLite's LIKE is case-insensitive for ASCII only, so it can prefilter ASCII
        // queries; non-ASCII queries (e.g. Cyrillic) are matched in Rust over every row.
        // LIKE wildcards in the query only widen the prefilter; the Rust match is exact.
        let rows = if query.is_ascii() {
            sqlx::query_as::<_, (String, String, String, String)>(
                "SELECT m.id, m.title, t.transcript, t.timestamp
                 FROM meetings m
                 JOIN transcripts t ON m.id = t.meeting_id
                 WHERE t.transcript LIKE ?",
            )
            .bind(format!("%{}%", query))
            .fetch_all(pool)
            .await?
        } else {
            sqlx::query_as::<_, (String, String, String, String)>(
                "SELECT m.id, m.title, t.transcript, t.timestamp
                 FROM meetings m
                 JOIN transcripts t ON m.id = t.meeting_id",
            )
            .fetch_all(pool)
            .await?
        };

        let results = rows
            .into_iter()
            .filter_map(|(id, title, transcript, timestamp)| {
                let match_range = Self::find_match_char_range(&transcript, &query_lower)?;
                let match_context = Self::format_match_context(&transcript, match_range);
                Some(TranscriptSearchResult {
                    id,
                    title,
                    match_context,
                    timestamp,
                })
            })
            .collect();

        Ok(results)
    }

    /// Finds the first Unicode case-insensitive match of `query_lower` in `transcript`,
    /// returning the `(start_char, end_char)` character indices in `transcript`.
    fn find_match_char_range(transcript: &str, query_lower: &[char]) -> Option<(usize, usize)> {
        if query_lower.is_empty() {
            return Some((0, 0));
        }

        let mut lower_chars = Vec::with_capacity(transcript.len());
        let mut lower_to_orig = Vec::with_capacity(transcript.len());
        for (orig_idx, ch) in transcript.chars().enumerate() {
            for lower_ch in ch.to_lowercase() {
                lower_chars.push(lower_ch);
                lower_to_orig.push(orig_idx);
            }
        }

        let pos = lower_chars
            .windows(query_lower.len())
            .position(|window| window == query_lower)?;

        let start_char = lower_to_orig[pos];
        let end_char = lower_to_orig[pos + query_lower.len() - 1] + 1;
        Some((start_char, end_char))
    }

    fn format_match_context(transcript: &str, (match_start, match_end): (usize, usize)) -> String {
        let chars: Vec<char> = transcript.chars().collect();
        let total_chars = chars.len();
        let start_index = match_start.saturating_sub(100);
        let end_index = (match_end + 100).min(total_chars);

        let mut context = String::new();
        if start_index > 0 {
            context.push_str("...");
        }
        context.extend(&chars[start_index..end_index]);
        if end_index < total_chars {
            context.push_str("...");
        }
        context
    }

    /// Helper function to extract a snippet of text around the first match of a query.
    #[cfg(test)]
    fn get_match_context(transcript: &str, query: &str) -> String {
        let query_lower: Vec<char> = query.chars().flat_map(|c| c.to_lowercase()).collect();
        match Self::find_match_char_range(transcript, &query_lower) {
            Some(range) => Self::format_match_context(transcript, range),
            None => transcript.chars().take(200).collect(), // Fallback to the start of the transcript
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_match_context_cyrillic_long_text_does_not_panic_and_preserves_casing() {
        // 29 * 9 + 4 = 265 bytes before the match so `match_index - 100` (165) would land
        // mid-UTF-8 character under byte-offset slicing.
        let prefix = format!("{}аб", "тест ".repeat(29));
        let suffix = format!(" вг{}", " тест".repeat(29));
        let transcript = format!("{}Привет{}", prefix, suffix);
        assert!(transcript.len() > 200);

        let context = TranscriptsRepository::get_match_context(&transcript, "привет");
        assert!(context.starts_with("..."));
        assert!(context.ends_with("..."));
        assert!(context.contains("Привет"));
        assert!(!context.contains("привет"));
        assert_eq!(context.chars().count(), 3 + 100 + "Привет".chars().count() + 100 + 3);
    }

    #[test]
    fn get_match_context_case_insensitive_cyrillic() {
        let transcript = "Обсудили Приветствие и Ёжика на утренней встрече";
        assert_eq!(
            TranscriptsRepository::get_match_context(transcript, "приветствие"),
            transcript
        );
        assert_eq!(
            TranscriptsRepository::get_match_context(transcript, "ПРИВЕТСТВИЕ"),
            transcript
        );
        assert_eq!(
            TranscriptsRepository::get_match_context(transcript, "ёжика"),
            transcript
        );

        // Characters whose lowercase expands in UTF-8 length ('İ' -> "i\u{307}") must not
        // skew offsets of subsequent Cyrillic matches.
        let with_expansion = "İ Привет мир";
        assert_eq!(
            TranscriptsRepository::get_match_context(with_expansion, "привет"),
            with_expansion
        );
    }

    #[test]
    fn get_match_context_ascii_behaviour_unchanged() {
        let short = "We reviewed Project Alpha timelines today.";
        assert_eq!(
            TranscriptsRepository::get_match_context(short, "project alpha"),
            short
        );

        let long = format!("{}TargetPhrase{}", "a".repeat(120), "b".repeat(120));
        let expected = format!("...{}TargetPhrase{}...", "a".repeat(100), "b".repeat(100));
        assert_eq!(
            TranscriptsRepository::get_match_context(&long, "targetphrase"),
            expected
        );
    }

    #[test]
    fn get_match_context_no_match_fallback() {
        let transcript = "абвгдежзий".repeat(25); // 250 chars
        let fallback = TranscriptsRepository::get_match_context(&transcript, "несуществующий");
        assert_eq!(fallback.chars().count(), 200);
        assert_eq!(fallback, transcript.chars().take(200).collect::<String>());
    }

    #[tokio::test]
    async fn search_transcripts_matches_cyrillic_case_insensitively() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query("CREATE TABLE meetings (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE transcripts (
                id TEXT PRIMARY KEY,
                meeting_id TEXT NOT NULL,
                transcript TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query("INSERT INTO meetings (id, title) VALUES ('m1', 'Встреча')")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query(
            "INSERT INTO transcripts (id, meeting_id, transcript, timestamp)
             VALUES ('t1', 'm1', 'Привет, коллеги!', '00:01:00'),
                    ('t2', 'm1', 'Другая реплика', '00:02:00'),
                    ('t3', 'm1', 'Review of Project Alpha', '00:03:00')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let results = TranscriptsRepository::search_transcripts(&pool, "привет")
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "m1");
        assert_eq!(results[0].match_context, "Привет, коллеги!");

        // ASCII queries take the SQL LIKE prefilter path
        let results = TranscriptsRepository::search_transcripts(&pool, "PROJECT alpha")
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].match_context, "Review of Project Alpha");
    }
}
