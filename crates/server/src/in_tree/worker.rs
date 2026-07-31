//! Bounded, single-consumer generation worker for the in-tree adapter.
//!
//! Every accepted job runs to completion on one dedicated thread. The bounded
//! queue makes overload explicit, while streaming jobs cooperate with their
//! response channel so a disconnected or slow client cannot detach compute.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const QUEUE_DEPTH: usize = 8;

type GenerationJob = Box<dyn FnOnce() + Send + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PostError {
    Full,
    Unavailable,
}

#[derive(Clone)]
pub(super) struct GenerationWorker {
    sender: tokio::sync::mpsc::Sender<GenerationJob>,
    depth: Arc<AtomicUsize>,
}

impl GenerationWorker {
    pub(super) fn spawn() -> Self {
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<GenerationJob>(QUEUE_DEPTH);
        let depth = Arc::new(AtomicUsize::new(0));
        let worker_depth = Arc::clone(&depth);
        std::thread::Builder::new()
            .name("camelid-in-tree-generation".to_string())
            .spawn(move || {
                while let Some(job) = receiver.blocking_recv() {
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    worker_depth.fetch_sub(1, Ordering::SeqCst);
                }
            })
            .expect("spawn the in-tree generation worker");
        Self { sender, depth }
    }

    pub(super) fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    pub(super) fn post(&self, job: GenerationJob) -> Result<(), PostError> {
        self.depth.fetch_add(1, Ordering::SeqCst);
        match self.sender.try_send(job) {
            Ok(()) => Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(PostError::Full)
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                self.depth.fetch_sub(1, Ordering::SeqCst);
                Err(PostError::Unavailable)
            }
        }
    }

    pub(super) async fn run<T, F>(&self, job: F) -> Result<T, PostError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (result_sender, result_receiver) = tokio::sync::oneshot::channel();
        self.post(Box::new(move || {
            let _ = result_sender.send(job());
        }))?;
        result_receiver.await.map_err(|_| PostError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::time::Duration;

    #[tokio::test]
    async fn worker_serializes_jobs_and_recovers_depth() {
        let worker = GenerationWorker::spawn();
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let mut jobs = tokio::task::JoinSet::new();

        for _ in 0..4 {
            let worker = worker.clone();
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            jobs.spawn(async move {
                worker
                    .run(move || {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        maximum.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_millis(5));
                        active.fetch_sub(1, Ordering::SeqCst);
                    })
                    .await
                    .unwrap();
            });
        }
        while jobs.join_next().await.is_some() {}

        assert_eq!(maximum.load(Ordering::SeqCst), 1);
        assert_eq!(worker.depth(), 0);
    }

    #[tokio::test]
    async fn the_ninth_queued_job_is_rejected() {
        let worker = GenerationWorker::spawn();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker_entered = Arc::clone(&entered);
        let worker_release = Arc::clone(&release);
        worker
            .post(Box::new(move || {
                worker_entered.wait();
                worker_release.wait();
            }))
            .unwrap();
        entered.wait();

        for _ in 0..QUEUE_DEPTH {
            worker.post(Box::new(|| {})).unwrap();
        }
        assert_eq!(worker.depth(), QUEUE_DEPTH + 1);
        assert_eq!(worker.post(Box::new(|| {})), Err(PostError::Full));

        release.wait();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while worker.depth() != 0 {
            assert!(std::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
}
