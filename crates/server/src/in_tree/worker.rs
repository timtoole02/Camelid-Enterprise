//! Bounded, fixed-width generation worker pool for the in-tree adapter.
//!
//! Every accepted job runs to completion on one of `slots` dedicated threads.
//! The bounded queue makes overload explicit, while streaming jobs cooperate
//! with their response channel so a disconnected or slow client cannot detach
//! compute.
//!
//! Width is a throughput decision and not a numeric one. Each generation builds
//! its own decoder and its own KV cache over read-only weights, so two running
//! side by side share no mutable state and neither can observe the other's
//! arithmetic: a request emits the same tokens at any width, including across a
//! width change. That is what separates a pool from batching, which fuses
//! requests into shared kernel shapes and would move the output.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Jobs admitted *beyond* the ones already running. A replica therefore admits
/// `slots + QUEUE_DEPTH` before it refuses, and the refusal is the client's
/// signal to retry rather than a queue that grows without bound.
const QUEUE_DEPTH: usize = 8;

type GenerationJob = Box<dyn FnOnce() + Send + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PostError {
    Full,
    Unavailable,
}

struct Queue {
    jobs: VecDeque<GenerationJob>,
    /// Cleared when the last public handle drops, which is what lets the
    /// threads leave their wait and exit rather than outlive the router.
    open: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    admitted: Condvar,
}

#[derive(Clone)]
pub(super) struct GenerationWorker {
    shared: Arc<Shared>,
    depth: Arc<AtomicUsize>,
    /// Counts live public handles only. The worker threads hold `shared`
    /// without holding this, so "the last handle went away" stays a question
    /// about callers and does not become a question the threads answer about
    /// themselves — which is the shape the previous channel gave for free and
    /// an `Arc<Shared>` count would quietly get wrong.
    handles: Arc<()>,
    slots: usize,
}

impl GenerationWorker {
    /// Spawn a pool `slots` generations wide.
    ///
    /// A width below one is raised to one rather than refused: the serving
    /// binary already fails closed on a zero width at the flag, and a router
    /// assembled in a test has no operator to report to.
    pub(super) fn spawn(slots: usize) -> Self {
        let slots = slots.max(1);
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                open: true,
            }),
            admitted: Condvar::new(),
        });
        let depth = Arc::new(AtomicUsize::new(0));

        for index in 0..slots {
            let shared = Arc::clone(&shared);
            let depth = Arc::clone(&depth);
            std::thread::Builder::new()
                .name(format!("camelid-in-tree-generation-{index}"))
                .spawn(move || loop {
                    let job = {
                        let mut queue = shared
                            .queue
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        loop {
                            if let Some(job) = queue.jobs.pop_front() {
                                break Some(job);
                            }
                            if !queue.open {
                                break None;
                            }
                            queue = shared
                                .admitted
                                .wait(queue)
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                        }
                    };
                    // The guard is released above, so a long generation never
                    // holds the queue against the other slots.
                    let Some(job) = job else { return };
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    depth.fetch_sub(1, Ordering::SeqCst);
                })
                .expect("spawn an in-tree generation worker");
        }

        Self {
            shared,
            depth,
            handles: Arc::new(()),
            slots,
        }
    }

    /// Running plus queued jobs.
    pub(super) fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// How many generations this replica runs at once.
    pub(super) fn slots(&self) -> usize {
        self.slots
    }

    pub(super) fn post(&self, job: GenerationJob) -> Result<(), PostError> {
        {
            let mut queue = self
                .shared
                .queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !queue.open {
                return Err(PostError::Unavailable);
            }
            if queue.jobs.len() >= QUEUE_DEPTH {
                return Err(PostError::Full);
            }
            queue.jobs.push_back(job);
            self.depth.fetch_add(1, Ordering::SeqCst);
        }
        self.shared.admitted.notify_one();
        Ok(())
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

impl Drop for GenerationWorker {
    fn drop(&mut self) {
        if Arc::strong_count(&self.handles) > 1 {
            return;
        }
        {
            let mut queue = self
                .shared
                .queue
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            queue.open = false;
        }
        self.shared.admitted.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::time::{Duration, Instant};

    /// Waits for the depth to fall back to zero.
    ///
    /// `run` resolves from inside the job closure, one statement before the
    /// worker loop decrements the depth, so a caller can observe its own
    /// completion while the slot it occupied is still counted. Asserting zero
    /// the instant the last await resolves is a race the worker thread usually
    /// wins and a loaded runner does not.
    async fn depth_recovers(worker: &GenerationWorker) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.depth() != 0 {
            assert!(Instant::now() < deadline, "depth never returned to zero");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    #[tokio::test]
    async fn one_slot_serializes_jobs_and_recovers_depth() {
        let worker = GenerationWorker::spawn(1);
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
        depth_recovers(&worker).await;
    }

    /// The property the pool exists for, asserted so it cannot regress to the
    /// serialized behavior without failing.
    ///
    /// Each job announces itself and then waits until two are inside at once,
    /// with its own deadline. At width two both jobs satisfy the wait and
    /// `maximum` reaches two. At width one the first job's wait expires and
    /// `maximum` stays one — so a regression fails this assertion instead of
    /// hanging the suite, which a bare `Barrier` here would do.
    #[tokio::test]
    async fn two_slots_run_two_generations_at_once() {
        let worker = GenerationWorker::spawn(2);
        let gate = Arc::new((Mutex::new(0usize), Condvar::new()));
        let maximum = Arc::new(AtomicUsize::new(0));
        let mut jobs = tokio::task::JoinSet::new();

        for _ in 0..2 {
            let worker = worker.clone();
            let gate = Arc::clone(&gate);
            let maximum = Arc::clone(&maximum);
            jobs.spawn(async move {
                worker
                    .run(move || {
                        let (lock, inside_changed) = &*gate;
                        let mut inside = lock.lock().unwrap();
                        *inside += 1;
                        maximum.fetch_max(*inside, Ordering::SeqCst);
                        inside_changed.notify_all();

                        let deadline = Instant::now() + Duration::from_secs(5);
                        while *inside < 2 && Instant::now() < deadline {
                            let (guard, _) = inside_changed
                                .wait_timeout(inside, Duration::from_millis(25))
                                .unwrap();
                            inside = guard;
                            maximum.fetch_max(*inside, Ordering::SeqCst);
                        }
                        *inside -= 1;
                    })
                    .await
                    .unwrap();
            });
        }
        while jobs.join_next().await.is_some() {}

        assert_eq!(
            maximum.load(Ordering::SeqCst),
            2,
            "two slots must run two generations concurrently"
        );
        depth_recovers(&worker).await;
    }

    #[tokio::test]
    async fn the_job_past_the_queue_is_rejected() {
        let worker = GenerationWorker::spawn(1);
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
        depth_recovers(&worker).await;
    }

    /// Admission is `slots` running plus `QUEUE_DEPTH` waiting, so a wider pool
    /// admits strictly more before refusing. Asserted because the queue bound
    /// and the width are separate numbers and it would be easy to write one
    /// that silently shadowed the other.
    #[tokio::test]
    async fn a_wider_pool_admits_its_extra_running_jobs() {
        let worker = GenerationWorker::spawn(2);
        let entered = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        for _ in 0..2 {
            let worker_entered = Arc::clone(&entered);
            let worker_release = Arc::clone(&release);
            worker
                .post(Box::new(move || {
                    worker_entered.wait();
                    worker_release.wait();
                }))
                .unwrap();
        }
        entered.wait();

        for _ in 0..QUEUE_DEPTH {
            worker.post(Box::new(|| {})).unwrap();
        }
        assert_eq!(worker.depth(), QUEUE_DEPTH + 2);
        assert_eq!(worker.post(Box::new(|| {})), Err(PostError::Full));

        release.wait();
        depth_recovers(&worker).await;
    }

    #[tokio::test]
    async fn the_reported_width_is_the_width_that_runs() {
        assert_eq!(GenerationWorker::spawn(3).slots(), 3);
        // Clamped rather than refused; the binary rejects zero at the flag.
        assert_eq!(GenerationWorker::spawn(0).slots(), 1);
    }
}
