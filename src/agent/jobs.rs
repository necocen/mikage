//! Bounded in-memory store for asynchronous agent jobs and their results.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::Notify;

use super::{AgentConfig, AgentResponse, JobId};

struct Job {
    created: Instant,
    response: Option<AgentResponse>,
}

struct JobState {
    next_id: JobId,
    jobs: HashMap<JobId, Job>,
    result_bytes: usize,
}

pub(super) struct JobStore {
    state: Mutex<JobState>,
    changed: Notify,
    max_jobs: usize,
    max_result_bytes: usize,
    ttl: Duration,
}

impl JobStore {
    pub(super) fn new(config: &AgentConfig) -> Self {
        Self {
            state: Mutex::new(JobState {
                next_id: 1,
                jobs: HashMap::new(),
                result_bytes: 0,
            }),
            changed: Notify::new(),
            max_jobs: config.max_jobs,
            max_result_bytes: config.max_result_bytes,
            ttl: config.job_ttl,
        }
    }

    fn expire(&self, state: &mut JobState) {
        state.jobs.retain(|_, job| job.created.elapsed() < self.ttl);
        state.result_bytes = state
            .jobs
            .values()
            .filter_map(|job| job.response.as_ref())
            .map(AgentResponse::byte_len)
            .sum();
    }

    pub(super) fn create(&self) -> Result<JobId, AgentResponse> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        if state.jobs.len() >= self.max_jobs {
            return Err(AgentResponse::busy(
                "job capacity exhausted; wait for results to expire",
            ));
        }
        let id = state.next_id;
        state.next_id = id
            .checked_add(1)
            .ok_or_else(|| AgentResponse::unavailable("job identifiers exhausted"))?;
        state.jobs.insert(
            id,
            Job {
                created: Instant::now(),
                response: None,
            },
        );
        Ok(id)
    }

    /// Release a reservation that was never accepted into the application queue.
    /// The HTTP caller does not receive its id, so retaining it would consume
    /// capacity until TTL without leaving a reachable result.
    pub(super) fn discard(&self, id: JobId) {
        let mut state = self.state.lock().unwrap();
        if let Some(job) = state.jobs.remove(&id) {
            if let Some(response) = job.response {
                state.result_bytes = state.result_bytes.saturating_sub(response.byte_len());
            }
            self.changed.notify_waiters();
        }
    }

    pub(super) fn is_pending(&self, id: JobId) -> bool {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        state
            .jobs
            .get(&id)
            .is_some_and(|job| job.response.is_none())
    }

    pub(super) fn complete(&self, id: JobId, mut response: AgentResponse) {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        if state.jobs.get(&id).is_none_or(|job| job.response.is_some()) {
            return;
        }
        let length = response.byte_len();
        if length > self.max_result_bytes.saturating_sub(state.result_bytes) {
            response =
                AgentResponse::busy("completed result exceeds the retained result memory budget");
        }
        state.result_bytes += response.byte_len();
        state.jobs.get_mut(&id).unwrap().response = Some(response);
        self.changed.notify_waiters();
    }

    pub(super) fn status(&self, id: JobId) -> Option<Value> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        let job = state.jobs.get(&id)?;
        Some(match &job.response {
            None => json!({"id":id,"state":"pending"}),
            Some(AgentResponse::Error { status, message }) => {
                json!({"id":id,"state":"failed","status":status,"error":message})
            }
            Some(response) => {
                let mut value = json!({"id":id,"state":"completed","result_url":format!("/jobs/{id}/result"),"bytes":response.byte_len()});
                if let AgentResponse::Bytes { metadata, .. } = response {
                    value["metadata"] = metadata.clone();
                }
                value
            }
        })
    }

    pub(super) fn result(&self, id: JobId) -> Result<Option<AgentResponse>, AgentResponse> {
        let mut state = self.state.lock().unwrap();
        self.expire(&mut state);
        state
            .jobs
            .get(&id)
            .map(|job| job.response.clone())
            .ok_or_else(|| AgentResponse::Error {
                status: 404,
                message: "unknown or expired job".into(),
            })
    }

    pub(super) async fn wait(
        &self,
        id: JobId,
        timeout: Duration,
    ) -> Result<AgentResponse, AgentResponse> {
        let started = Instant::now();
        loop {
            // Register before checking the store so completion cannot be lost.
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let remaining = {
                let mut state = self.state.lock().unwrap();
                self.expire(&mut state);
                let job = state.jobs.get(&id).ok_or_else(|| AgentResponse::Error {
                    status: 404,
                    message: "unknown or expired job".into(),
                })?;
                if let Some(response) = &job.response {
                    return Ok(response.clone());
                }
                timeout
                    .saturating_sub(started.elapsed())
                    .min(self.ttl.saturating_sub(job.created.elapsed()))
            };
            if remaining.is_zero() {
                return Err(AgentResponse::Error {
                    status: 504,
                    message: format!("job {id} is still pending; inspect /jobs/{id}"),
                });
            }
            let _ = tokio::time::timeout(remaining, changed).await;
        }
    }

    pub(super) fn fail_all(&self, message: &str) {
        let ids: Vec<_> = self
            .state
            .lock()
            .unwrap()
            .jobs
            .iter()
            .filter(|(_, job)| job.response.is_none())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.complete(id, AgentResponse::unavailable(message));
        }
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.state.lock().unwrap().jobs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_capacity_ttl_and_result_bytes_are_bounded() {
        let config = AgentConfig {
            max_jobs: 1,
            max_result_bytes: 4,
            ..Default::default()
        };
        let jobs = JobStore::new(&config);
        let id = jobs
            .create()
            .unwrap_or_else(|_| panic!("first job should fit"));
        assert!(matches!(
            jobs.create(),
            Err(AgentResponse::Error { status: 429, .. })
        ));
        jobs.complete(id, AgentResponse::Png(vec![0; 5]));
        assert_eq!(jobs.status(id).unwrap()["state"], "failed");
        assert_eq!(jobs.state.lock().unwrap().result_bytes, 0);
        jobs.state
            .lock()
            .unwrap()
            .jobs
            .get_mut(&id)
            .unwrap()
            .created = Instant::now() - config.job_ttl;
        assert!(jobs.status(id).is_none());
        assert!(jobs.create().is_ok());
    }
}
