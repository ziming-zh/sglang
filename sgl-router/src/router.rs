use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, info, warn};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use tokio;

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Arc<RwLock<Vec<String>>>,
        current_index: AtomicUsize,
    },
    Random {
        worker_urls: Arc<RwLock<Vec<String>>>,
    },
    CacheAware {
        /*
            Cache-Aware Load Balancing Router

            This router combines two strategies to optimize both cache utilization and request distribution:

            1. Cache-Aware Routing (Approximate Tree)
            2. Load Balancing (Shortest Queue with Balance Thresholds)

            The router dynamically switches between these strategies based on load conditions:
            - Uses load balancing when the system is imbalanced
            - Uses cache-aware routing when the system is balanced

            A system is considered imbalanced if both conditions are met:
            1. (max - min) > abs_threshold
            2. max > rel_threshold * min

            Strategy Details:

            1. Cache-Aware Routing (Approximate Tree)
            -------------------------------------------
            This strategy maintains an approximate radix tree for each worker based on request history,
            eliminating the need for direct cache state queries. The tree stores raw text characters
            instead of token IDs to avoid tokenization overhead.

            Process:
            a. For each request, find the worker with the highest prefix match
            b. If match rate > cache_threshold:
            Route to the worker with highest match (likely has relevant data cached)
            c. If match rate ≤ cache_threshold:
            Route to the worker with smallest tree size (most available cache capacity)
            d. Background maintenance:
            Periodically evict least recently used leaf nodes to prevent memory overflow

            2. Load Balancing (Shortest Queue)
            -------------------------------------------
            This strategy tracks pending request counts per worker and routes new requests
            to the least busy worker when the system is detected to be imbalanced.

            Configuration Parameters:
            ------------------------
            1. cache_threshold: (float, 0.0 to 1.0)
            Minimum prefix match ratio to use highest-match routing.
            Below this threshold, routes to worker with most available cache space.

            2. balance_abs_threshold: (integer)
            Absolute difference threshold for load imbalance detection.
            System is potentially imbalanced if (max_load - min_load) > abs_threshold

            3. balance_rel_threshold: (float)
            Relative ratio threshold for load imbalance detection.
            System is potentially imbalanced if max_load > min_load * rel_threshold
            Used in conjunction with abs_threshold to determine final imbalance state.

            4. eviction_interval_secs: (integer)
            Interval between LRU eviction cycles for the approximate trees.

            5. max_tree_size: (integer)
            Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
            during the next eviction cycle.
        */
        worker_urls: Arc<RwLock<Vec<String>>>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        _eviction_thread: Option<thread::JoinHandle<()>>,
    },
}

#[derive(Debug, Clone)]
pub enum PolicyConfig {
    RandomConfig,
    RoundRobinConfig,
    CacheAwareConfig {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    },
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Result<Self, String> {
        // Wait until all workers are healthy
        Self::wait_for_healthy_workers(&worker_urls, 300, 10)?;

        // Create router based on policy...
        Ok(match policy_config {
            PolicyConfig::RandomConfig => Router::Random {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
            },
            PolicyConfig::RoundRobinConfig => Router::RoundRobin {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
                current_index: std::sync::atomic::AtomicUsize::new(0),
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let mut running_queue = HashMap::new();
                for url in &worker_urls {
                    running_queue.insert(url.clone(), 0);
                }

                let mut processed_queue = HashMap::new();
                for url in &worker_urls {
                    processed_queue.insert(url.clone(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let running_queue_clone = Arc::clone(&running_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_by_size(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        info!("Processed Queue: {:?}", locked_processed_queue);

                        // Print the running queue
                        let locked_running_queue = running_queue_clone.lock().unwrap();
                        info!("Running Queue: {:?}", locked_running_queue);
                    }
                });

                for url in &worker_urls {
                    tree.lock().unwrap().insert(&"".to_string(), url);
                }

                Router::CacheAware {
                    worker_urls: Arc::new(RwLock::new(worker_urls)),
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    _eviction_thread: Some(eviction_thread),
                }
            }
        })
    }

    fn wait_for_healthy_workers(
        worker_urls: &[String],
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let sync_client = reqwest::blocking::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                return Err(format!(
                    "Timeout {}s waiting for workers to become healthy",
                    timeout_secs
                ));
            }

            let mut all_healthy = true;
            let mut unhealthy_workers = Vec::new();

            for url in worker_urls {
                match sync_client.get(&format!("{}/health", url)).send() {
                    Ok(res) => {
                        if !res.status().is_success() {
                            info!(
                                "Worker {} health check is pending with status: {}.",
                                url,
                                res.status()
                            );
                            all_healthy = false;
                            unhealthy_workers.push((url, format!("Status: {}", res.status())));
                        }
                    }
                    Err(e) => {
                        info!("Worker {} health check is pending with error: {}", url, e);
                        all_healthy = false;
                        unhealthy_workers.push((url, format!("Error: {}", e)));
                    }
                }
            }

            if all_healthy {
                info!("All workers are healthy");
                return Ok(());
            } else {
                info!("Unhealthy workers:");
                for (url, reason) in &unhealthy_workers {
                    info!("  {} - {}", url, reason);
                }
                thread::sleep(Duration::from_secs(interval_secs));
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls }
            | Router::CacheAware { worker_urls, .. } => {
                if worker_urls.read().unwrap().is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(worker_urls.read().unwrap()[0].clone())
                }
            }
        }
    }

    async fn send_request(
        &self,
        client: &reqwest::Client,
        worker_url: &str,
        route: &str,
    ) -> HttpResponse {
        match client.get(format!("{}{}", worker_url, route)).send().await {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                    Err(e) => HttpResponse::InternalServerError()
                        .body(format!("Failed to read response body: {}", e)),
                }
            }
            Err(e) => HttpResponse::InternalServerError().body(format!(
                "Failed to send request to worker {}: {}",
                worker_url, e
            )),
        }
    }

    pub async fn route_to_first(&self, client: &reqwest::Client, route: &str) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            match self.select_first_worker() {
                Ok(worker_url) => {
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                        }

                        let response = self.send_request(client, &worker_url, route).await;

                        if response.status().is_success() {
                            return response;
                        }

                        warn!(
                            "Request to {} failed (attempt {}/{})",
                            worker_url,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", worker_url);
                            self.remove_worker(&worker_url);
                            break;
                        }
                    }
                }
                Err(e) => return HttpResponse::InternalServerError().body(e),
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    fn get_text_from_request(&self, body: &Bytes, route: &str) -> String {
        // convert body to json
        let json = serde_json::from_slice::<serde_json::Value>(body).unwrap();

        if route == "generate" {
            // get the "text" field
            let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
            return text.to_string();
        } else if route == "v1/chat/completions" {
            // get the messages field as raw text
            if let Some(messages) = json.get("messages") {
                // Convert messages back to a string, preserving all JSON formatting
                return serde_json::to_string(messages).unwrap_or_default();
            }
        } else if route == "v1/completions" {
            let prompt = json.get("prompt").and_then(|t| t.as_str()).unwrap_or("");
            return prompt.to_string();
        }

        return "".to_string();
    }

    // TODO: return Result<String, String> instead of panicking
    fn select_generate_worker(&self, body: &Bytes, route: &str) -> String {
        let text = self.get_text_from_request(&body, route);

        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.read().unwrap().len()),
                    )
                    .unwrap();
                worker_urls.read().unwrap()[idx].clone()
            }

            Router::Random { worker_urls } => worker_urls.read().unwrap()
                [rand::random::<usize>() % worker_urls.read().unwrap().len()]
            .clone(),

            Router::CacheAware {
                worker_urls,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                ..
            } => {
                // TODO: delay scheduling if cache hit rate is high because it may cause imbalance. prioritize low hit rate ones

                let tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Get current load statistics
                let max_load = *running_queue.values().max().unwrap_or(&0);
                let min_load = *running_queue.values().min().unwrap_or(&0);

                // Load is considered imbalanced if:
                // 1. (max - min) > abs_threshold AND
                // 2. max > rel_threshold * min
                let is_imbalanced = max_load.saturating_sub(min_load) > *balance_abs_threshold
                    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);

                let selected_url = if is_imbalanced {
                    // Log load balancing trigger and current queue state
                    info!(
                        "Load balancing triggered due to workload imbalance:\n\
                        Max load: {}, Min load: {}\n\
                        Current running queue: {:?}",
                        max_load, min_load, running_queue
                    );

                    // Use shortest queue routing when load is imbalanced
                    running_queue
                        .iter()
                        .min_by_key(|(_url, &count)| count)
                        .map(|(url, _)| url.clone())
                        .unwrap_or_else(|| worker_urls.read().unwrap()[0].clone())
                } else {
                    // Use cache-aware routing when load is balanced
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        matched_worker.to_string()
                    } else {
                        tree.get_smallest_tenant()
                    }
                };

                // Update queues and tree
                *running_queue.get_mut(&selected_url).unwrap() += 1;

                *processed_queue
                    .lock()
                    .unwrap()
                    .get_mut(&selected_url)
                    .unwrap() += 1;
                tree.insert(&text, &selected_url);

                selected_url
            }
        };

        worker_url
    }

    async fn send_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
        worker_url: &str,
    ) -> HttpResponse {
        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let res = match client
            .post(format!("{}{}", worker_url, route))
            .header(
                "Content-Type",
                req.headers()
                    .get("Content-Type")
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("application/json"),
            )
            .body(body.to_vec())
            .send()
            .await
        {
            Ok(res) => res,
            Err(_) => return HttpResponse::InternalServerError().finish(),
        };

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    HttpResponse::InternalServerError().body(error_msg)
                }
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            response
        } else if let Router::CacheAware { running_queue, .. } = self {
            let running_queue = Arc::clone(running_queue);
            let worker_url = worker_url.to_string();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            let bytes = bytes.as_ref().unwrap();
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                let mut locked_queue = running_queue.lock().unwrap();
                                let count = locked_queue.get_mut(&worker_url).unwrap();
                                *count = count.saturating_sub(1);
                                debug!("Streaming is done!!")
                            }
                        }),
                )
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }

    pub async fn route_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
    ) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            let worker_url = self.select_generate_worker(body, route);
            let mut request_retries = 0;

            // Try the same worker multiple times
            while request_retries < MAX_REQUEST_RETRIES {
                if total_retries >= 1 {
                    info!("Retrying request after {} failed attempts", total_retries);
                }
                let response = self
                    .send_generate_request(client, req, body, route, &worker_url)
                    .await;

                if response.status().is_success() {
                    return response;
                }

                warn!(
                    "Generate request to {} failed (attempt {}/{})",
                    worker_url,
                    request_retries + 1,
                    MAX_REQUEST_RETRIES
                );

                request_retries += 1;
                total_retries += 1;

                if request_retries == MAX_REQUEST_RETRIES {
                    warn!("Removing failed worker: {}", worker_url);
                    self.remove_worker(&worker_url);
                    break;
                }
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let interval_secs = 10; // check every 10 seconds
        let timeout_secs = 300; // 5 minutes

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy",
                    timeout_secs, worker_url
                ));
            }

            match client.get(&format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        match self {
                            Router::RoundRobin { worker_urls, .. }
                            | Router::Random { worker_urls }
                            | Router::CacheAware { worker_urls, .. } => {
                                info!("Worker {} health check passed", worker_url);
                                let mut urls = worker_urls.write().unwrap();
                                if urls.contains(&worker_url.to_string()) {
                                    return Err(format!("Worker {} already exists", worker_url));
                                }
                                info!("Added worker: {}", worker_url);
                                urls.push(worker_url.to_string());
                            }
                        }

                        // If cache aware, initialize the queues for the new worker
                        if let Router::CacheAware {
                            running_queue,
                            processed_queue,
                            tree,
                            ..
                        } = self
                        {
                            // Add worker to running queue with initial count of 0
                            running_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to processed queue with initial count of 0
                            processed_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to tree
                            tree.lock().unwrap().insert(&"".to_string(), &worker_url);
                        }

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        info!(
                            "Worker {} health check is pending with status: {}.",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                }
                Err(e) => {
                    info!(
                        "Worker {} health check is pending with error: {}",
                        worker_url, e
                    );

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls }
            | Router::CacheAware { worker_urls, .. } => {
                let mut urls = worker_urls.write().unwrap();
                if let Some(index) = urls.iter().position(|url| url == &worker_url) {
                    urls.remove(index);
                    info!("Removed worker: {}", worker_url);
                } else {
                    warn!("Worker {} not found, skipping removal", worker_url);
                    return;
                }
            }
        }

        // if cache aware, remove the worker from the tree
        if let Router::CacheAware {
            tree,
            running_queue,
            processed_queue,
            ..
        } = self
        {
            tree.lock().unwrap().remove_tenant(&worker_url);
            running_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            processed_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            info!(
                "Removed worker from tree and cleaned up queues: {}",
                worker_url
            );
        }
    }
}
