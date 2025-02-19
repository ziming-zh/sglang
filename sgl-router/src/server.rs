use crate::router::PolicyConfig;
use crate::router::Router;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use bytes::Bytes;
use env_logger::Builder;
use log::{info, LevelFilter};
use std::collections::HashMap;
use std::io::Write;

#[derive(Debug)]
pub struct AppState {
    router: Router,
    client: reqwest::Client,
}

impl AppState {
    pub fn new(
        worker_urls: Vec<String>,
        client: reqwest::Client,
        policy_config: PolicyConfig,
    ) -> Self {
        // Create router based on policy
        let router = match Router::new(worker_urls, policy_config) {
            Ok(router) => router,
            Err(error) => panic!("Failed to create router: {}", error),
        };

        Self { router, client }
    }
}

#[get("/health")]
async fn health(data: web::Data<AppState>) -> impl Responder {
    data.router.route_to_first(&data.client, "/health").await
}

#[get("/health_generate")]
async fn health_generate(data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health_generate")
        .await
}

#[get("/get_server_info")]
async fn get_server_info(data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_server_info")
        .await
}

#[get("/v1/models")]
async fn v1_models(data: web::Data<AppState>) -> impl Responder {
    data.router.route_to_first(&data.client, "/v1/models").await
}

#[get("/get_model_info")]
async fn get_model_info(data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_model_info")
        .await
}

#[post("/generate")]
async fn generate(req: HttpRequest, body: Bytes, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/generate")
        .await
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/chat/completions")
        .await
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/completions")
        .await
}

#[post("/add_worker")]
async fn add_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => {
            return HttpResponse::BadRequest()
                .body("Worker URL required. Provide 'url' query parameter")
        }
    };

    match data.router.add_worker(&worker_url).await {
        Ok(message) => HttpResponse::Ok().body(message),
        Err(error) => HttpResponse::BadRequest().body(error),
    }
}

#[post("/remove_worker")]
async fn remove_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => return HttpResponse::BadRequest().finish(),
    };
    data.router.remove_worker(&worker_url);
    HttpResponse::Ok().body(format!("Successfully removed worker: {}", worker_url))
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
    pub verbose: bool,
    pub max_payload_size: usize,
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
    Builder::new()
        .format(|buf, record| {
            use chrono::Local;
            writeln!(
                buf,
                "[Router (Rust)] {} - {} - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(
            None,
            if config.verbose {
                LevelFilter::Debug
            } else {
                LevelFilter::Info
            },
        )
        .init();

    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    let app_state = web::Data::new(AppState::new(
        config.worker_urls.clone(),
        client,
        config.policy_config.clone(),
    ));

    info!("✅ Starting router on {}:{}", config.host, config.port);
    info!("✅ Serving Worker URLs: {:?}", config.worker_urls);
    info!("✅ Policy Config: {:?}", config.policy_config);
    info!(
        "✅ Max payload size: {} MB",
        config.max_payload_size / (1024 * 1024)
    );

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(web::JsonConfig::default().limit(config.max_payload_size))
            .app_data(web::PayloadConfig::default().limit(config.max_payload_size))
            .service(generate)
            .service(v1_chat_completions)
            .service(v1_completions)
            .service(v1_models)
            .service(get_model_info)
            .service(health)
            .service(health_generate)
            .service(get_server_info)
            .service(add_worker)
            .service(remove_worker)
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
