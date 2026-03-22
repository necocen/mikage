/// tracing ログを初期化する。
/// Native: tracing-subscriber (RUST_LOG 環境変数で制御)
/// WASM: tracing-web (console に出力、GC 負荷が軽い)
pub fn init_logging() {
    #[cfg(not(target_family = "wasm"))]
    {
        use tracing_subscriber::EnvFilter;
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .init();
    }

    #[cfg(target_family = "wasm")]
    {
        console_error_panic_hook::set_once();

        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .without_time();

        tracing_subscriber::registry()
            .with(fmt_layer.with_writer(tracing_web::MakeWebConsoleWriter::new()))
            .init();
    }
}
